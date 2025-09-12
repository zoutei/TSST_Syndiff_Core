"""
Modern Sliding Window Pipeline Implementation

Refactored to use a high-throughput, memory-efficient, four-stage pipeline.
This version uses parallel upstream workers for data ingestion and preprocessing,
feeding a single, sequential assembler that processes one projection at a time
to correctly manage the sliding window state.
"""

import gc
import logging
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import Process, Queue
from queue import Empty
from typing import Optional

import numpy as np
import pandas as pd
import zarr

# Import existing utilities
import band_utils
import convolution_utils
import zarr_utils
from band_utils import process_skycell_bands, remove_background
from csv_utils import find_csv_file, get_projections_from_csv, load_csv_data
from zarr_utils import load_skycell_bands_masks_and_headers

logger = logging.getLogger(__name__)

_child_processes = []

# Key constants from the specification
CELL_OVERLAP = 480
EDGE_EXCLUSION = 10
EFFECTIVE_OVERLAP = CELL_OVERLAP - EDGE_EXCLUSION
PAD_SIZE = 480
GATHER_TIMEOUT_SECONDS = 300  # 5 minutes


# --- Data Structures (Retained) ---
@dataclass
class MasterArrayConfig:
    """Configuration for master arrays."""

    width: int
    height: int
    cell_width: int
    cell_height: int
    max_cells: int
    starting_x: int
    cell_dimensions: dict


@dataclass
class ProcessingState:
    """Current processing state for sliding window."""

    current_array: Optional[np.ndarray] = None
    next_array: Optional[np.ndarray] = None
    current_masks: dict[str, np.ndarray] = None
    next_masks: dict[str, np.ndarray] = None
    current_row_id: Optional[int] = None
    next_row_id: Optional[int] = None
    cell_locations: dict[str, tuple[int, int, int, int]] = None
    next_cell_locations: dict[str, tuple[int, int, int, int]] = None

    def __post_init__(self):
        if self.current_masks is None:
            self.current_masks = {}
        if self.next_masks is None:
            self.next_masks = {}
        if self.cell_locations is None:
            self.cell_locations = {}
        if self.next_cell_locations is None:
            self.next_cell_locations = {}


# --- Core Logic Functions (Retained and Modified) ---


def initialize_processing_state(config: MasterArrayConfig) -> ProcessingState:
    """Initialize processing state with empty master arrays."""
    current_array = np.full((config.height, config.width), np.nan, dtype=np.float32)
    next_array = np.full((config.height, config.width), np.nan, dtype=np.float32)
    return ProcessingState(current_array=current_array, next_array=next_array)


def advance_sliding_window(state: ProcessingState) -> None:
    """Advance the sliding window (next becomes current)."""
    state.current_array, state.next_array = state.next_array, state.current_array
    state.next_array.fill(np.nan)
    state.current_masks, state.next_masks = state.next_masks, {}
    state.current_row_id = state.next_row_id
    state.next_row_id = None
    state.cell_locations.clear()
    state.cell_locations.update(state.next_cell_locations)
    state.next_cell_locations.clear()
    gc.collect()


def apply_cross_row_padding(state: ProcessingState, config: MasterArrayConfig) -> None:
    """Apply cross-row padding between current and next arrays with correct overlap handling."""
    if state.next_array is None:
        return

    # Source region from next row
    next_source_y_start = PAD_SIZE + CELL_OVERLAP - EDGE_EXCLUSION
    next_source_y_end = PAD_SIZE * 2 + CELL_OVERLAP

    # Target region in current row (top padding area)
    current_target_y_start = config.cell_height - EDGE_EXCLUSION + PAD_SIZE
    state.current_array[current_target_y_start:, :] = state.next_array[next_source_y_start:next_source_y_end, :]

    # Opposite direction: from current to next
    current_source_y_start = config.cell_height - CELL_OVERLAP
    current_source_y_end = PAD_SIZE + config.cell_height - CELL_OVERLAP + EDGE_EXCLUSION
    next_target_y_end = PAD_SIZE + EDGE_EXCLUSION
    state.next_array[:next_target_y_end, :] = state.current_array[current_source_y_start:current_source_y_end, :]


def extract_cell_results(convolved_array: np.ndarray, cell_positions: dict) -> dict[str, np.ndarray]:
    """Extract individual cell results from convolved array."""
    results = {}
    for cell_name, (x_start, x_end, y_start, y_end) in cell_positions.items():
        results[cell_name] = convolved_array[y_start:y_end, x_start:x_end].copy()
    return results


def extract_projection_metadata(df: pd.DataFrame, projection: str) -> dict:
    """Extract projection metadata from a pre-loaded DataFrame."""
    proj_df = df[df["projection"].astype(str) == projection]
    if proj_df.empty:
        raise ValueError(f"No data found for projection {projection}")

    rows = {}
    cell_dimensions = {}
    starting_x = 10
    for _, row in proj_df.iterrows():
        row_id = int(row["y"])
        cell_name = row["NAME"]
        x_coord = int(row["x"])
        starting_x = x_coord if x_coord < starting_x else starting_x
        cell_width = int(row.get("NAXIS1"))
        cell_height = int(row.get("NAXIS2"))
        cell_dimensions[cell_name] = (cell_width, cell_height)
        if row_id not in rows:
            rows[row_id] = []
        rows[row_id].append((cell_name, x_coord))

    for row_id in rows:
        rows[row_id].sort(key=lambda x: x[1])

    all_dims = list(cell_dimensions.values())
    if len(set(all_dims)) > 1:
        logger.warning(f"Inconsistent cell dimensions found: {set(all_dims)}")
    typical_width, typical_height = max(all_dims, key=lambda item: item[0] * item[1]) if all_dims else (0, 0)
    max_cells_per_row = max(len(cells) for cells in rows.values()) if rows else 0

    return {
        "projection": projection,
        "rows": rows,
        "cell_width": typical_width,
        "cell_height": typical_height,
        "max_cells_per_row": max_cells_per_row,
        "starting_x": starting_x,
        "cell_dimensions": cell_dimensions,
        "dataframe": proj_df,
    }


def create_master_array_config(metadata: dict) -> MasterArrayConfig:
    """Create master array configuration from metadata."""
    cell_width = metadata["cell_width"]
    cell_height = metadata["cell_height"]
    max_cells = metadata["max_cells_per_row"]
    starting_x = metadata["starting_x"]
    master_width = PAD_SIZE + (max_cells * (cell_width - CELL_OVERLAP)) + CELL_OVERLAP + PAD_SIZE
    master_height = cell_height + (2 * PAD_SIZE)
    return MasterArrayConfig(width=master_width, height=master_height, cell_width=cell_width, cell_height=cell_height, max_cells=max_cells, starting_x=starting_x, cell_dimensions=metadata["cell_dimensions"])


def create_master_task_list(df: pd.DataFrame, projection: str) -> tuple[dict, list[tuple[str, str, int]]]:
    """Generate the task list for a single projection from a pre-loaded DataFrame."""
    metadata = extract_projection_metadata(df, projection)
    task_list = []
    for row_id in sorted(metadata["rows"].keys()):
        for skycell_id in metadata["rows"][row_id]:
            task_list.append((skycell_id, projection, row_id))
    return metadata, task_list


# --- NEW Pipeline Worker Functions ---


def process_single_cell(raw_bundle: dict) -> dict:
    """
    Standalone function for processing a single cell in a separate process.
    This function replaces the pre_processor_worker for ProcessPoolExecutor.
    """
    try:
        import logging

        from band_utils import process_skycell_bands, remove_background

        # Set up logging for this process
        logger = logging.getLogger(__name__)

        logger.info(f"[PreProcessor] Starting {raw_bundle['skycell_id']}")
        combined_image, combined_mask, combined_uncert = process_skycell_bands(bands_data=raw_bundle["bands_data"], masks_data=raw_bundle["masks_data"], weights_data=raw_bundle["weights_data"], headers_data=raw_bundle["headers_data"], headers_weight_data=raw_bundle["headers_weight_data"])

        logger.info(f"[PreProcessor] Starting Source Extractor {raw_bundle['skycell_id']}")
        combined_image = remove_background(combined_image, combined_uncert)

        # Return the processed bundle
        combined_bundle = {"skycell_id": raw_bundle["skycell_id"], "projection": raw_bundle["projection"], "row_id": raw_bundle["row_id"], "x_coord": raw_bundle["x_coord"], "combined_image": combined_image, "combined_mask": combined_mask}

        logger.info(f"[PreProcessor] Processed {raw_bundle['skycell_id']}")
        return combined_bundle

    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"[PreProcessor] Failed for {raw_bundle['skycell_id']}: {e}", exc_info=True)
        return None


def reader_worker(task_queue: Queue, raw_cell_queue: Queue, zarr_store):
    """Stage 1: Reads raw cell data from Zarr based on tasks."""
    while True:
        task = task_queue.get()
        if task is None:
            break
        skycell_id, projection, row_id, x_coord = task
        try:
            bands, masks, weights, headers, headers_weight = zarr_utils.load_skycell_bands_masks_and_headers(zarr_store, projection, skycell_id)
            if not bands:
                logger.warning(f"[Reader] No band data for {skycell_id}, skipping.")
                continue
            # SPEC: Pass all metadata, including x_coord, to the next stage.
            raw_bundle = {"skycell_id": skycell_id, "projection": projection, "row_id": row_id, "x_coord": x_coord, "bands_data": bands, "masks_data": masks, "headers_data": headers, "weights_data": weights, "headers_weight_data": headers_weight}
            raw_cell_queue.put(raw_bundle)
            logger.info(f"[Reader] Loaded {skycell_id}")
        except Exception as e:
            logger.error(f"[Reader] Failed to load {skycell_id}: {e}", exc_info=True)


def pre_processor_worker(raw_cell_queue: Queue, combined_cell_queue: Queue):
    """
    DEPRECATED: Stage 2 worker for ThreadPoolExecutor (kept for compatibility).
    Use process_single_cell with ProcessPoolExecutor instead.
    """
    while True:
        raw_bundle = raw_cell_queue.get()
        if raw_bundle is None:
            break
        try:
            logger.info(f"[PreProcessor] Starting {raw_bundle['skycell_id']}")
            combined_image, combined_mask, combined_uncert = band_utils.process_skycell_bands(bands_data=raw_bundle["bands_data"], masks_data=raw_bundle["masks_data"], weights_data=raw_bundle["weights_data"], headers_data=raw_bundle["headers_data"], headers_weight_data=raw_bundle["headers_weight_data"])
            logger.info(f"[PreProcessor] Starting Source Extractor {raw_bundle['skycell_id']}")
            combined_image = remove_background(combined_image, combined_uncert)
            # SPEC: Pass essential metadata through to the assembler stage.
            combined_bundle = {"skycell_id": raw_bundle["skycell_id"], "projection": raw_bundle["projection"], "row_id": raw_bundle["row_id"], "x_coord": raw_bundle["x_coord"], "combined_image": combined_image, "combined_mask": combined_mask}
            combined_cell_queue.put(combined_bundle)
            logger.info(f"[PreProcessor] Processed {raw_bundle['skycell_id']}")
        except Exception as e:
            logger.error(f"[PreProcessor] Failed for {raw_bundle['skycell_id']}: {e}", exc_info=True)


def process_coordinator(raw_cell_queue: Queue, combined_cell_queue: Queue, num_workers: int = 4):
    """
    Coordinates between raw cell queue and ProcessPoolExecutor for preprocessing.
    This function runs in a separate thread and bridges the queue-based system
    with the process-based preprocessing.
    """
    logger.info(f"[ProcessCoordinator] Starting with {num_workers} process workers")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Track submitted futures
        futures = {}
        active_tasks = set()

        while True:
            try:
                # Get raw bundle from queue (non-blocking check)
                try:
                    raw_bundle = raw_cell_queue.get(timeout=1.0)
                except Exception:
                    # Timeout or other queue error - continue the loop
                    continue

                if raw_bundle is None:
                    logger.info("[ProcessCoordinator] Received shutdown signal")
                    break

                # Submit to process pool
                future = executor.submit(process_single_cell, raw_bundle)
                futures[future] = raw_bundle["skycell_id"]
                active_tasks.add(future)

                # Check for completed tasks
                completed = set()
                for future in list(active_tasks):
                    if future.done():
                        completed.add(future)
                        try:
                            result = future.result()
                            if result is not None:
                                combined_cell_queue.put(result)
                            else:
                                logger.warning(f"[ProcessCoordinator] Got None result for {futures[future]}")
                        except Exception as e:
                            logger.error(f"[ProcessCoordinator] Process failed for {futures[future]}: {e}")

                        # Clean up
                        del futures[future]

                active_tasks -= completed

            except Exception as e:
                logger.error(f"[ProcessCoordinator] Error in coordination loop: {e}", exc_info=True)
                break

        # Wait for remaining tasks
        logger.info(f"[ProcessCoordinator] Waiting for {len(active_tasks)} remaining tasks")
        for future in active_tasks:
            try:
                result = future.result(timeout=30)  # 30 second timeout per task
                if result is not None:
                    combined_cell_queue.put(result)
            except Exception as e:
                logger.error(f"[ProcessCoordinator] Final task failed for {futures.get(future, 'unknown')}: {e}")

    logger.info("[ProcessCoordinator] Finished")


def saver_worker(results_queue: Queue, output_path: str):
    """Stage 4: Saves final results to an output Zarr store."""
    try:
        output_store = zarr.open(output_path, mode="a")
        logger.info(f"[Saver] Opened output store {output_path}")
        while True:
            processed_bundle = results_queue.get()
            if processed_bundle is None:
                break
            try:
                zarr_utils.save_convolved_results(output_store, processed_bundle["projection"], processed_bundle["row_id"], processed_bundle["results_data"], processed_bundle["results_masks"])
                logger.info(f"[Saver] Saved row {processed_bundle['row_id']} for projection {processed_bundle['projection']}")
            except Exception as e:
                logger.error(f"[Saver] Failed saving row {processed_bundle['row_id']}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"[Saver] Failed opening output store at {output_path}: {e}", exc_info=True)


# --- NEW: Sequential Assembler & Convolution (Stage 3) ---


def assemble_row_from_bundles(target_array: np.ndarray, cell_bundles: list[dict], config: MasterArrayConfig) -> tuple[dict, dict]:
    """
    SPEC: Assembles a master row image from a pre-gathered list of cell bundles.
    This function is purely for image assembly and does not interact with queues.
    """
    target_array.fill(np.nan)
    cell_positions = {}
    cell_masks = {}
    first_x_coord = config.starting_x

    for bundle in cell_bundles:
        cell_name = bundle["skycell_id"]
        image = bundle["combined_image"]
        mask = bundle["combined_mask"]
        x_coord = bundle["x_coord"]
        cell_index = x_coord - first_x_coord

        target_y_start = PAD_SIZE
        target_x_start_full = PAD_SIZE + cell_index * (config.cell_width - CELL_OVERLAP)

        if cell_index == 0:
            source_x_start = 0
            target_x_start = target_x_start_full
        else:
            source_x_start = EFFECTIVE_OVERLAP
            target_x_start = target_x_start_full + EFFECTIVE_OVERLAP

        source_height, source_width = image.shape
        place_width = source_width - source_x_start
        place_height = source_height

        target_x_end = target_x_start + place_width
        target_y_end = target_y_start + place_height

        if target_x_end <= target_array.shape[1] and target_y_end <= target_array.shape[0]:
            target_array[target_y_start:target_y_end, target_x_start:target_x_end] = image[:, source_x_start:]
            cell_masks[cell_name] = mask
            cell_positions[cell_name] = (target_x_start_full, target_x_start_full + config.cell_width, PAD_SIZE, PAD_SIZE + config.cell_height)
        else:
            logger.warning(f"Cell {cell_name} out of bounds for master array. Skipping placement.")

    logger.info(f"Assembled row with {len(cell_bundles)} cells.")
    return cell_positions, cell_masks


def _gather_cells_for_row(projection: str, row_id: int, metadata: dict, combined_cell_queue: Queue, cell_buffer: dict, zarr_path: str) -> list[dict]:
    """
    Gathers all necessary cell bundles for a given row from the queue.
    Handles out-of-order arrivals using a buffer and has a timeout mechanism
    to manually load/process cells if they don't arrive in time.
    """
    expected_cells = metadata["rows"].get(row_id, [])
    num_to_expect = len(expected_cells)
    if num_to_expect == 0:
        return []

    # Check buffer first for any cells that have already arrived
    key = (projection, row_id)
    gathered_bundles = cell_buffer.pop(key, [])

    last_arrival_time = time.time()
    while len(gathered_bundles) < num_to_expect:
        remaining_time = GATHER_TIMEOUT_SECONDS - (time.time() - last_arrival_time)
        if remaining_time <= 0:
            logger.warning(f"Timeout waiting for cells for P:{projection} R:{row_id}. Manually loading missing cells.")
            break  # Exit loop to proceed with manual loading

        try:
            # Wait for the next cell with a calculated timeout
            bundle = combined_cell_queue.get(timeout=remaining_time)
            if bundle is None:
                logger.warning("Shutdown signal received while gathering cells.")
                break

            last_arrival_time = time.time()  # Reset timer on successful arrival
            bundle_key = (bundle["projection"], bundle["row_id"])

            if bundle_key == key:
                gathered_bundles.append(bundle)
            else:
                # It's for a future row/projection, so buffer it
                cell_buffer.setdefault(bundle_key, []).append(bundle)

        except Empty:
            logger.warning(f"Timeout hit via queue.get() for P:{projection} R:{row_id}. Manually loading.")
            break  # Exit loop to proceed with manual loading

    # If, after waiting, cells are still missing, load them manually
    if len(gathered_bundles) < num_to_expect:
        received_cell_names = {b["skycell_id"] for b in gathered_bundles}
        expected_cell_info = {name: x for name, x in expected_cells}
        missing_cell_names = set(expected_cell_info.keys()) - received_cell_names

        logger.info(f"Identified {len(missing_cell_names)} missing cells for P:{projection} R:{row_id}.")
        for cell_name in missing_cell_names:
            try:
                logger.info(f"Manually loading and processing missing cell: {cell_name}")
                zarr_store = zarr.open(zarr_path, mode="r")
                bands_data, masks_data, weights_data, headers_data, headers_weight_data = load_skycell_bands_masks_and_headers(zarr_store, projection, cell_name)
                combined_image, combined_mask, combined_uncert = process_skycell_bands(bands_data, masks_data, weights_data, headers_data, headers_weight_data)
                combined_image = remove_background(combined_image, combined_uncert)

                manual_bundle = {
                    "skycell_id": cell_name,
                    "projection": projection,
                    "row_id": row_id,
                    "x_coord": expected_cell_info[cell_name],
                    "combined_image": combined_image,
                    "combined_mask": combined_mask,
                    # "combined_uncert": combined_uncert,
                }
                gathered_bundles.append(manual_bundle)
            except Exception as e:
                logger.error(f"Failed to manually load/process {cell_name}: {e}", exc_info=True)

    return gathered_bundles


def process_row_step_from_queue(state: ProcessingState, config: MasterArrayConfig, metadata: dict, current_row_id: int, next_row_id: Optional[int], combined_cell_queue: Queue, cell_buffer: dict, psf_sigma: float, zarr_path: str, projection: str) -> tuple[dict, dict]:
    """
    Encapsulates the logic for processing a single row step in the sliding window.
    It loads necessary data, applies padding, performs convolution, and extracts results.
    """
    # 1. Load the Current Row (Only If Necessary)
    if state.current_row_id != current_row_id:
        logger.info(f"Loading initial current row ID {current_row_id}")
        current_row_bundles = _gather_cells_for_row(projection, current_row_id, metadata, combined_cell_queue, cell_buffer, zarr_path)
        positions, masks = assemble_row_from_bundles(state.current_array, current_row_bundles, config)
        state.cell_locations.update(positions)
        state.current_masks.update(masks)
        state.current_row_id = current_row_id
        logger.info(f"Built current row ID {current_row_id} with {len(positions)} cells.")

    # 2. Load the Next Row (Always)
    if next_row_id is not None:
        logger.info(f"Preparing next row ID {next_row_id}")
        next_row_bundles = _gather_cells_for_row(projection, next_row_id, metadata, combined_cell_queue, cell_buffer, zarr_path)
        positions, masks = assemble_row_from_bundles(state.next_array, next_row_bundles, config)
        state.next_cell_locations.update(positions)
        state.next_masks.update(masks)
        state.next_row_id = next_row_id
        logger.info(f"Prepared next row ID {next_row_id} with {len(state.next_cell_locations)} cells.")
    else:
        # Clear next state if there is no next row
        state.next_array.fill(np.nan)
        state.next_cell_locations.clear()
        state.next_masks.clear()
        state.next_row_id = None
        logger.info("No next row to prepare.")

    # 3. Apply Cross-Row Padding
    apply_cross_row_padding(state, config)

    # 4. Perform Convolution
    nan_mask = np.isnan(state.current_array)
    logger.info(f"Applying convolution for row ID {current_row_id}")
    convolved_array = convolution_utils.apply_gaussian_convolution(state.current_array, sigma=psf_sigma)
    # Restore NaNs on the result, not the state array which will be replaced
    convolved_array[nan_mask] = np.nan

    # 5. Extract and Return Results
    results_data = extract_cell_results(convolved_array, state.cell_locations)
    results_masks = {name: mask for name, mask in state.current_masks.items() if name in state.cell_locations}

    return results_data, results_masks


def sequential_processor(projections: list[str], df: pd.DataFrame, combined_cell_queue: Queue, results_queue: Queue, psf_sigma: float, zarr_path: str):
    """
    SPEC: This is Stage 3. It iterates through projections sequentially,
    calling a helper function to process each row, queues results, and manages
    the sliding window state.
    """
    for projection in projections:
        logger.info(f"--- Starting sequential processing for projection: {projection} ---")
        # Initialize buffer once per projection
        cell_buffer = {}
        try:
            metadata = extract_projection_metadata(df, projection)
            config = create_master_array_config(metadata)
            state = initialize_processing_state(config)
            row_ids = sorted(metadata["rows"].keys())
        except Exception as e:
            logger.error(f"Failed to initialize projection {projection}: {e}. Skipping.")
            # Attempt to drain the queue of cells for this failed projection
            num_cells_to_skip = sum(len(cells) for _, cells in metadata.get("rows", {}).items())
            for _ in range(num_cells_to_skip):
                try:
                    combined_cell_queue.get(timeout=1)
                except Empty:
                    break
            continue

        # Inner Loop: Process each row
        for i, current_row_id in enumerate(row_ids):
            logger.info(f"--- Processing step for row {i + 1}/{len(row_ids)}: ROW ID {current_row_id} ---")

            # Determine Next Row
            next_row_id = row_ids[i + 1] if i + 1 < len(row_ids) else None

            # Call the Helper Function to do the heavy lifting
            results_data, results_masks = process_row_step_from_queue(state, config, metadata, current_row_id, next_row_id, combined_cell_queue, cell_buffer, psf_sigma, zarr_path, projection)

            # Queue the Results
            processed_bundle = {"projection": projection, "row_id": current_row_id, "results_data": results_data, "results_masks": results_masks}
            results_queue.put(processed_bundle)
            logger.info(f"Finished processing and queued results for row {current_row_id}")

            # Advance the Window if not the last row
            if next_row_id is not None:
                advance_sliding_window(state)

        logger.info(f"--- Finished sequential processing for projection: {projection} ---")

    # Shutdown Signal for the saver
    results_queue.put(None)


# --- Main Orchestrator ---


def run_modern_sliding_window_pipeline(sector: int, camera: int, ccd: int, data_root: str = "data", projections_limit: Optional[int] = None, psf_sigma: float = 60.0):
    """The top-level master orchestrator for the entire pipeline."""
    global _child_processes
    _child_processes.clear()
    signal.signal(signal.SIGINT, shutdown_handler)

    logger.info(f"Starting pipeline for sector {sector}, camera {camera}, ccd {ccd}")
    zarr_path = f"{data_root}/ps1_skycells_zarr/ps1_skycells.zarr"
    output_path = f"{data_root}/convolved_results/sector_{sector:04d}_camera_{camera}_ccd_{ccd}.zarr"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        csv_path = find_csv_file(data_root, sector, camera, ccd)
        projections = get_projections_from_csv(csv_path)
        if projections_limit:
            projections = projections[:projections_limit]
        df = load_csv_data(csv_path)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return {"error": str(e)}

    # --- Setup ---
    zarr_store = zarr.open(zarr_path, mode="r")
    num_pre_processors = 50
    num_readers = 2

    task_queue = Queue()
    raw_cell_queue = Queue(maxsize=20)
    combined_cell_queue = Queue(maxsize=30)
    results_queue = Queue(maxsize=3)

    # --- Start Persistent Workers (Stages 1, 4) ---
    saver_proc = Process(target=saver_worker, args=(results_queue, output_path))
    saver_proc.start()

    # Start the process coordinator in a separate thread
    import threading

    process_coordinator_thread = threading.Thread(target=process_coordinator, args=(raw_cell_queue, combined_cell_queue, num_pre_processors), daemon=True)
    process_coordinator_thread.start()

    with ThreadPoolExecutor(max_workers=num_readers) as reader_executor:
        # Start Stage 1 workers (readers only - preprocessing now handled by process coordinator)
        for _ in range(num_readers):
            reader_executor.submit(reader_worker, task_queue, raw_cell_queue, zarr_store)

        # --- Dispatch All Tasks ---
        logger.info(f"Dispatching tasks for {len(projections)} projections.")
        master_task_list = []
        for projection in projections:
            try:
                metadata = extract_projection_metadata(df, projection)
                for row_id, cells in sorted(metadata["rows"].items()):
                    for cell_name, x_coord in cells:
                        master_task_list.append((cell_name, projection, row_id, x_coord))
            except Exception as e:
                logger.error(f"Failed to create tasks for projection {projection}: {e}")

        for task in master_task_list:
            task_queue.put(task)
        logger.info("All tasks have been dispatched to the reader workers.")

        # --- Signal Reader Shutdown ---
        for _ in range(num_readers):
            task_queue.put(None)

        # --- Run Sequential Processor (Stage 3) in Main Thread ---
        sequential_processor(projections, df, combined_cell_queue, results_queue, psf_sigma, zarr_path)

        # --- Final Shutdown Sequence ---
        logger.info("Sequential processor finished. Shutting down upstream workers.")
        # Signal the process coordinator to shutdown
        raw_cell_queue.put(None)

        # Wait for process coordinator thread to finish
        if process_coordinator_thread.is_alive():
            logger.info("Waiting for process coordinator to finish...")
            process_coordinator_thread.join(timeout=30)

    saver_proc.join()
    logger.info("Pipeline completed successfully!")
    return {"status": "success"}


def shutdown_handler(signum, frame):
    """
    Handles SIGINT (Ctrl+C) to ensure all child processes are terminated.
    """
    logger.warning("Ctrl+C detected! Initiating graceful shutdown...")
    for p in _child_processes:
        if p.is_alive():
            logger.info(f"Terminating process: {p.name} (PID: {p.pid})")
            p.terminate()
            p.join()
    logger.info("All child processes terminated. Exiting.")
    sys.exit(1)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Modern Sliding Window PS1 Processing Pipeline")
    parser.add_argument("sector", type=int, help="TESS sector number")
    parser.add_argument("camera", type=int, help="TESS camera number")
    parser.add_argument("ccd", type=int, help="TESS CCD number")
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument("--limit", type=int, help="Limit projections for testing")
    parser.add_argument("--psf-sigma", type=float, default=40.0, help="PSF sigma for convolution")
    args = parser.parse_args()
    results = run_modern_sliding_window_pipeline(args.sector, args.camera, args.ccd, data_root=args.data_root, projections_limit=args.limit, psf_sigma=args.psf_sigma)

    if results.get("status") == "success":
        print("\n✅ Modern sliding window pipeline completed successfully!")
    else:
        print(f"\n❌ Pipeline failed: {results.get('error', 'Unknown error')}")
