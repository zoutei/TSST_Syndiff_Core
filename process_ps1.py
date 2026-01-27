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
from correct_saturation import apply_saturation_to_row
from csv_utils import find_csv_file, get_projections_from_csv, load_csv_data
from zarr_utils import load_skycell_bands_masks_and_headers
from cross_projection_padding import apply_cross_projection_padding

logger = logging.getLogger(__name__)

_child_processes = []

# Key constants from the specification
CELL_OVERLAP = 480
EDGE_EXCLUSION = 10
EFFECTIVE_OVERLAP = CELL_OVERLAP - EDGE_EXCLUSION
PAD_SIZE = 480

GATHER_TIMEOUT_SECONDS = 180
MAX_ACTIVE_TASKS = 35  # Maximum concurrent preprocessing tasks
MAX_TOTAL_PENDING_WORK = 30  # Maximum total pending work (queue + buffer + active tasks)
QUEUE_CHECK_INTERVAL = 20  # Log queue state every N iterations


def calculate_total_buffer_size(cell_buffer: dict) -> int:
    """Calculate total number of cells in the buffer across all projection/row combinations."""
    return sum(len(cells) for cells in cell_buffer.values())


def load_gaia_catalog(data_root: str, sector: int, camera: int, ccd: int, catalog_path: Optional[str] = None) -> pd.DataFrame:
    """Load Gaia catalog for the specified sector/camera/ccd."""
    if catalog_path is None:
        catalog_dir = os.path.join(data_root, "catalogs", f"sector_{sector:04d}", f"camera_{camera}", f"ccd_{ccd}")
        catalog_path = f"{catalog_dir}/gaia_catalog_s{sector:04d}_{camera}_{ccd}.csv"

    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    logger.info(f"Loading catalog from {catalog_path}")
    catalog = pd.read_csv(catalog_path)

    # Validate required columns
    required_cols = ["ra", "dec", "phot_rp_mean_mag"]
    missing_cols = [col for col in required_cols if col not in catalog.columns]
    if missing_cols:
        raise ValueError(f"Catalog missing required columns: {missing_cols}")

    logger.info(f"Loaded {len(catalog)} stars from catalog")
    return catalog


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
        logger.warning(f"[Metadata] Inconsistent cell dimensions found: {set(all_dims)}")
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
        combined_bundle = {"skycell_id": raw_bundle["skycell_id"], "projection": raw_bundle["projection"], "row_id": raw_bundle["row_id"], "x_coord": raw_bundle["x_coord"], "combined_image": combined_image, "combined_mask": combined_mask, "headers_data": raw_bundle["headers_data"]}

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
            combined_bundle = {"skycell_id": raw_bundle["skycell_id"], "projection": raw_bundle["projection"], "row_id": raw_bundle["row_id"], "x_coord": raw_bundle["x_coord"], "combined_image": combined_image, "combined_mask": combined_mask, "headers_data": raw_bundle["headers_data"]}
            combined_cell_queue.put(combined_bundle)
            logger.info(f"[PreProcessor] Processed {raw_bundle['skycell_id']}")
        except Exception as e:
            logger.error(f"[PreProcessor] Failed for {raw_bundle['skycell_id']}: {e}", exc_info=True)


def process_coordinator(raw_cell_queue: Queue, combined_cell_queue: Queue, cell_buffer: dict, num_workers: int = 4):
    """
    Coordinates between raw cell queue and ProcessPoolExecutor for preprocessing.
    This function runs in a separate thread and bridges the queue-based system
    with the process-based preprocessing.
    """
    logger.info(f"[ProcessCoordinator] Starting with {num_workers} process workers")

    iteration_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Track submitted futures
        futures = {}
        active_tasks = set()

        while True:
            try:
                iteration_count += 1

                # Periodic queue monitoring
                if iteration_count % QUEUE_CHECK_INTERVAL == 0:
                    current_queue_size = combined_cell_queue.qsize()
                    current_buffer_size = calculate_total_buffer_size(cell_buffer)
                    total_pending = current_queue_size + current_buffer_size
                    logger.info(f"[ProcessCoordinator] Queue States - Raw: {raw_cell_queue.qsize()}, Combined: {current_queue_size}, Buffer: {current_buffer_size}, Active Tasks: {len(active_tasks)}, Total Pending: {total_pending}")

                # 1. ALWAYS check for completed tasks first
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

                # 2. Check Capacity (Look before you leap)
                current_queue_size = combined_cell_queue.qsize()
                current_buffer_size = calculate_total_buffer_size(cell_buffer)
                total_pending = len(active_tasks) + current_queue_size + current_buffer_size

                if total_pending >= MAX_TOTAL_PENDING_WORK:
                    # System is overloaded. Do NOT fetch new work.
                    # Just sleep briefly to avoid busy loop and continue to process completions.
                    time.sleep(0.1)
                    continue

                # 3. Fetch New Work (Only if we have capacity)
                try:
                    # Non-blocking fetch
                    raw_bundle = raw_cell_queue.get(timeout=0.1)
                except Empty:
                    # No new work, continue loop
                    continue
                except Exception:
                    continue

                if raw_bundle is None:
                    logger.info("[ProcessCoordinator] Received shutdown signal")
                    break

                # 4. Submit Work
                future = executor.submit(process_single_cell, raw_bundle)
                futures[future] = raw_bundle["skycell_id"]
                active_tasks.add(future)

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
            logger.warning(f"[Assembler] Cell {cell_name} out of bounds for master array. Skipping placement.")

    logger.info(f"[Assembler] Assembled row with {len(cell_bundles)} cells.")
    return cell_positions, cell_masks


def _gather_cells_for_row(projection: str, row_id: int, metadata: dict, combined_cell_queue: Queue, cell_buffer: dict, zarr_path: str) -> list[dict]:
    """
    Gathers all necessary cell bundles for a given row from the queue.
    Handles out-of-order arrivals using a buffer and has a timeout mechanism
    to manually load/process cells if they don't arrive in time.

    Timeout is measured from the last arrival of a cell for the same projection/row.
    """
    expected_cells = metadata["rows"].get(row_id, [])
    num_to_expect = len(expected_cells)
    if num_to_expect == 0:
        return []

    # Check buffer first for any cells that have already arrived
    key = (projection, row_id)
    gathered_bundles = cell_buffer.pop(key, [])

    logger.info(f"[Gather] Gathering {num_to_expect} cells for P:{projection} R:{row_id}")
    logger.info(f"[Gather] Expected cells: {[cell[0] for cell in expected_cells]}")
    logger.info(f"[Gather] Queue size at start: {combined_cell_queue.qsize()}")
    logger.info(f"[Gather] Already buffered: {len(gathered_bundles)} cells")

    # Track both function start time and last relevant cell arrival time
    function_start_time = time.time()
    last_relevant_arrival_time = time.time() if len(gathered_bundles) > 0 else None
    loop_count = 0

    while len(gathered_bundles) < num_to_expect:
        loop_count += 1

        # Calculate timeout from last relevant arrival, or from function start if none yet
        if last_relevant_arrival_time is not None:
            time_since_last_relevant = time.time() - last_relevant_arrival_time
            remaining_time = GATHER_TIMEOUT_SECONDS - time_since_last_relevant
        else:
            # If no relevant cells yet, calculate from function start
            time_since_function_start = time.time() - function_start_time
            remaining_time = GATHER_TIMEOUT_SECONDS - time_since_function_start

        # Periodic logging
        if loop_count % 10 == 0:
            logger.info(f"[Gather] Still gathering P:{projection} R:{row_id}, have {len(gathered_bundles)}/{num_to_expect}, queue_size: {combined_cell_queue.qsize()}, remaining_timeout: {remaining_time:.1f}s")

        if remaining_time <= 0:
            logger.warning(f"[Gather] Timeout waiting for cells for P:{projection} R:{row_id}. {GATHER_TIMEOUT_SECONDS}s since last relevant cell. Manually loading missing cells.")
            break  # Exit loop to proceed with manual loading

        try:
            # Wait for the next cell with calculated timeout
            bundle = combined_cell_queue.get(timeout=min(remaining_time, 5.0))  # Max 5 second wait per iteration
            if bundle is None:
                logger.warning("[Gather] Shutdown signal received while gathering cells.")
                break

            bundle_key = (bundle["projection"], bundle["row_id"])

            if bundle_key == key:
                # This is a cell for our target row/projection
                gathered_bundles.append(bundle)
                last_relevant_arrival_time = time.time()  # Reset timer on relevant arrival
                logger.debug(f"[Gather] Received relevant cell {bundle['skycell_id']} for P:{projection} R:{row_id}")
            else:
                # It's for a different row/projection, so buffer it
                cell_buffer.setdefault(bundle_key, []).append(bundle)
                logger.debug(f"[Gather] Buffered cell {bundle['skycell_id']} for P:{bundle['projection']} R:{bundle['row_id']}")

        except Empty:
            # Timeout on queue.get() - continue loop to check overall timeout
            continue

    # If, after waiting, cells are still missing, load them manually
    if len(gathered_bundles) < num_to_expect:
        received_cell_names = {b["skycell_id"] for b in gathered_bundles}
        expected_cell_info = {name: x for name, x in expected_cells}
        missing_cell_names = set(expected_cell_info.keys()) - received_cell_names

        logger.warning(f"[Gather] Identified {len(missing_cell_names)} missing cells for P:{projection} R:{row_id}: {missing_cell_names}")
        for cell_name in missing_cell_names:
            try:
                logger.info(f"[ManualPreProcessor] Manually loading and processing missing cell: {cell_name}")
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
                    "headers_data": headers_data,
                    # "combined_uncert": combined_uncert,
                }
                gathered_bundles.append(manual_bundle)
            except Exception as e:
                logger.error(f"[ManualPreProcessor] Failed to manually load/process {cell_name}: {e}", exc_info=True)

    logger.info(f"[Gather] Successfully gathered {len(gathered_bundles)}/{num_to_expect} cells for P:{projection} R:{row_id}")
    return gathered_bundles


def process_row_step_from_queue(
    state: ProcessingState, config: MasterArrayConfig, metadata: dict, current_row_id: int, next_row_id: Optional[int], combined_cell_queue: Queue, cell_buffer: dict, psf_sigma: float, zarr_path: str, projection: str, catalog: Optional[pd.DataFrame] = None, enable_saturation_correction: bool = True, csv_path: Optional[str] = None
) -> tuple[dict, dict]:
    """
    Encapsulates the logic for processing a single row step in the sliding window.
    It loads necessary data, applies padding, performs convolution, and extracts results.
    """
    # 1. Load the Current Row (Only If Necessary)
    if state.current_row_id != current_row_id:
        logger.info(f"[SequentialProcessor] Loading initial current row ID {current_row_id}")
        current_row_bundles = _gather_cells_for_row(projection, current_row_id, metadata, combined_cell_queue, cell_buffer, zarr_path)
        positions, masks = assemble_row_from_bundles(state.current_array, current_row_bundles, config)
        state.cell_locations.update(positions)
        state.current_masks.update(masks)
        state.current_row_id = current_row_id
        logger.info(f"[SequentialProcessor] Built current row ID {current_row_id} with {len(positions)} cells.")

        # Apply Saturation Correction
        if enable_saturation_correction and catalog is not None:
             logger.info(f"[SequentialProcessor] Applying parallel saturation correction for current row {current_row_id}...")
             start_sat = time.time()
             apply_saturation_to_row(state.current_array, state.current_masks, state.cell_locations, current_row_bundles, catalog)
             logger.info(f"[SequentialProcessor] Saturation correction finished in {time.time() - start_sat:.2f}s")

    # 2. Load the Next Row (Always)
    if next_row_id is not None:
        logger.info(f"[SequentialProcessor] Preparing next row ID {next_row_id}")
        next_row_bundles = _gather_cells_for_row(projection, next_row_id, metadata, combined_cell_queue, cell_buffer, zarr_path)
        positions, masks = assemble_row_from_bundles(state.next_array, next_row_bundles, config)
        state.next_cell_locations.update(positions)
        state.next_masks.update(masks)
        state.next_row_id = next_row_id
        logger.info(f"[SequentialProcessor] Prepared next row ID {next_row_id} with {len(state.next_cell_locations)} cells.")

        # Apply Saturation Correction
        if enable_saturation_correction and catalog is not None:
             logger.info(f"[SequentialProcessor] Applying parallel saturation correction for next row {next_row_id}...")
             start_sat = time.time()
             apply_saturation_to_row(state.next_array, state.next_masks, state.next_cell_locations, next_row_bundles, catalog)
             logger.info(f"[SequentialProcessor] Saturation correction finished in {time.time() - start_sat:.2f}s")

    else:
        # Clear next state if there is no next row
        state.next_array.fill(np.nan)
        state.next_cell_locations.clear()
        state.next_masks.clear()
        state.next_row_id = None
        logger.info("[SequentialProcessor] No next row to prepare.")

    # 3. Apply Cross-Row Padding
    apply_cross_row_padding(state, config)

    # np.savez(f"debug_cross_proj_row_{current_row_id}.npz", state=state, config=config, metadata=metadata, current_row_id=current_row_id, next_row_id=next_row_id, zarr_path=zarr_path, csv_path=csv_path)
    # raise RuntimeError("Debug stop")

    # 4. Apply Cross-Projection Padding (if applicable)
    if csv_path:
        logger.info(f"[SequentialProcessor] Applying parallel cross-projection padding for row {current_row_id}...")
        start_cp = time.time()
        apply_cross_projection_padding(state, config, metadata, current_row_id, next_row_id, zarr_path, csv_path)
        logger.info(f"[SequentialProcessor] Cross-projection padding finished in {time.time() - start_cp:.2f}s")

    # 5. Perform Convolution
    nan_mask = np.isnan(state.current_array)
    state.current_array[nan_mask] = 0.0
    logger.info(f"[SequentialProcessor] Applying convolution for row ID {current_row_id}")
    convolved_array = convolution_utils.apply_gaussian_convolution(state.current_array, sigma=psf_sigma)
    # Restore NaNs on the result, not the state array which will be replaced
    convolved_array[nan_mask] = np.nan

    # 6. Extract and Return Results
    results_data = extract_cell_results(convolved_array, state.cell_locations)
    results_masks = {name: mask for name, mask in state.current_masks.items() if name in state.cell_locations}

    return results_data, results_masks


def sequential_processor(projections: list[str], df: pd.DataFrame, combined_cell_queue: Queue, results_queue: Queue, psf_sigma: float, zarr_path: str, cell_buffer: dict, catalog: Optional[pd.DataFrame] = None, enable_saturation_correction: bool = True, csv_path: Optional[str] = None):
    """
    SPEC: This is Stage 3. It iterates through projections sequentially,
    calling a helper function to process each row, queues results, and manages
    the sliding window state.
    """
    for projection in projections:
        logger.info(f"[SequentialProcessor] --- Starting sequential processing for projection: {projection} ---")
        try:
            metadata = extract_projection_metadata(df, projection)
            config = create_master_array_config(metadata)
            state = initialize_processing_state(config)
            row_ids = sorted(metadata["rows"].keys())
        except Exception as e:
            logger.error(f"[SequentialProcessor] Failed to initialize projection {projection}: {e}. Skipping.")
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
            logger.info(f"[SequentialProcessor] --- Processing step for row {i + 1}/{len(row_ids)}: ROW ID {current_row_id} ---")
            logger.info(f"[SequentialProcessor] Combined queue size: {combined_cell_queue.qsize()}")

            # Determine Next Row
            next_row_id = row_ids[i + 1] if i + 1 < len(row_ids) else None

            try:
                # Call the Helper Function to do the heavy lifting
                results_data, results_masks = process_row_step_from_queue(state, config, metadata, current_row_id, next_row_id, combined_cell_queue, cell_buffer, psf_sigma, zarr_path, projection, catalog, enable_saturation_correction, csv_path)

                # Queue the Results
                processed_bundle = {"projection": projection, "row_id": current_row_id, "results_data": results_data, "results_masks": results_masks}
                results_queue.put(processed_bundle)
                logger.info(f"[SequentialProcessor] Finished processing and queued results for row {current_row_id}")

                # Advance the Window if not the last row
                if next_row_id is not None:
                    advance_sliding_window(state)
            except Exception:
                logger.exception(f"[SequentialProcessor] Critical failure processing row {current_row_id} for projection {projection}")
                # If a row fails, the sliding window state for this projection is likely corrupted.
                # Skip the rest of this projection.
                break

        logger.info(f"[SequentialProcessor] --- Finished sequential processing for projection: {projection} ---")

    # Shutdown Signal for the saver
    results_queue.put(None)


# --- Main Orchestrator ---


def run_modern_sliding_window_pipeline(sector: int, camera: int, ccd: int, data_root: str = "data", projections_limit: Optional[int] = None, psf_sigma: float = 60.0, enable_saturation_correction: bool = True, catalog_path: Optional[str] = None):
    """The top-level master orchestrator for the entire pipeline."""
    global _child_processes
    _child_processes.clear()
    signal.signal(signal.SIGINT, shutdown_handler)

    logger.info(f"[Pipeline] Starting pipeline for sector {sector}, camera {camera}, ccd {ccd}")
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
        logger.error(f"[Pipeline] Failed to load configuration: {e}")
        return {"error": str(e)}

    # Load catalog for saturation correction
    catalog = None
    if enable_saturation_correction:
        try:
            catalog = load_gaia_catalog(data_root, sector, camera, ccd, catalog_path)
            logger.info(f"[Pipeline] Loaded {len(catalog)} stars from catalog")
        except Exception as e:
            logger.warning(f"[Pipeline] Failed to load catalog: {e}")
            catalog = None

    # --- Setup ---
    zarr_store = zarr.open(zarr_path, mode="r")
    num_pre_processors = 50
    num_readers = 2

    task_queue = Queue()
    raw_cell_queue = Queue(maxsize=20)
    combined_cell_queue = Queue(maxsize=30)
    results_queue = Queue(maxsize=3)
    cell_buffer = {}  # Shared buffer for all projections

    # --- Start Persistent Workers (Stages 1, 4) ---
    saver_proc = Process(target=saver_worker, args=(results_queue, output_path))
    saver_proc.start()

    # Start the process coordinator in a separate thread
    import threading

    process_coordinator_thread = threading.Thread(target=process_coordinator, args=(raw_cell_queue, combined_cell_queue, cell_buffer, num_pre_processors), daemon=True)
    process_coordinator_thread.start()

    with ThreadPoolExecutor(max_workers=num_readers) as reader_executor:
        # Start Stage 1 workers (readers only - preprocessing now handled by process coordinator)
        for _ in range(num_readers):
            reader_executor.submit(reader_worker, task_queue, raw_cell_queue, zarr_store)

        # --- Dispatch All Tasks ---
        logger.info(f"[Pipeline] Dispatching tasks for {len(projections)} projections.")
        master_task_list = []
        for projection in projections:
            try:
                metadata = extract_projection_metadata(df, projection)
                for row_id, cells in sorted(metadata["rows"].items()):
                    for cell_name, x_coord in cells:
                        master_task_list.append((cell_name, projection, row_id, x_coord))
            except Exception as e:
                logger.error(f"[Pipeline] Failed to create tasks for projection {projection}: {e}")

        for task in master_task_list:
            task_queue.put(task)
        logger.info("[Pipeline] All tasks have been dispatched to the reader workers.")

        # --- Signal Reader Shutdown ---
        for _ in range(num_readers):
            task_queue.put(None)

        # --- Run Sequential Processor (Stage 3) in Main Thread ---
        sequential_processor(projections, df, combined_cell_queue, results_queue, psf_sigma, zarr_path, cell_buffer, catalog, enable_saturation_correction, csv_path)

        # --- Final Shutdown Sequence ---
        logger.info("[Pipeline] Sequential processor finished. Shutting down upstream workers.")
        # Signal the process coordinator to shutdown
        raw_cell_queue.put(None)

        # Wait for process coordinator thread to finish
        if process_coordinator_thread.is_alive():
            logger.info("[Pipeline] Waiting for process coordinator to finish...")
            process_coordinator_thread.join(timeout=30)

    saver_proc.join()
    logger.info("[Pipeline] Pipeline completed successfully!")
    return {"status": "success"}


def shutdown_handler(signum, frame):
    """
    Handles SIGINT (Ctrl+C) to ensure all child processes are terminated.
    """
    logger.warning("[Pipeline] Ctrl+C detected! Initiating graceful shutdown...")
    for p in _child_processes:
        if p.is_alive():
            logger.info(f"[Pipeline] Terminating process: {p.name} (PID: {p.pid})")
            p.terminate()
            p.join()
    logger.info("[Pipeline] All child processes terminated. Exiting.")
    sys.exit(1)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Syndiff Template PS1 Processing Pipeline")
    parser.add_argument("sector", type=int, help="TESS sector number")
    parser.add_argument("camera", type=int, help="TESS camera number")
    parser.add_argument("ccd", type=int, help="TESS CCD number")
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument("--limit", type=int, help="Limit projections for testing")
    parser.add_argument("--psf-sigma", type=float, default=40.0, help="PSF sigma for convolution")
    parser.add_argument("--enable-saturation-correction", action="store_true", default=False, help="Enable saturation correction")
    parser.add_argument("--catalog-path", help="Path to Gaia catalog CSV file")
    args = parser.parse_args()
    results = run_modern_sliding_window_pipeline(args.sector, args.camera, args.ccd, data_root=args.data_root, projections_limit=args.limit, psf_sigma=args.psf_sigma, enable_saturation_correction=args.enable_saturation_correction, catalog_path=args.catalog_path)

    if results.get("status") == "success":
        print("\n✅ Syndiff Template PS1 processing pipeline completed successfully!")
    else:
        print(f"\n❌ Pipeline failed: {results.get('error', 'Unknown error')}")
