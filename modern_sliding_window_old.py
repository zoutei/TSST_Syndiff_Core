"""
Modern Sliding Window Pipeline Implementation

Function-oriented implementation of the PS1 Simplified Processing Specification
using a sliding window approach with master arrays for efficient row-based processing.
"""

import gc
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import Manager
from typing import Optional

import numpy as np
from astropy.wcs import WCS

# Import existing utilities
from band_utils import process_skycell_bands
from convolution_utils import apply_gaussian_convolution
from csv_utils import load_csv_data
from zarr_utils import load_skycell_bands_masks_and_headers

logger = logging.getLogger(__name__)

# Key constants from the specification
CELL_OVERLAP = 480  # Natural overlap between adjacent cells
EDGE_EXCLUSION = 10  # Amount to overwrite for smooth blending
EFFECTIVE_OVERLAP = CELL_OVERLAP - EDGE_EXCLUSION  # 470 pixels
PAD_SIZE = 480  # Padding around each row


@dataclass
class CellInfo:
    """Information about a single cell."""

    name: str
    projection: str
    row_id: int
    x_coord: int
    cell_index: int
    width: int
    height: int


@dataclass
class MasterArrayConfig:
    """Configuration for master arrays."""

    width: int
    height: int
    cell_width: int
    cell_height: int
    max_cells: int
    cell_dimensions: dict  # mapping cell_name -> (width, height)


@dataclass
class PaddingCell:
    """Information about a cell used for padding."""

    name: str
    projection: str
    loaded: bool = False
    used_for: list[str] = None  # list of rows this cell is padding
    locations: list[str] = None  # list of locations like "top", "bottom"

    def __post_init__(self):
        if self.used_for is None:
            self.used_for = []
        if self.locations is None:
            self.locations = []


@dataclass
class ProcessingState:
    """Current processing state for sliding window."""

    current_array: Optional[np.ndarray] = None
    next_array: Optional[np.ndarray] = None
    current_masks: dict[str, np.ndarray] = None  # current row masks
    next_masks: dict[str, np.ndarray] = None  # next row masks
    current_row_id: Optional[int] = None
    next_row_id: Optional[int] = None
    cell_locations: dict[str, tuple[int, int, int, int]] = None  # name -> (x_start, x_end, y_start, y_end)
    next_cell_locations: dict[str, tuple[int, int, int, int]] = None  # next row cell locations
    padding_cells: dict[str, PaddingCell] = None  # name -> PaddingCell

    def __post_init__(self):
        if self.current_masks is None:
            self.current_masks = {}
        if self.next_masks is None:
            self.next_masks = {}
        if self.cell_locations is None:
            self.cell_locations = {}
        if self.next_cell_locations is None:
            self.next_cell_locations = {}
        if self.padding_cells is None:
            self.padding_cells = {}


def extract_projection_metadata(csv_path: str, projection: str) -> dict:
    """Extract projection metadata from CSV file."""
    df = load_csv_data(csv_path)
    proj_df = df[df["projection"].astype(str) == projection]

    if len(proj_df) == 0:
        raise ValueError(f"No data found for projection {projection}")

    # Extract row information
    rows = {}
    cell_dimensions = {}
    for _, row in proj_df.iterrows():
        row_id = int(row["y"])
        cell_name = row["NAME"]
        x_coord = int(row["x"])

        # Store actual cell dimensions
        cell_width = int(row.get("NAXIS1"))
        cell_height = int(row.get("NAXIS2"))
        cell_dimensions[cell_name] = (cell_width, cell_height)

        if row_id not in rows:
            rows[row_id] = []
        rows[row_id].append((cell_name, x_coord))

    # Sort cells within each row by x coordinate
    for row_id in rows:
        rows[row_id].sort(key=lambda x: x[1])
        rows[row_id] = [cell_name for cell_name, _ in rows[row_id]]

    # Get typical cell dimensions (use most common size)
    # Check all cell dimensions are the same
    all_dims = list(cell_dimensions.values())
    if len(set(all_dims)) == 1:
        typical_width, typical_height = all_dims[0]
    else:
        raise ValueError(f"Not all cell dimensions are the same: found {set(all_dims)}")

    # Calculate max cells per row
    max_cells_per_row = max(len(cells) for cells in rows.values()) if rows else 0

    logger.info(f"Found {len(cell_dimensions)} cells with dimensions ranging from typical {typical_width}x{typical_height}")

    return {"projection": projection, "rows": rows, "cell_width": typical_width, "cell_height": typical_height, "max_cells_per_row": max_cells_per_row, "cell_dimensions": cell_dimensions, "dataframe": proj_df}


def create_master_array_config(metadata: dict) -> MasterArrayConfig:
    """Create master array configuration from metadata."""
    cell_width = metadata["cell_width"]
    cell_height = metadata["cell_height"]
    max_cells = metadata["max_cells_per_row"]
    cell_dimensions = metadata["cell_dimensions"]

    master_width = (max_cells * cell_width) + (2 * PAD_SIZE)
    master_height = cell_height + (2 * PAD_SIZE)

    return MasterArrayConfig(width=master_width, height=master_height, cell_width=cell_width, cell_height=cell_height, max_cells=max_cells, cell_dimensions=cell_dimensions)


def initialize_processing_state(config: MasterArrayConfig) -> ProcessingState:
    """Initialize processing state with empty master arrays."""
    current_array = np.zeros((config.height, config.width), dtype=np.float32)
    next_array = np.zeros((config.height, config.width), dtype=np.float32)

    return ProcessingState(current_array=current_array, next_array=next_array)


def _parallel_cell_loader(tasks_queue, results_queue, zarr_path):
    """
    Producer function: loads cell data using threads.
    Pulls cell tasks from the queue and loads their data.
    """
    while True:
        try:
            task = tasks_queue.get_nowait()
        except Exception:
            break  # No more tasks

        cell_name, projection, cell_index = task

        try:
            logger.debug(f"[Loader] Loading cell {cell_name} (index {cell_index})")
            # Load and combine bands using parallel loading with headers
            bands_data, masks_data, headers_data = load_skycell_bands_masks_and_headers(zarr_path, projection, cell_name)
            combined_image, combined_mask = process_skycell_bands(bands_data, masks_data, headers_data)

            # Put the loaded data onto the results queue
            results_queue.put(("success", cell_name, cell_index, combined_image, combined_mask))

        except Exception as e:
            logger.warning(f"[Loader] Failed to load cell {cell_name}: {e}")
            results_queue.put(("error", cell_name, cell_index, None, None))

    logger.debug("Cell loader finished")


def _parallel_cell_processor(results_queue, target_array, config: MasterArrayConfig, cell_x_coords: dict, first_x_coord: int, num_cells: int):
    """
    Consumer function: processes loaded cell data and places in target array.
    """
    processed_count = 0
    cell_positions = {}
    cell_masks = {}

    while processed_count < num_cells:
        try:
            result = results_queue.get(timeout=30)  # 30 second timeout
        except Exception:
            logger.warning("Timeout waiting for cell data")
            break

        status, cell_name, loaded_cell_index, combined_image, combined_mask = result

        if status != "success" or combined_image is None:
            logger.warning(f"[Processor] Skipping failed cell {cell_name}")
            processed_count += 1
            continue

        try:
            logger.debug(f"[Processor] Processing cell {cell_name}")

            # Calculate cell_index relative to the first cell in this row
            cell_index = cell_x_coords[cell_name] - first_x_coord if cell_name in cell_x_coords else loaded_cell_index

            # Get actual cell dimensions from config
            actual_width, actual_height = config.cell_dimensions.get(cell_name, (config.cell_width, config.cell_height))

            # Use actual loaded dimensions for calculation
            loaded_height, loaded_width = combined_image.shape

            # Calculate position using standard cell width for positioning, INCLUDE PAD_SIZE
            target_x_start_full = PAD_SIZE + cell_index * (config.cell_width - CELL_OVERLAP)
            target_x_end = target_x_start_full + config.cell_width
            target_y_start = PAD_SIZE
            target_y_end = target_y_start + config.cell_height

            # Handle overlap for source region
            if cell_index == 0:
                # First cell: use full width
                source_x_start, source_x_end = 0, loaded_width
                target_x_start = target_x_start_full
            else:
                # Subsequent cells: skip overlap region if possible
                source_x_start = EFFECTIVE_OVERLAP
                source_x_end = loaded_width
                target_x_start = target_x_start_full + EFFECTIVE_OVERLAP

            source_y_start, source_y_end = 0, loaded_height

            # Place the cell and store results
            target_array[target_y_start:target_y_end, target_x_start:target_x_end] = combined_image[source_y_start:source_y_end, source_x_start:source_x_end]

            logger.debug(f"Placed {cell_name} ({loaded_width}x{loaded_height}) at target[{target_y_start}:{target_y_end}, {target_x_start}:{target_x_end}] from source[{source_y_start}:{source_y_end}, {source_x_start}:{source_x_end}]")

            # Extract the mask region that was actually placed
            cell_positions[cell_name] = (target_x_start_full, target_x_end, target_y_start, target_y_end)
            cell_masks[cell_name] = combined_mask
            logger.debug(f"Successfully placed cell {cell_name} at position ({target_x_start}, {target_x_end}, {target_y_start}, {target_y_end})")

        except Exception as e:
            logger.warning(f"[Processor] Failed to process cell {cell_name}: {e}")

        processed_count += 1

    logger.debug(f"Cell processor finished: {len(cell_positions)}/{num_cells} cells successfully processed")
    return cell_positions, cell_masks


def load_row_into_array_parallel(zarr_path: str, projection: str, row_cells: list[str], target_array: np.ndarray, config: MasterArrayConfig, metadata: dict, num_loaders: int = 4) -> tuple[dict[str, tuple[int, int, int, int]], dict[str, np.ndarray]]:
    """Load a complete row of cells into target array using parallel loading.

    Returns:
        Tuple of (cell_positions, cell_masks)
    """
    logger.info(f"Loading row with {len(row_cells)} cells in parallel: {row_cells}")
    target_array.fill(np.nan)  # Clear array

    if not row_cells:
        return {}, {}

    # Get the minimum x-coordinate to handle non-zero starting indices
    proj_df = metadata["dataframe"]
    first_x_coord = None
    cell_x_coords = {}

    for cell_name in row_cells:
        cell_row = proj_df[proj_df["NAME"] == cell_name]
        if not cell_row.empty:
            x_coord = int(cell_row.iloc[0]["x"])
            cell_x_coords[cell_name] = x_coord
            if first_x_coord is None:
                first_x_coord = x_coord
            else:
                first_x_coord = min(first_x_coord, x_coord)

    # Use multiprocessing Manager for thread-safe communication
    with Manager() as manager:
        tasks_queue = manager.Queue()
        results_queue = manager.Queue()

        # Add all cell loading tasks to the queue
        for i, cell_name in enumerate(row_cells):
            tasks_queue.put((cell_name, projection, i))

        # Start loader threads
        with ThreadPoolExecutor(max_workers=num_loaders) as executor:
            # Start loaders
            for _ in range(num_loaders):
                executor.submit(_parallel_cell_loader, tasks_queue, results_queue, zarr_path)

        # Process results in main thread to avoid array sharing issues
        cell_positions, cell_masks = _parallel_cell_processor(results_queue, target_array, config, cell_x_coords, first_x_coord, len(row_cells))

    logger.info(f"Parallel row loading complete: {len(cell_positions)}/{len(row_cells)} cells successfully loaded")
    return cell_positions, cell_masks


def calculate_cell_position(cell_index: int, config: MasterArrayConfig) -> tuple[int, int, int, int]:
    """Calculate cell position in master array with overlap handling."""
    if cell_index == 0:
        # First cell: place normally
        x_start = PAD_SIZE
        x_end = PAD_SIZE + config.cell_width
    else:
        # Subsequent cells: handle overlap
        x_start = PAD_SIZE + (config.cell_width - EFFECTIVE_OVERLAP) * cell_index - EDGE_EXCLUSION
        x_end = x_start + config.cell_width

    y_start = PAD_SIZE
    y_end = PAD_SIZE + config.cell_height

    return x_start, x_end, y_start, y_end


def load_and_place_cell(zarr_path: str, cell_name: str, projection: str, cell_index: int, target_array: np.ndarray, config: MasterArrayConfig) -> tuple[int, int, int, int, np.ndarray]:
    """Load a cell and place it in the target array with overlap handling.

    Returns:
        Tuple of (x_start, x_end, y_start, y_end, combined_mask_uint16)
    """
    try:
        # logger.info(f"Loading cell {cell_name} (index {cell_index}) from projection {projection}")

        # Load and combine bands using parallel loading with headers
        bands_data, masks_data, headers_data = load_skycell_bands_masks_and_headers(zarr_path, projection, cell_name)
        combined_image, combined_mask = process_skycell_bands(bands_data, masks_data, headers_data)

        # Get actual cell dimensions from config
        actual_width, actual_height = config.cell_dimensions.get(cell_name, (config.cell_width, config.cell_height))

        # Log dimension mismatch if any
        if combined_image.shape != (actual_height, actual_width):
            logger.debug(f"Cell {cell_name} size mismatch: loaded {combined_image.shape}, expected ({actual_height}, {actual_width})")

        # Use actual loaded dimensions for calculation
        loaded_height, loaded_width = combined_image.shape

        # Calculate position using standard cell width for positioning, INCLUDE PAD_SIZE
        x_start = PAD_SIZE + cell_index * (config.cell_width - CELL_OVERLAP)
        y_start = PAD_SIZE

        # Handle overlap for source region
        if cell_index == 0:
            # First cell: use full width
            source_x_start, source_x_end = 0, loaded_width
        else:
            # Subsequent cells: skip overlap region if possible
            source_x_start = EFFECTIVE_OVERLAP
            source_x_end = loaded_width

        source_y_start, source_y_end = 0, loaded_height

        # Ensure we don't go out of bounds
        target_x_end = min(x_start + (source_x_end - source_x_start), target_array.shape[1])
        target_y_end = min(y_start + (source_y_end - source_y_start), target_array.shape[0])

        # Adjust source to match actual target size
        target_width = target_x_end - x_start
        target_height = target_y_end - y_start
        source_x_end = min(source_x_start + target_width, loaded_width)
        source_y_end = min(source_y_start + target_height, loaded_height)

        # Place the cell and return mask
        if target_x_end > x_start and target_y_end > y_start and source_x_end > source_x_start and source_y_end > source_y_start:
            target_array[y_start:target_y_end, x_start:target_x_end] = combined_image[source_y_start:source_y_end, source_x_start:source_x_end]

            logger.debug(f"Placed {cell_name} ({loaded_width}x{loaded_height}) at target[{y_start}:{target_y_end}, {x_start}:{target_x_end}] from source[{source_y_start}:{source_y_end}, {source_x_start}:{source_x_end}]")

            # Extract the mask region that was actually placed
            extracted_mask = combined_mask[source_y_start:source_y_end, source_x_start:source_x_end]
            return x_start, target_x_end, y_start, target_y_end, extracted_mask
        else:
            logger.warning(f"Could not place cell {cell_name} - invalid region")
            return 0, 0, 0, 0, np.array([])

    except Exception as e:
        logger.warning(f"Failed to load cell {cell_name}: {e}")
        return 0, 0, 0, 0, np.array([])


def load_row_into_array(zarr_path: str, projection: str, row_cells: list[str], target_array: np.ndarray, config: MasterArrayConfig, metadata: dict) -> tuple[dict[str, tuple[int, int, int, int]], dict[str, np.ndarray]]:
    """Load a complete row of cells into target array.

    Returns:
        Tuple of (cell_positions, cell_masks)
    """
    logger.info(f"Loading row with {len(row_cells)} cells: {row_cells}")
    target_array.fill(0)  # Clear array
    cell_positions = {}
    cell_masks = {}

    # Get the minimum x-coordinate to handle non-zero starting indices
    proj_df = metadata["dataframe"]
    first_x_coord = None
    cell_x_coords = {}

    for cell_name in row_cells:
        cell_row = proj_df[proj_df["NAME"] == cell_name]
        if not cell_row.empty:
            x_coord = int(cell_row.iloc[0]["x"])
            cell_x_coords[cell_name] = x_coord
            if first_x_coord is None:
                first_x_coord = x_coord
            else:
                first_x_coord = min(first_x_coord, x_coord)

    for cell_name in row_cells:
        # Calculate cell_index relative to the first cell in this row
        cell_index = cell_x_coords[cell_name] - first_x_coord if cell_name in cell_x_coords else 0

        logger.info(f"Processing cell {cell_name} (x={cell_x_coords.get(cell_name, 'unknown')}, index={cell_index})")
        x_start, x_end, y_start, y_end, cell_mask = load_and_place_cell(zarr_path, cell_name, projection, cell_index, target_array, config)
        if (x_start, x_end, y_start, y_end) != (0, 0, 0, 0):  # Valid position
            cell_positions[cell_name] = (x_start, x_end, y_start, y_end)
            cell_masks[cell_name] = cell_mask
            logger.info(f"Successfully placed cell {cell_name} at position ({x_start}, {x_end}, {y_start}, {y_end})")
        else:
            logger.warning(f"Failed to place cell {cell_name}")

    logger.info(f"Row loading complete: {len(cell_positions)}/{len(row_cells)} cells successfully loaded")
    return cell_positions, cell_masks


def apply_cross_row_padding(state: ProcessingState, config: MasterArrayConfig) -> None:
    """Apply cross-row padding between current and next arrays with correct overlap handling."""
    if state.next_array is None:
        return

    # From next row: take region from EFFECTIVE_OVERLAP to (EFFECTIVE_OVERLAP + PAD_SIZE)
    # Place into current row: from -EDGE_EXCLUSION (which means near the top padding)

    # Source region from next row
    next_source_y_start = PAD_SIZE + CELL_OVERLAP - EDGE_EXCLUSION
    next_source_y_end = PAD_SIZE * 2 + CELL_OVERLAP

    # Target region in current row (top padding area, offset by EDGE_EXCLUSION)
    current_target_y_start = config.cell_height - EDGE_EXCLUSION + PAD_SIZE

    state.current_array[current_target_y_start:, :] = state.next_array[next_source_y_start:next_source_y_end, :]

    # Opposite direction: from current to next
    # Source region from current row
    current_source_y_start = config.cell_height - CELL_OVERLAP
    current_source_y_end = PAD_SIZE + config.cell_height - CELL_OVERLAP + EDGE_EXCLUSION

    next_target_y_end = PAD_SIZE + EDGE_EXCLUSION

    state.next_array[:next_target_y_end, :] = state.current_array[current_source_y_start:current_source_y_end, :]


def extract_cell_results(convolved_array: np.ndarray, cell_positions: dict[str, tuple[int, int, int, int]], config: MasterArrayConfig) -> dict[str, np.ndarray]:
    """Extract individual cell results from convolved array."""
    results = {}

    for cell_name, (x_start, x_end, y_start, y_end) in cell_positions.items():
        try:
            # Extract the cell region (could add overlap exclusion logic here)
            extracted = convolved_array[y_start:y_end, x_start:x_end].copy()
            results[cell_name] = extracted

        except Exception as e:
            logger.warning(f"Failed to extract {cell_name}: {e}")
            continue

    return results


def process_row_sliding_window(zarr_path: str, metadata: dict, config: MasterArrayConfig, state: ProcessingState, current_row_id: int, next_row_id: Optional[int], csv_path: str, psf_sigma: float = 60.0) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Process a single row using sliding window approach."""
    logger.info(f"Processing row {current_row_id} (next: {next_row_id})")
    rows = metadata["rows"]
    projection = metadata["projection"]

    # Load current row if not already loaded (first iteration)
    if state.current_row_id != current_row_id:
        logger.info(f"Loading current row {current_row_id}")
        current_cells = rows.get(current_row_id, [])
        current_positions, current_masks = load_row_into_array_parallel(zarr_path, projection, current_cells, state.current_array, config, metadata, num_loaders=4)
        state.cell_locations.update(current_positions)
        state.current_masks.update(current_masks)
        state.current_row_id = current_row_id
        logger.info(f"Current row {current_row_id} loaded with {len(current_positions)} cells")

    # Load next row if needed
    if next_row_id is not None and state.next_row_id != next_row_id:
        logger.info(f"Loading next row {next_row_id}")
        next_cells = rows.get(next_row_id, [])
        next_positions, next_masks = load_row_into_array_parallel(zarr_path, projection, next_cells, state.next_array, config, metadata, num_loaders=4)
        state.next_masks.update(next_masks)
        state.next_cell_locations.update(next_positions)  # Store next row cell locations
        state.next_row_id = next_row_id
        logger.info(f"Next row {next_row_id} loaded")

    # Apply cross-row padding
    logger.info("Applying cross-row padding")
    apply_cross_row_padding(state, config)

    # # Load cross-projection padding using modern system
    # logger.info("Loading cross-projection padding")
    # # Add csv_path to metadata for cross-projection loading
    # metadata_with_csv = dict(metadata)
    # metadata_with_csv["csv_path"] = csv_path
    # from modern_padding import load_modern_cross_projection_padding

    # load_modern_cross_projection_padding(state, config, metadata_with_csv, current_row_id, zarr_path)

    # Convolve current array
    logger.info(f"Starting convolution with sigma={psf_sigma}")
    nans_loc = np.isnan(state.current_array)
    state.current_array[nans_loc] = 0  # Replace NaNs with 0 for convolution
    convolved = apply_gaussian_convolution(state.current_array, sigma=psf_sigma)
    logger.info("Convolution completed")

    # Extract results for current row cells
    current_positions = {name: pos for name, pos in state.cell_locations.items() if name in rows.get(current_row_id, [])}

    logger.info(f"Extracting results for {len(current_positions)} cells")
    results_data = extract_cell_results(convolved, current_positions, config)

    # Extract corresponding masks for the current row
    current_masks_for_row = {name: mask for name, mask in state.current_masks.items() if name in current_positions}

    logger.info(f"Row {current_row_id} processing complete: {len(results_data)} results extracted")

    return results_data, current_masks_for_row


def create_master_array_wcs(metadata: dict, config: MasterArrayConfig, current_row_id: int) -> WCS:
    """Create WCS object for the master array based on first cell + padding offset."""
    # Get first cell's WCS information from the dataframe
    proj_df = metadata["dataframe"]
    # Filter to the current projection and row
    proj_df = proj_df[proj_df["projection"].astype(str) == str(metadata["projection"])]
    proj_df = proj_df[proj_df["y"] == int(current_row_id)]
    proj_df = proj_df.sort_values(by="x")
    first_cell = proj_df.iloc[0]

    # Extract WCS parameters; use PC and CDELT instead of CD
    crval1 = float(first_cell["CRVAL1"])
    crval2 = float(first_cell["CRVAL2"])
    crpix1 = float(first_cell["CRPIX1"]) + PAD_SIZE
    crpix2 = float(first_cell["CRPIX2"]) + PAD_SIZE
    pc1_1 = float(first_cell["PC1_1"])
    pc1_2 = float(first_cell["PC1_2"])
    pc2_1 = float(first_cell["PC2_1"])
    pc2_2 = float(first_cell["PC2_2"])
    cdelt1 = float(first_cell["CDELT1"])
    cdelt2 = float(first_cell["CDELT2"])

    master_wcs = WCS(naxis=2)
    master_wcs.wcs.crval = [crval1, crval2]
    master_wcs.wcs.crpix = [crpix1, crpix2]
    master_wcs.wcs.pc = [[pc1_1, pc1_2], [pc2_1, pc2_2]]
    master_wcs.wcs.cdelt = [cdelt1, cdelt2]
    master_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    master_wcs.wcs.cunit = ["deg", "deg"]

    return master_wcs


def create_cell_wcs(cell_name: str, metadata: dict) -> WCS:
    """Create WCS object for a specific cell."""
    proj_df = metadata["dataframe"]
    cell_row = proj_df[proj_df["NAME"] == cell_name]

    if cell_row.empty:
        raise ValueError(f"Cell {cell_name} not found in metadata")

    cell_data = cell_row.iloc[0]

    # Extract WCS parameters
    crval1 = float(cell_data["CRVAL1"])
    crval2 = float(cell_data["CRVAL2"])
    crpix1 = float(cell_data["CRPIX1"]) + PAD_SIZE
    crpix2 = float(cell_data["CRPIX2"]) + PAD_SIZE
    pc1_1 = float(cell_data["PC1_1"])
    pc1_2 = float(cell_data["PC1_2"])
    pc2_1 = float(cell_data["PC2_1"])
    pc2_2 = float(cell_data["PC2_2"])
    cdelt1 = float(cell_data["CDELT1"])
    cdelt2 = float(cell_data["CDELT2"])

    # Create WCS object
    cell_wcs = WCS(naxis=2)
    cell_wcs.wcs.crval = [crval1, crval2]
    cell_wcs.wcs.crpix = [crpix1, crpix2]
    cell_wcs.wcs.cd = [[pc1_1, pc1_2], [pc2_1, pc2_2]]
    cell_wcs.wcs.cdelt = [cdelt1, cdelt2]
    cell_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    cell_wcs.wcs.cunit = ["deg", "deg"]

    return cell_wcs


def advance_sliding_window(state: ProcessingState) -> None:
    """Advance the sliding window (next becomes current)."""
    logger.info(f"Before advance: current_row_id={state.current_row_id}, next_row_id={state.next_row_id}")

    state.current_array, state.next_array = state.next_array, state.current_array
    state.next_array.fill(0)

    # Advance masks
    state.current_masks, state.next_masks = state.next_masks, {}

    # Advance row IDs
    state.current_row_id = state.next_row_id
    state.next_row_id = None

    # Transfer next cell locations to current and clear next
    state.cell_locations.clear()
    state.cell_locations.update(state.next_cell_locations)
    state.next_cell_locations.clear()

    logger.info(f"After advance: current_row_id={state.current_row_id}, next_row_id={state.next_row_id}")


def process_projection_sliding_window(zarr_path: str, csv_path: str, projection: str, output_path: str, psf_sigma: float = 60.0) -> dict:
    """Process a complete projection using sliding window approach."""
    logger.info(f"Processing projection {projection} with sliding window")

    # Extract metadata
    logger.info("Extracting projection metadata")
    metadata = extract_projection_metadata(csv_path, projection)
    logger.info("Creating master array configuration")
    config = create_master_array_config(metadata)
    logger.info("Initializing processing state")
    state = initialize_processing_state(config)

    # Identify padding requirements
    logger.info("Identifying padding requirements")
    # state.padding_cells = identify_padding_requirements(metadata, csv_path)

    logger.info(f"Projection {projection}: {len(metadata['rows'])} rows, max {config.max_cells} cells/row, cell size {config.cell_width}x{config.cell_height}")

    # Process rows
    row_ids = sorted(metadata["rows"].keys())
    logger.info(f"Processing {len(row_ids)} rows: {row_ids}")
    processed_rows = 0
    total_cells = 0

    for i, current_row_id in enumerate(row_ids):
        try:
            logger.info(f"=== Processing row {i + 1}/{len(row_ids)}: row_id={current_row_id} ===")

            # Determine next row
            next_row_id = row_ids[i + 1] if i + 1 < len(row_ids) else None

            # Process row
            results_data, results_masks = process_row_sliding_window(zarr_path, metadata, config, state, current_row_id, next_row_id, csv_path, psf_sigma)

            # Save results
            if results_data:
                logger.info(f"Saving {len(results_data)} results for row {current_row_id}")
                from zarr_utils import save_convolved_results

                save_convolved_results(output_path, projection, current_row_id, results_data, results_masks)
                total_cells += len(results_data)
                logger.info("Results saved successfully")

            processed_rows += 1

            # Advance window for next iteration (except on last row)
            if i < len(row_ids) - 1:
                logger.info("Advancing sliding window")
                advance_sliding_window(state)

            # Memory cleanup
            logger.info("Running garbage collection")
            gc.collect()

            logger.info(f"Row {current_row_id} complete. Progress: {processed_rows}/{len(row_ids)} rows, {total_cells} total cells")

        except Exception as e:
            logger.error(f"Failed to process row {current_row_id}: {e}")
            continue

    logger.info(f"Completed projection {projection}: {processed_rows}/{len(row_ids)} rows, {total_cells} cells")

    return {"projection": projection, "rows_processed": processed_rows, "total_rows": len(row_ids), "cells_processed": total_cells}


def run_modern_sliding_window_pipeline(sector: int, camera: int, ccd: int, data_root: str = "data", projections_limit: Optional[int] = None) -> dict:
    """Run the complete modern sliding window pipeline."""
    logger.info(f"Starting modern sliding window pipeline for sector {sector}, camera {camera}, ccd {ccd}")

    # Set up paths
    zarr_path = f"{data_root}/ps1_skycells_zarr/ps1_skycells.zarr"
    output_path = f"{data_root}/convolved_results/sector_{sector:04d}_camera_{camera}_ccd_{ccd}.zarr"

    try:
        from csv_utils import find_csv_file, get_projections_from_csv

        csv_path = find_csv_file(data_root, sector, camera, ccd)
        projections = get_projections_from_csv(csv_path)

        if projections_limit:
            projections = projections[:projections_limit]

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return {"error": str(e)}

    # Process each projection
    results = []
    for i, projection in enumerate(projections):
        logger.info(f"Progress: {i + 1}/{len(projections)} projections")

        try:
            result = process_projection_sliding_window(zarr_path, csv_path, projection, output_path, psf_sigma=60.0)
            results.append(result)

        except Exception as e:
            logger.error(f"Failed to process projection {projection}: {e}")
            continue

    # Summary
    total_rows = sum(r["rows_processed"] for r in results)
    total_cells = sum(r["cells_processed"] for r in results)

    logger.info("Modern sliding window pipeline completed!")
    logger.info(f"  Processed {len(results)}/{len(projections)} projections")
    logger.info(f"  Total rows: {total_rows}")
    logger.info(f"  Total cells: {total_cells}")

    return {"sector": sector, "camera": camera, "ccd": ccd, "projections_processed": len(results), "total_projections": len(projections), "total_rows": total_rows, "total_cells": total_cells, "results": results}


if __name__ == "__main__":
    import argparse

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Modern Sliding Window PS1 Processing Pipeline")
    parser.add_argument("sector", type=int, help="TESS sector number")
    parser.add_argument("camera", type=int, help="TESS camera number")
    parser.add_argument("ccd", type=int, help="TESS CCD number")
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument("--limit", type=int, help="Limit projections for testing")
    parser.add_argument("--psf-sigma", type=float, default=60.0, help="PSF sigma for convolution")

    args = parser.parse_args()

    results = run_modern_sliding_window_pipeline(args.sector, args.camera, args.ccd, data_root=args.data_root, projections_limit=args.limit)

    if "error" not in results:
        print("\n✅ Modern sliding window pipeline completed successfully!")
        print(f"Processed {results['projections_processed']} projections")
        print(f"Total cells processed: {results['total_cells']}")
    else:
        print(f"\n❌ Pipeline failed: {results['error']}")
        exit(1)
