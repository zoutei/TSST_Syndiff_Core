"""
Modern Padding System for PS1 Sliding Window Pipeline

Function-oriented implementation of optimized multi-skycell padding with:
- Row-specific reverse mapping
- Corner padding filtering for interior cells
- Cross-row skycell sharing detection
- Intelligent reprojection caching
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from astropy.wcs import WCS
from reproject import reproject_interp

from band_utils import process_skycell_bands
from csv_utils import load_csv_data
from zarr_utils import load_skycell_bands_masks_and_headers

logger = logging.getLogger(__name__)

# Constants
PAD_SIZE = 480
CELL_OVERLAP = 480
EDGE_EXCLUSION = 10


@dataclass
class CellPositionInfo:
    """Information about a cell's position within its row."""

    actual_index: int  # Index within this specific row
    is_first: bool  # First in this actual row
    is_last: bool  # Last in this actual row
    is_interior: bool  # Interior cell in row
    cell_position: str  # "first", "last", "interior"


@dataclass
class RowPositionInfo:
    """Information about a row's position within the projection."""

    row_id: int
    row_position: str  # "top", "bottom", "middle"
    actual_size: int  # Actual number of cells in this row
    max_size: int  # Maximum cells possible in projection


@dataclass
class SkycellPaddingInfo:
    """Information about a skycell's padding requirements."""

    skycell_name: str
    projection: str
    locations: list[str]  # Filtered locations this skycell serves
    cell_position: str  # Position of the cell this skycell pads
    row_position: str  # Position of the row this skycell pads
    actual_index: int  # Actual index in the row
    sharing_type: str  # "immediate", "shared", "cached"


@dataclass
class ReprojectionTask:
    """A reprojection task for one or more skycells."""

    skycell_name: str
    projection: str
    target_locations: list[str]  # All locations this reprojection serves
    target_shape: tuple[int, int]
    target_bounds: tuple[int, int, int, int]  # (min_y, max_y, min_x, max_x)
    union_locations: list[str]  # If shared across rows
    cache_key: str


# Location coordinate mapping functions
LOCATION_COORDINATES = {"top": lambda config: (0, PAD_SIZE, 0, config.width), "bottom": lambda config: (PAD_SIZE + config.cell_height, config.height, 0, config.width), "left": lambda config: (0, config.height, 0, PAD_SIZE), "right": lambda config: (0, config.height, config.width - PAD_SIZE, config.width), "top_left": lambda config: (0, PAD_SIZE, 0, PAD_SIZE), "top_right": lambda config: (0, PAD_SIZE, config.width - PAD_SIZE, config.width), "bottom_left": lambda config: (PAD_SIZE + config.cell_height, config.height, 0, PAD_SIZE), "bottom_right": lambda config: (PAD_SIZE + config.cell_height, config.height, config.width - PAD_SIZE, config.width)}


def determine_row_position(row_id: int, all_row_ids: list[int]) -> str:
    """Determine if row is top, bottom, or middle row in projection."""
    if row_id == min(all_row_ids):
        return "bottom"
    elif row_id == max(all_row_ids):
        return "top"
    else:
        return "middle"


def analyze_cell_positions(row_cells: list[str]) -> dict[str, CellPositionInfo]:
    """Analyze position of each cell within its actual row."""
    positions = {}
    actual_row_size = len(row_cells)

    for i, cell_name in enumerate(row_cells):
        is_first = i == 0
        is_last = i == actual_row_size - 1
        # is_interior = 0 < i < actual_row_size - 1

        if is_first:
            position_type = "first"
        elif is_last:
            position_type = "last"
        else:
            position_type = "interior"

        positions[cell_name] = position_type

    return positions


def filter_padding_locations(all_locations: list[str], cell_info: CellPositionInfo) -> list[str]:
    """Filter padding locations based on cell position within row."""
    filtered = []

    for location in all_locations:
        if location in ["top", "bottom"]:
            # Always valid regardless of cell position
            filtered.append(location)
        elif location in ["left", "top_left", "bottom_left"]:
            # Only for actual first cell in row
            if cell_info.cell_position == "first":
                filtered.append(location)
        elif location in ["right", "top_right", "bottom_right"]:
            # Only for actual last cell in row
            if cell_info.cell_position == "last":
                filtered.append(location)

    return filtered


def parse_row_padding_requirements(metadata: dict, csv_path: str, row_id: int, row_cells: list[str], all_row_ids: list[int]) -> dict[str, SkycellPaddingInfo]:
    """Parse padding requirements for a specific row with filtering."""
    df = load_csv_data(csv_path)
    current_projection = metadata["projection"]

    # Get row position context
    row_position = determine_row_position(row_id, all_row_ids)

    # Get cell position analysis
    cell_positions = analyze_cell_positions(row_cells)

    # Get all rows for current projection matching this row_id
    proj_df = df[df["projection"].astype(str) == current_projection]
    row_df = proj_df[proj_df["y"] == row_id]

    padding_columns = ["pad_skycell_top", "pad_skycell_bottom", "pad_skycell_left", "pad_skycell_right", "pad_skycell_top_left", "pad_skycell_top_right", "pad_skycell_bottom_left", "pad_skycell_bottom_right"]

    # Build skycell to locations mapping
    skycell_requirements = {}

    for _, row_data in row_df.iterrows():
        current_cell_name = row_data["NAME"]

        # Only process cells that are actually in this row
        if current_cell_name not in row_cells:
            continue

        cell_info = cell_positions[current_cell_name]

        for column in padding_columns:
            padding_value = row_data.get(column, "")
            if not padding_value or str(padding_value) == "nan" or not str(padding_value).strip():
                continue

            # Handle slash-separated cells
            cell_names = str(padding_value).split("/")
            location = column.replace("pad_skycell_", "")

            for cell_name in cell_names:
                cell_name = cell_name.strip()
                if not cell_name:
                    continue

                # Extract projection from cell name
                try:
                    cell_projection = cell_name.split(".")[1]
                except IndexError:
                    logger.warning(f"Could not extract projection from cell name: {cell_name}")
                    continue

                # Only include cells from different projections
                if cell_projection == current_projection:
                    continue

                else:
                    # Only include padding that matches row and cell positions:
                    # - bottom row gets "bottom"
                    # - top row gets "top"
                    # - first cell gets "left"
                    # - last cell gets "right"
                    # - corners only when both row_position and cell_position match
                    position_type = cell_info  # "first", "last", or "interior"

                    if location == "top" and row_position != "top":
                        continue
                    if location == "bottom" and row_position != "bottom":
                        continue
                    if location == "left" and position_type != "first":
                        continue
                    if location == "right" and position_type != "last":
                        continue
                    if location in ("top_left", "top_right"):
                        if row_position != "top":
                            continue
                        if location.endswith("left") and position_type != "first":
                            continue
                        if location.endswith("right") and position_type != "last":
                            continue
                    if location in ("bottom_left", "bottom_right"):
                        if row_position != "bottom":
                            continue
                        if location.endswith("left") and position_type != "first":
                            continue
                        if location.endswith("right") and position_type != "last":
                            continue

                    # Register this padding skycell and its approved location
                    if cell_name not in skycell_requirements:
                        skycell_requirements[cell_name] = {"main_skycells": [], "all_locations": []}
                    skycell_requirements[cell_name]["all_locations"].append(location)
                    skycell_requirements[cell_name]["main_skycells"].append(current_cell_name)

    return skycell_requirements


def analyze_cross_row_sharing(current_row_padding: dict[str, SkycellPaddingInfo], next_row_padding: dict[str, SkycellPaddingInfo]) -> dict[str, dict]:
    """Placeholder for cross-row sharing analysis"""
    # For now, return empty dict - will implement complex sharing logic later
    shared = {}
    for skycell_name in current_row_padding:
        if skycell_name in next_row_padding:
            shared[skycell_name] = {"locations": list(set(current_row_padding[skycell_name].locations + next_row_padding[skycell_name].locations)), "sharing_type": "cross_row"}
    return shared


# Main integration function for modern_sliding_window.py
def load_modern_cross_projection_padding(state, config, metadata: dict, current_row_id: int, zarr_path: str) -> None:
    """Modern replacement for load_cross_projection_padding in modern_sliding_window.py"""
    try:
        # Get current row cells
        csv_path = metadata.get("csv_path")
        if not csv_path:
            logger.warning("No CSV path in metadata, skipping cross-projection padding")
            return

        df = load_csv_data(csv_path)
        current_projection = metadata["projection"]

        # Get all row IDs for context
        proj_df = df[df["projection"].astype(str) == current_projection]
        all_row_ids = sorted(proj_df["y"].unique())

        # Get current row cells
        row_df = proj_df[proj_df["y"] == current_row_id]
        current_row_cells = row_df["NAME"].tolist()

        if not current_row_cells:
            logger.warning(f"No cells found for row {current_row_id} in projection {current_projection}")
            return

        # Parse padding requirements
        padding_requirements = parse_row_padding_requirements(metadata, csv_path, current_row_id, current_row_cells, all_row_ids)

        if not padding_requirements:
            logger.info(f"No cross-projection padding needed for row {current_row_id}")
            return

        logger.info(f"Processing {len(padding_requirements)} cross-projection padding skycells for row {current_row_id}")

        # Process each padding skycell
        for skycell_name, padding_info in padding_requirements.items():
            try:
                # Load the skycell data
                logger.info(f"Loading padding skycell: {skycell_name} for locations: {padding_info.locations}")

                # Extract projection and skycell from full name (e.g., "skycell.2586.003" -> "2586", "003")
                if skycell_name.startswith("skycell."):
                    parts = skycell_name.split(".")
                    if len(parts) >= 3:
                        projection = parts[1]
                        skycell = parts[2]
                    else:
                        logger.warning(f"Invalid skycell name format: {skycell_name}")
                        continue
                else:
                    # Assume it's already in short format, use the projection from padding_info
                    projection = padding_info.projection
                    skycell = skycell_name

                data, mask, headers = load_skycell_bands_masks_and_headers(zarr_path, projection, skycell)

                if data is None or mask is None:
                    logger.warning(f"Failed to load data for padding skycell: {skycell_name}")
                    continue

                # Convert dictionaries to numpy arrays using band_utils
                from band_utils import process_skycell_bands

                combined_data, combined_mask = process_skycell_bands(data, mask)

                # Process to master projection with proper shape
                target_shape = (PAD_SIZE, PAD_SIZE)  # Default padding region size

                # Create WCS for the master array (need metadata and current row)
                from modern_sliding_window import create_master_array_wcs

                target_wcs = create_master_array_wcs(metadata, config, current_row_id)

                # Use the reproject function from modern_sliding_window.py
                from modern_sliding_window import reproject_cell_to_master

                result = reproject_cell_to_master(combined_data, combined_mask, skycell_name, {"projection": padding_info.projection}, target_wcs, target_shape)

                if result is None:
                    logger.warning(f"Failed to reproject padding skycell: {skycell_name}")
                    continue

                projected_data, projected_mask = result

                # Place into master array padding regions
                place_padding_in_master_array(state, config, projected_data, projected_mask, padding_info.locations)

                logger.info(f"Successfully processed padding skycell: {skycell_name}")

            except Exception as e:
                logger.error(f"Error processing padding skycell {skycell_name}: {e}")
                continue

    except Exception as e:
        logger.error(f"Error in modern cross-projection padding for row {current_row_id}: {e}")


def place_padding_in_master_array(state, config, data: np.ndarray, mask: np.ndarray, locations: list[str]) -> None:
    """Place reprojected data into appropriate padding regions of master array"""

    # Location coordinate mapping
    location_coords = {
        "top": (0, PAD_SIZE, 0, config.width),
        "bottom": (PAD_SIZE + config.cell_height, config.height, 0, config.width),
        "left": (0, config.height, 0, PAD_SIZE),
        "right": (0, config.height, config.width - PAD_SIZE, config.width),
        "top_left": (0, PAD_SIZE, 0, PAD_SIZE),
        "top_right": (0, PAD_SIZE, config.width - PAD_SIZE, config.width),
        "bottom_left": (PAD_SIZE + config.cell_height, config.height, 0, PAD_SIZE),
        "bottom_right": (PAD_SIZE + config.cell_height, config.height, config.width - PAD_SIZE, config.width),
    }

    for location in locations:
        if location not in location_coords:
            logger.warning(f"Unknown padding location: {location}")
            continue

        y_start, y_end, x_start, x_end = location_coords[location]

        # Ensure we don't exceed array bounds
        y_start = max(0, y_start)
        y_end = min(config.height, y_end)
        x_start = max(0, x_start)
        x_end = min(config.width, x_end)

        target_height = y_end - y_start
        target_width = x_end - x_start

        if target_height <= 0 or target_width <= 0:
            logger.warning(f"Invalid target region for location {location}: ({y_start}:{y_end}, {x_start}:{x_end})")
            continue

        # Resize data to fit target region if needed
        if data.shape[-2:] != (target_height, target_width):
            # For now, use simple slicing - could implement proper resampling later
            data_resized = data[:, :target_height, :target_width]
            mask_resized = mask[:, :target_height, :target_width]
        else:
            data_resized = data
            mask_resized = mask

        try:
            # Place data in master array
            state.master_array[:, y_start:y_end, x_start:x_end] = data_resized
            state.master_mask[:, y_start:y_end, x_start:x_end] = mask_resized

            logger.debug(f"Placed padding data in location {location}: ({y_start}:{y_end}, {x_start}:{x_end})")

        except Exception as e:
            logger.error(f"Error placing padding data in location {location}: {e}")
            continue


def calculate_optimal_target_shape(locations: list[str], config) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    """Calculate minimal bounding rectangle covering all locations."""
    if not locations:
        return (0, 0), (0, 0, 0, 0)

    y_ranges = []
    x_ranges = []

    for location in locations:
        y_start, y_end, x_start, x_end = LOCATION_COORDINATES[location](config)
        y_ranges.append((y_start, y_end))
        x_ranges.append((x_start, x_end))

    # Find union bounding box
    min_y = min(y_start for y_start, y_end in y_ranges)
    max_y = max(y_end for y_start, y_end in y_ranges)
    min_x = min(x_start for x_start, x_end in x_ranges)
    max_x = max(x_end for x_start, x_end in x_ranges)

    target_shape = (max_y - min_y, max_x - min_x)
    target_bounds = (min_y, max_y, min_x, max_x)

    return target_shape, target_bounds


def create_cell_wcs(cell_name: str, metadata: dict) -> WCS:
    """Create WCS object for a specific cell."""
    proj_df = metadata["dataframe"]
    cell_row = proj_df[proj_df["NAME"] == cell_name]

    if cell_row.empty:
        raise ValueError(f"Cell {cell_name} not found in metadata")

    cell_data = cell_row.iloc[0]

    # Extract WCS parameters
    crval1 = float(cell_data.get("CRVAL1", 0.0))
    crval2 = float(cell_data.get("CRVAL2", 0.0))
    crpix1 = float(cell_data.get("CRPIX1", 0.0))
    crpix2 = float(cell_data.get("CRPIX2", 0.0))
    cd1_1 = float(cell_data.get("CD1_1", -1.0 / 3600))
    cd1_2 = float(cell_data.get("CD1_2", 0.0))
    cd2_1 = float(cell_data.get("CD2_1", 0.0))
    cd2_2 = float(cell_data.get("CD2_2", 1.0 / 3600))

    # Create WCS object
    cell_wcs = WCS(naxis=2)
    cell_wcs.wcs.crval = [crval1, crval2]
    cell_wcs.wcs.crpix = [crpix1, crpix2]
    cell_wcs.wcs.cd = [[cd1_1, cd1_2], [cd2_1, cd2_2]]
    cell_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    cell_wcs.wcs.cunit = ["deg", "deg"]

    return cell_wcs


def create_master_array_wcs(metadata: dict, config, current_row_id: int) -> WCS:
    """Create WCS object for the master array based on first cell + padding offset."""
    # Get first cell's WCS information from the dataframe
    proj_df = metadata["dataframe"]
    proj_df = proj_df[proj_df["projection"] == metadata["projection"]]
    proj_df = proj_df[proj_df["y"] == current_row_id]
    proj_df = proj_df.sort_values(by=["x"])
    first_cell = proj_df.iloc[0]

    # Extract WCS parameters
    crval1 = float(first_cell.get("CRVAL1", 0.0))
    crval2 = float(first_cell.get("CRVAL2", 0.0))
    crpix1 = float(first_cell.get("CRPIX1", 0.0))
    crpix2 = float(first_cell.get("CRPIX2", 0.0))
    cd1_1 = float(first_cell.get("CD1_1", -1.0 / 3600))
    cd1_2 = float(first_cell.get("CD1_2", 0.0))
    cd2_1 = float(first_cell.get("CD2_1", 0.0))
    cd2_2 = float(first_cell.get("CD2_2", 1.0 / 3600))

    # Create WCS object
    master_wcs = WCS(naxis=2)
    master_wcs.wcs.crval = [crval1, crval2]
    master_wcs.wcs.crpix = [crpix1 + PAD_SIZE, crpix2 + PAD_SIZE]  # Adjust for padding
    master_wcs.wcs.cd = [[cd1_1, cd1_2], [cd2_1, cd2_2]]
    master_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    master_wcs.wcs.cunit = ["deg", "deg"]

    return master_wcs


def reproject_skycell_to_master(cell_data: np.ndarray, cell_mask: np.ndarray, source_cell_name: str, source_metadata: dict, target_wcs: WCS, target_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Reproject a cell from its native projection to the master array coordinate system."""
    try:
        # Create source WCS
        source_wcs = create_cell_wcs(source_cell_name, source_metadata)

        # Reproject data
        reprojected_data, footprint = reproject_interp((cell_data, source_wcs), target_wcs, target_shape, order="linear")

        # Reproject mask
        reprojected_mask_float, _ = reproject_interp((cell_mask.astype(np.float32), source_wcs), target_wcs, target_shape, order="nearest")

        # Convert mask back to uint8, threshold at 0.5
        reprojected_mask = (reprojected_mask_float > 0.5).astype(np.uint8)

        # Handle NaN values in reprojected data
        if np.any(np.isnan(reprojected_data)):
            nan_mask = np.isnan(reprojected_data)
            reprojected_data[nan_mask] = 0.0
            reprojected_mask[nan_mask] = 1  # Mark NaN regions as masked

        logger.debug(f"Successfully reprojected cell {source_cell_name} to target shape {target_shape}")
        return reprojected_data, reprojected_mask

    except Exception as e:
        logger.error(f"Reprojection failed for cell {source_cell_name}: {e}")
        # Fallback: simple resize/crop
        height, width = target_shape
        if cell_data.shape[0] >= height and cell_data.shape[1] >= width:
            reprojected_data = cell_data[:height, :width]
            reprojected_mask = cell_mask[:height, :width]
        else:
            # Pad if too small
            reprojected_data = np.zeros((height, width), dtype=cell_data.dtype)
            reprojected_mask = np.zeros((height, width), dtype=np.uint8)
            h_copy = min(height, cell_data.shape[0])
            w_copy = min(width, cell_data.shape[1])
            reprojected_data[:h_copy, :w_copy] = cell_data[:h_copy, :w_copy]
            reprojected_mask[:h_copy, :w_copy] = cell_mask[:h_copy, :w_copy]

        return reprojected_data, reprojected_mask


def place_reprojected_in_multiple_locations(reprojected_data: np.ndarray, target_bounds: tuple[int, int, int, int], locations: list[str], target_array: np.ndarray, config) -> None:
    """Place reprojected data in multiple locations within the master array."""
    min_y, max_y, min_x, max_x = target_bounds

    for location in locations:
        loc_y_start, loc_y_end, loc_x_start, loc_x_end = LOCATION_COORDINATES[location](config)

        # Calculate region within reprojected array
        rel_y_start = max(0, loc_y_start - min_y)
        rel_y_end = min(reprojected_data.shape[0], loc_y_end - min_y)
        rel_x_start = max(0, loc_x_start - min_x)
        rel_x_end = min(reprojected_data.shape[1], loc_x_end - min_x)

        # Adjust target coordinates if needed
        actual_y_start = loc_y_start + max(0, min_y - loc_y_start)
        actual_y_end = loc_y_end - max(0, loc_y_end - max_y)
        actual_x_start = loc_x_start + max(0, min_x - loc_x_start)
        actual_x_end = loc_x_end - max(0, loc_x_end - max_x)

        # Extract and place
        if rel_y_end > rel_y_start and rel_x_end > rel_x_start and actual_y_end > actual_y_start and actual_x_end > actual_x_start:
            region_data = reprojected_data[rel_y_start:rel_y_end, rel_x_start:rel_x_end]

            # Only overwrite NaN or zero values, preserve existing valid data
            target_region = target_array[actual_y_start:actual_y_end, actual_x_start:actual_x_end]

            # Create mask for values to update (NaN or zero in target, valid in source)
            target_is_invalid = np.isnan(target_region) | (target_region == 0.0)
            source_is_valid = ~np.isnan(region_data) & (region_data != 0.0)
            update_mask = target_is_invalid & source_is_valid

            target_region[update_mask] = region_data[update_mask]

            logger.debug(f"Placed {location} region: target[{actual_y_start}:{actual_y_end}, {actual_x_start}:{actual_x_end}] from reprojected[{rel_y_start}:{rel_y_end}, {rel_x_start}:{rel_x_end}]")


def initialize_padding_regions_with_nan(target_array: np.ndarray, config) -> None:
    """Initialize all padding regions with np.nan."""
    # Top padding
    target_array[0:PAD_SIZE, :] = np.nan
    # Bottom padding
    target_array[PAD_SIZE + config.cell_height :, :] = np.nan
    # Left padding
    target_array[:, 0:PAD_SIZE] = np.nan
    # Right padding
    target_array[:, config.width - PAD_SIZE :] = np.nan


def load_and_reproject_skycell(skycell_info: SkycellPaddingInfo, zarr_path: str, target_wcs: WCS, target_shape: tuple[int, int], source_metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load a skycell and reproject it to master coordinate system."""
    try:
        # Load skycell data with headers
        bands_data, masks_data, headers_data = load_skycell_bands_masks_and_headers(zarr_path, skycell_info.projection, skycell_info.skycell_name)
        cell_data, cell_mask = process_skycell_bands(bands_data, masks_data, headers_data)

        # Reproject to master coordinate system
        reprojected_data, reprojected_mask = reproject_skycell_to_master(cell_data, cell_mask, skycell_info.skycell_name, source_metadata, target_wcs, target_shape)

        logger.info(f"Successfully loaded and reprojected {skycell_info.skycell_name}")
        return reprojected_data, reprojected_mask

    except Exception as e:
        logger.warning(f"Failed to load/reproject {skycell_info.skycell_name}: {e}")
        # Return empty arrays with NaN
        reprojected_data = np.full(target_shape, np.nan, dtype=np.float32)
        reprojected_mask = np.ones(target_shape, dtype=np.uint8)  # Fully masked
        return reprojected_data, reprojected_mask


def load_cross_projection_padding_optimized(state, config, metadata: dict, current_row_id: int, next_row_id: Optional[int], zarr_path: str, csv_path: str, reprojection_cache: dict[str, dict] = None) -> dict[str, dict]:
    """Load cross-projection padding using optimized multi-skycell strategy."""
    if reprojection_cache is None:
        reprojection_cache = {}

    # Get row information
    rows = metadata["rows"]
    all_row_ids = list(rows.keys())
    current_row_cells = rows.get(current_row_id, [])
    next_row_cells = rows.get(next_row_id, []) if next_row_id else []

    # Parse padding requirements for both rows
    current_row_padding = parse_row_padding_requirements(metadata, csv_path, current_row_id, current_row_cells, all_row_ids)

    next_row_padding = {}
    if next_row_id:
        next_row_padding = parse_row_padding_requirements(metadata, csv_path, next_row_id, next_row_cells, all_row_ids)

    # Analyze cross-row sharing
    shared_skycells = analyze_cross_row_sharing(current_row_padding, next_row_padding)

    # Initialize padding regions with NaN
    initialize_padding_regions_with_nan(state.current_array, config)
    if next_row_id:
        initialize_padding_regions_with_nan(state.next_array, config)

    # Create master WCS
    master_wcs = create_master_array_wcs(metadata, config, current_row_id)

    # Process immediate skycells (not shared)
    for skycell_name, skycell_info in current_row_padding.items():
        if skycell_name in shared_skycells:
            continue  # Will be processed in shared section

        # Check cache first
        if skycell_name in reprojection_cache:
            cached_data = reprojection_cache[skycell_name]
            logger.info(f"Using cached reprojection for {skycell_name}")

            place_reprojected_in_multiple_locations(cached_data["reprojected_data"], cached_data["target_bounds"], skycell_info.locations, state.current_array, config)
        else:
            # Calculate optimal target shape for this skycell's locations
            target_shape, target_bounds = calculate_optimal_target_shape(skycell_info.locations, config)

            if target_shape[0] > 0 and target_shape[1] > 0:
                # Load source metadata
                try:
                    source_metadata = {"dataframe": load_csv_data(csv_path)}  # Simplified for now

                    # Load and reproject
                    reprojected_data, reprojected_mask = load_and_reproject_skycell(skycell_info, zarr_path, master_wcs, target_shape, source_metadata)

                    # Place in current array
                    place_reprojected_in_multiple_locations(reprojected_data, target_bounds, skycell_info.locations, state.current_array, config)

                except Exception as e:
                    logger.warning(f"Failed to process immediate skycell {skycell_name}: {e}")

    # Process shared skycells
    for skycell_name, shared_info in shared_skycells.items():
        # Use union of locations from both rows
        union_locations = shared_info["union_locations"]
        target_shape, target_bounds = calculate_optimal_target_shape(union_locations, config)

        if target_shape[0] > 0 and target_shape[1] > 0:
            try:
                source_metadata = {"dataframe": load_csv_data(csv_path)}  # Simplified for now

                # Load and reproject once for both rows
                reprojected_data, reprojected_mask = load_and_reproject_skycell(shared_info["current_info"], zarr_path, master_wcs, target_shape, source_metadata)

                # Place in current array
                place_reprojected_in_multiple_locations(reprojected_data, target_bounds, shared_info["current_locations"], state.current_array, config)

                # Place in next array if available
                if next_row_id:
                    place_reprojected_in_multiple_locations(reprojected_data, target_bounds, shared_info["next_locations"], state.next_array, config)

                # Cache for future use
                reprojection_cache[skycell_name] = {"reprojected_data": reprojected_data, "reprojected_mask": reprojected_mask, "target_bounds": target_bounds, "target_shape": target_shape}

                logger.info(f"Processed shared skycell {skycell_name} for both current and next rows")

            except Exception as e:
                logger.warning(f"Failed to process shared skycell {skycell_name}: {e}")

    # Clean up old cache entries to manage memory
    if len(reprojection_cache) > 5:  # Keep only recent entries
        oldest_keys = list(reprojection_cache.keys())[:-3]  # Keep last 3
        for key in oldest_keys:
            del reprojection_cache[key]

    logger.info(f"Completed optimized padding for row {current_row_id} with {len(current_row_padding)} skycells")

    return reprojection_cache
