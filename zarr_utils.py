"""
Simple zarr utilities for PS1 data loading and saving.

Function-oriented approach for maximum simplicity.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import zarr

logger = logging.getLogger(__name__)


def load_skycell_bands_and_masks(zarr_path: str, projection: str, skycell: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load both band data and masks for a single skycell in parallel.

    Args:
        zarr_path: Path to zarr store
        projection: PS1 projection ID
        skycell: Skycell ID

    Returns:
        Tuple of (bands_data, masks_data) dictionaries
    """
    store = zarr.open(zarr_path, mode="r")
    # Handle both short format (e.g., "080") and full format (e.g., "skycell.2556.080")
    if skycell.startswith("skycell."):
        skycell_key = skycell
    else:
        skycell_key = f"skycell.{projection}.{skycell}"
    skycell_group = store[projection][skycell_key]

    bands_data = {}
    masks_data = {}

    # Load bands and masks in parallel
    def load_band_and_mask(band):
        band_data = None
        mask_data = None

        if band in skycell_group:
            band_data = np.array(skycell_group[band])

        mask_key = f"{band}_mask"
        if mask_key in skycell_group:
            mask_data = np.array(skycell_group[mask_key])

        return band, band_data, mask_data

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(load_band_and_mask, band) for band in ["r", "i", "z", "y"]]

        for future in futures:
            band, band_data, mask_data = future.result()
            if band_data is not None:
                bands_data[band] = band_data
            if mask_data is not None:
                masks_data[band] = mask_data

    logger.debug(f"Loaded {len(bands_data)} bands and {len(masks_data)} masks for {projection}/{skycell}")
    return bands_data, masks_data


def load_skycell_bands(zarr_path: str, projection: str, skycell: str) -> dict[str, np.ndarray]:
    """Load all band data for a single skycell.

    Args:
        zarr_path: Path to zarr store
        projection: PS1 projection ID
        skycell: Skycell ID

    Returns:
        Dictionary mapping band names to arrays
    """
    store = zarr.open(zarr_path, mode="r")
    # Handle both short format (e.g., "080") and full format (e.g., "skycell.2556.080")
    if skycell.startswith("skycell."):
        skycell_key = skycell
    else:
        skycell_key = f"skycell.{projection}.{skycell}"
    skycell_group = store[projection][skycell_key]

    bands_data = {}
    for band in ["r", "i", "z", "y"]:
        if band in skycell_group:
            bands_data[band] = np.array(skycell_group[band])

    logger.debug(f"Loaded {len(bands_data)} bands for {projection}/{skycell}")
    return bands_data


def load_skycell_masks(zarr_path: str, projection: str, skycell: str) -> dict[str, np.ndarray]:
    """Load all mask data for a single skycell.

    Args:
        zarr_path: Path to zarr store
        projection: PS1 projection ID
        skycell: Skycell ID

    Returns:
        Dictionary mapping band names to mask arrays
    """
    store = zarr.open(zarr_path, mode="r")
    # Handle both short format (e.g., "080") and full format (e.g., "skycell.2556.080")
    if skycell.startswith("skycell."):
        skycell_key = skycell
    else:
        skycell_key = f"skycell.{projection}.{skycell}"
    skycell_group = store[projection][skycell_key]

    masks_data = {}
    for band in ["r", "i", "z", "y"]:
        mask_key = f"{band}_mask"
        if mask_key in skycell_group:
            masks_data[band] = np.array(skycell_group[mask_key])

    logger.debug(f"Loaded {len(masks_data)} masks for {projection}/{skycell}")
    return masks_data


def load_multiple_skycells(zarr_path: str, skycell_list: list[tuple[str, str]]) -> dict[str, dict[str, np.ndarray]]:
    """Load multiple skycells efficiently.

    Args:
        zarr_path: Path to zarr store
        skycell_list: List of (projection, skycell) tuples

    Returns:
        Dictionary mapping "projection/skycell" to band data
    """
    results = {}

    for projection, skycell in skycell_list:
        key = f"{projection}/{skycell}"
        try:
            bands_data = load_skycell_bands(zarr_path, projection, skycell)
            results[key] = bands_data
        except Exception as e:
            logger.warning(f"Failed to load {key}: {e}")
            continue

    logger.info(f"Loaded {len(results)}/{len(skycell_list)} skycells")
    return results


def save_convolved_results(output_path: str, projection: str, row_id: int, results_data: dict[str, np.ndarray], results_masks: dict[str, np.ndarray]) -> None:
    """Save convolved results and masks for a row.

    Args:
        output_path: Output zarr path
        projection: PS1 projection ID
        row_id: Row identifier
        results_data: Dictionary mapping skycell names to convolved data arrays
        results_masks: Dictionary mapping skycell names to mask arrays
    """
    store = zarr.open(output_path, mode="a")

    # Create projection group if needed
    if projection not in store:
        proj_group = store.create_group(projection)
    else:
        proj_group = store[projection]

    # Create row group
    row_key = f"row_{row_id:03d}"
    if row_key in proj_group:
        del proj_group[row_key]
    row_group = proj_group.create_group(row_key)

    # Create data and mask subgroups
    data_group = row_group.create_group("data")
    mask_group = row_group.create_group("mask")

    # Save each convolved skycell data
    for skycell_name, convolved_data in results_data.items():
        # Ensure data is numpy array
        if not isinstance(convolved_data, np.ndarray):
            convolved_data = np.array(convolved_data)

        # Save the array data
        data_group[skycell_name] = convolved_data

    # Save each mask
    for skycell_name, mask_data in results_masks.items():
        # Ensure mask is uint8 numpy array
        if not isinstance(mask_data, np.ndarray):
            mask_data = np.array(mask_data)

        # Convert to uint8 if needed
        if mask_data.dtype != np.uint8:
            mask_data = mask_data.astype(np.uint8)

        # Save the mask data
        mask_group[skycell_name] = mask_data

    logger.info(f"Saved {len(results_data)} convolved cells and {len(results_masks)} masks for {projection}/row_{row_id}")


def get_available_projections(zarr_path: str) -> list[str]:
    """Get list of available projections in zarr store.

    Args:
        zarr_path: Path to zarr store

    Returns:
        List of projection IDs
    """
    store = zarr.open(zarr_path, mode="r")
    projections = [key for key in store.keys() if key.isdigit()]
    projections.sort()

    logger.info(f"Found {len(projections)} projections")
    return projections


def get_projection_skycells(zarr_path: str, projection: str) -> list[str]:
    """Get list of skycells for a projection.

    Args:
        zarr_path: Path to zarr store
        projection: PS1 projection ID

    Returns:
        List of skycell names
    """
    store = zarr.open(zarr_path, mode="r")
    proj_group = store[projection]
    skycells = list(proj_group.keys())
    skycells.sort()

    logger.debug(f"Found {len(skycells)} skycells in projection {projection}")
    return skycells
