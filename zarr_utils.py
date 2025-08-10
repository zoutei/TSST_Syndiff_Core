"""
Simple zarr utilities for PS1 data loading and saving.

Function-oriented approach for maximum simplicity.
"""

import logging

import numpy as np
import zarr

logger = logging.getLogger(__name__)


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
    skycell_group = store[projection][skycell]

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
    skycell_group = store[projection][skycell]

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


def save_convolved_results(output_path: str, projection: str, row_id: int, results: dict[str, np.ndarray]) -> None:
    """Save convolved results for a row.

    Args:
        output_path: Output zarr path
        projection: PS1 projection ID
        row_id: Row identifier
        results: Dictionary mapping skycell names to convolved arrays
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

    # Save each convolved skycell
    for skycell_name, convolved_data in results.items():
        if skycell_name in row_group:
            del row_group[skycell_name]

        # Ensure data is numpy array
        if not isinstance(convolved_data, np.ndarray):
            convolved_data = np.array(convolved_data)

        # Simply save the array data
        row_group[skycell_name] = convolved_data

    logger.info(f"Saved {len(results)} convolved cells for {projection}/row_{row_id}")


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
