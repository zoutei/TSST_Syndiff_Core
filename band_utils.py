"""
Simple band combination utilities.

Function-oriented approach for PS1 r,i,z,y band combination.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def apply_flux_conversion(data: np.ndarray, boffset: float = 1000.0, bsoften: float = 1000.0, exptime: float = 1.0) -> np.ndarray:
    """Apply PS1 flux conversion from log scale.

    Args:
        data: Raw data array
        boffset: BOFFSET header value
        bsoften: BSOFTEN header value
        exptime: EXPTIME header value

    Returns:
        Converted flux data
    """
    a = 2.5 / np.log(10)
    x = data / a
    flux = boffset + bsoften * 2 * np.sinh(x)
    return flux / exptime


def detect_needs_flux_conversion(data: np.ndarray) -> bool:
    """Detect if data needs flux conversion based on value ranges.

    Args:
        data: Data array to analyze

    Returns:
        True if flux conversion appears needed
    """
    valid_data = data[~np.isnan(data)]
    if len(valid_data) == 0:
        return False

    min_val = np.min(valid_data)
    max_val = np.max(valid_data)
    median_val = np.median(valid_data)

    # Log-scale data typically: has negatives, large range, median near zero
    has_negatives = min_val < -0.1
    large_range = (max_val - min_val) > 5.0
    median_near_zero = abs(median_val) < 1.0

    return has_negatives and large_range and median_near_zero


def combine_rizy_bands(bands_data: dict[str, np.ndarray], weights: list[float] = None, apply_flux_conv: bool = True) -> np.ndarray:
    """Combine r,i,z,y bands into single image.

    Args:
        bands_data: Dictionary mapping band names to arrays
        weights: Weights for [r, i, z, y]. Defaults to optimized values.
        apply_flux_conv: Whether to apply flux conversion

    Returns:
        Combined image array
    """
    if weights is None:
        weights = [0.238, 0.344, 0.283, 0.135]  # r, i, z, y

    bands = ["r", "i", "z", "y"]

    # Get first available band for shape reference
    first_band = None
    for band in bands:
        if band in bands_data:
            first_band = band
            break

    if first_band is None:
        raise ValueError("No valid bands found in data")

    combined = np.zeros_like(bands_data[first_band], dtype=np.float32)

    # Process and combine each band
    for i, band in enumerate(bands):
        if band not in bands_data:
            logger.warning(f"Missing band {band}, skipping")
            continue

        band_data = bands_data[band].astype(np.float32)

        # Apply flux conversion if needed
        if apply_flux_conv and detect_needs_flux_conversion(band_data):
            logger.debug(f"Applying flux conversion to {band} band")
            band_data = apply_flux_conversion(band_data)

        # Add weighted contribution
        combined += band_data * weights[i]

    logger.debug(f"Combined {len(bands_data)} bands, range: [{combined.min():.3f}, {combined.max():.3f}]")
    return combined


def combine_masks(masks_data: dict[str, np.ndarray]) -> np.ndarray:
    """Combine multiple mask bands using bitwise OR.

    Args:
        masks_data: Dictionary mapping band names to mask arrays

    Returns:
        Combined mask array
    """
    bands = ["r", "i", "z", "y"]

    # Get first available mask for shape reference
    first_mask = None
    for band in bands:
        if band in masks_data:
            first_mask = band
            break

    if first_mask is None:
        logger.warning("No masks available")
        return None

    combined = np.zeros_like(masks_data[first_mask], dtype=bool)

    # Combine masks with bitwise OR
    for band in bands:
        if band in masks_data:
            combined |= masks_data[band].astype(bool)

    logger.debug(f"Combined {len(masks_data)} masks, {combined.sum()} masked pixels")
    return combined


def process_skycell_bands(bands_data: dict[str, np.ndarray], masks_data: dict[str, np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """Process single skycell: combine bands and masks.

    Args:
        bands_data: Dictionary of band arrays
        masks_data: Dictionary of mask arrays (optional)

    Returns:
        Tuple of (combined_image, combined_mask)
    """
    # Combine bands
    combined_image = combine_rizy_bands(bands_data)

    # Combine masks if provided
    combined_mask = None
    if masks_data:
        combined_mask = combine_masks(masks_data)

    # Create dummy mask if none provided
    if combined_mask is None:
        combined_mask = np.zeros_like(combined_image, dtype=bool)

    return combined_image, combined_mask
