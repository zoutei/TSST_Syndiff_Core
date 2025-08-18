"""
Simple band combination utilities.

Function-oriented approach for PS1 r,i,z,y band combination.
"""

import logging

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)


def extract_header_values(header_string: str) -> tuple[float, float, float]:
    """Extract BOFFSET, BSOFTEN, and EXPTIME from FITS header string.

    Args:
        header_string: FITS header as string

    Returns:
        Tuple of (boffset, bsoften, exptime)
    """
    try:
        header = fits.Header.fromstring(header_string)
        boffset = float(header["BOFFSET"])
        bsoften = float(header["BSOFTEN"])
        exptime = float(header["EXPTIME"])
        return boffset, bsoften, exptime
    except Exception as e:
        logger.warning(f"Failed to parse header, using defaults: {e}")
        return 1000.0, 1000.0, 1.0


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


def combine_rizy_bands(bands_data: dict[str, np.ndarray], weights: list[float] = None, apply_flux_conv: bool = True, headers_data: dict[str, str] = None) -> np.ndarray:
    """Combine r,i,z,y bands into single image.

    Args:
        bands_data: Dictionary mapping band names to arrays
        weights: Weights for [r, i, z, y]. Defaults to optimized values.
        apply_flux_conv: Whether to apply flux conversion
        headers_data: Dictionary mapping band names to FITS header strings

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

        if apply_flux_conv:
            # Extract header values for this specific band
            if headers_data and band in headers_data:
                boffset, bsoften, exptime = extract_header_values(headers_data[band])
                logger.debug(f"Band {band}: using BOFFSET={boffset}, BSOFTEN={bsoften}, EXPTIME={exptime}")
            else:
                logger.warning(f"Band {band}: no header data available, using defaults")
                boffset, bsoften, exptime = 1000.0, 1000.0, 1.0

            band_data = apply_flux_conversion(band_data, boffset, bsoften, exptime)

        # Add weighted contribution
        combined += band_data * weights[i]

    logger.debug(f"Combined {len(bands_data)} bands, range: [{combined.min():.3f}, {combined.max():.3f}]")
    return combined


def combine_masks(masks_data: dict[str, np.ndarray]) -> np.ndarray:
    """Combine multiple mask bands using bitwise OR.

    Args:
        masks_data: Dictionary mapping band names to mask arrays

    Returns:
        Combined mask array as uint8
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

    combined = np.zeros_like(masks_data[first_mask], dtype=np.uint16)

    # Combine masks with bitwise OR
    for band in bands:
        if band in masks_data:
            combined |= masks_data[band].astype(np.uint16)

    logger.debug(f"Combined {len(masks_data)} masks, {combined.sum()} masked pixels")
    return combined


def process_skycell_bands(bands_data: dict[str, np.ndarray], masks_data: dict[str, np.ndarray] = None, headers_data: dict[str, str] = None) -> tuple[np.ndarray, np.ndarray]:
    """Process single skycell: combine bands and masks.

    Args:
        bands_data: Dictionary of band arrays
        masks_data: Dictionary of mask arrays (optional)
        headers_data: Dictionary of FITS header strings (optional)

    Returns:
        Tuple of (combined_image, combined_mask_uint16)
    """
    # Combine bands with proper flux conversion using headers
    combined_image = combine_rizy_bands(bands_data, headers_data=headers_data)

    # Combine masks if provided
    combined_mask = None
    if masks_data:
        combined_mask = combine_masks(masks_data)

    # Create dummy mask if none provided
    if combined_mask is None:
        combined_mask = np.zeros_like(combined_image, dtype=np.uint16)

    return combined_image, combined_mask
