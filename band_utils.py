"""
Simple band combination utilities.

Function-oriented approach for PS1 r,i,z,y band combination.
"""

import logging
import multiprocessing

import numpy as np
import sep
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
        logger.warning(f"[Band] Failed to parse header, using defaults: {e}")
        return 1000.0, 1000.0, 1.0


def apply_flux_conversion(data: np.ndarray, boffset: float, bsoften: float, exptime: float, std: bool = False) -> np.ndarray:
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
    val = flux if not std else np.sqrt(flux)
    return val / exptime


def _process_single_band(band_data, weight, header_str=None):
    """Worker function to process one band. For parallel execution."""
    band_data = band_data.astype(np.float32)

    # Apply flux conversion if header is provided
    if header_str:
        boffset, bsoften, exptime = extract_header_values(header_str)
        band_data = apply_flux_conversion(band_data, boffset, bsoften, exptime)
    # If no header, use default flux conversion values
    else:
        logger.warning("[Band] No header data available for a band, using default flux conversion.")
        band_data = apply_flux_conversion(band_data)  # Uses defaults

    # Return the weighted contribution
    return band_data * weight


def combine_rizy_bands_parallel(bands_data: dict[str, np.ndarray], weights: list[float] = None, apply_flux_conv: bool = True, headers_data: dict[str, str] = None) -> np.ndarray:
    """
    Combine r,i,z,y bands into a single image in parallel using 4 processes.
    """
    if weights is None:
        weights = [0.238, 0.344, 0.283, 0.135]  # r, i, z, y

    bands = ["r", "i", "z", "y"]

    tasks = []
    for i, band in enumerate(bands):
        if band not in bands_data:
            logger.warning(f"[Band] Missing band {band}, skipping")
            continue

        current_band_data = bands_data[band]
        current_weight = weights[i]
        header_str = None

        if apply_flux_conv and headers_data and band in headers_data:
            header_str = headers_data[band]
            logger.debug(f"Queuing band {band} for processing with its header.")
        elif apply_flux_conv:
            logger.warning(f"Band {band}: no header data available, will use defaults.")

        tasks.append((current_band_data, current_weight, header_str))

    if not tasks:
        raise ValueError("No valid bands found in data")

    with multiprocessing.Pool(processes=4) as pool:
        processed_bands = pool.starmap(_process_single_band, tasks)

    combined = np.sum(processed_bands, axis=0)

    logger.debug(f"[Band] Combined {len(processed_bands)} bands, range: [{combined.min():.3f}, {combined.max():.3f}]")
    return combined


def combine_rizy_bands(bands_data: dict[str, np.ndarray], weights: list[float] = None, apply_flux_conv: bool = True, headers_data: dict[str, str] = None, bands_weights: dict[str, float] = None, headers_weight_data: dict[str, str] = None) -> np.ndarray:
    """Combine r,i,z,y bands into single image.

    Args:
        bands_data: Dictionary mapping band names to arrays
        weights: Weights for [r, i, z, y]. Defaults to optimized values.
        apply_flux_conv: Whether to apply flux conversion
        headers_data: Dictionary mapping band names to FITS header strings

    Returns:
        combined_image array
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
    combined_uncert = np.zeros_like(combined, dtype=np.float32)

    # Process and combine each band
    for i, band in enumerate(bands):
        if band not in bands_data:
            logger.warning(f"[Band] Missing band {band}, skipping")
            continue

        band_data = bands_data[band].astype(np.float32)

        if apply_flux_conv:
            # Extract header values for this specific band
            if headers_data and band in headers_data:
                boffset, bsoften, exptime = extract_header_values(headers_data[band])
                logger.debug(f"[Band] Band {band}: using BOFFSET={boffset}, BSOFTEN={bsoften}, EXPTIME={exptime}")
                band_data = apply_flux_conversion(band_data, boffset, bsoften, exptime)
            else:
                logger.warning(f"[Band] Band {band}: no header data available")

        # Add weighted contribution
        combined += band_data * weights[i]

        if bands_weights and band in bands_weights:
            band_weight = bands_weights[band].astype(np.float32)

            if headers_weight_data and band in headers_weight_data:
                boffset_wt, bsoften_wt, exptime_wt = extract_header_values(headers_weight_data[band])
                logger.debug(f"[Band] Band {band} weights: using BOFFSET={boffset_wt}, BSOFTEN={bsoften_wt}, EXPTIME={exptime_wt}")
                band_weight = apply_flux_conversion(band_weight, boffset_wt, bsoften_wt, exptime_wt, std=True)
            else:
                logger.warning(f"[Band] Band {band} weights: no header data available")

            combined_uncert += (band_weight**2) * (weights[i] ** 2)

    combined_uncert = np.sqrt(combined_uncert)
    logger.debug(f"[Band] Combined {len(bands_data)} bands, range: [{combined.min():.3f}, {combined.max():.3f}]")
    return combined, combined_uncert


def combine_masks(masks_data: dict[str, np.ndarray]) -> np.ndarray:
    """
    Combine multiple mask bands using a vectorized bitwise OR.

    Args:
        masks_data: Dictionary mapping band names to mask arrays.

    Returns:
        Combined mask array as uint16, or None if no masks are provided.
    """
    bands = ["r", "i", "z", "y"]

    # 1. Create a list of all mask arrays that exist in the input dict.
    # This is a single pass over the data.
    valid_masks = [masks_data[b] for b in bands if b in masks_data]

    # 2. Handle the case where no valid masks were found.
    if not valid_masks:
        logger.warning("[Band] No masks available to combine.")
        return None

    # 3. Use np.bitwise_or.reduce to combine all arrays in the list at once.
    # This operation is highly optimized and runs in C code.
    # We cast to uint16 once on the result.
    combined = np.bitwise_or.reduce(valid_masks).astype(np.uint16)

    # Note: For a bitmask, counting non-zero elements is a more accurate
    # way to find the number of affected pixels than using .sum().
    masked_pixel_count = np.count_nonzero(combined)
    logger.debug(f"[Band] Combined {len(valid_masks)} masks, {masked_pixel_count} masked pixels")

    return combined


def process_skycell_bands(bands_data: dict[str, np.ndarray], masks_data: dict[str, np.ndarray] = None, weights_data: dict[str, np.ndarray] = None, headers_data: dict[str, str] = None, headers_weight_data: dict[str, str] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process single skycell: combine bands and masks with variance propagation.

    Args:
        bands_data: Dictionary of band arrays
        masks_data: Dictionary of mask arrays (optional)
        weights_data: Dictionary of variance arrays (optional)
        headers_data: Dictionary of FITS header strings (optional)

    Returns:
        Tuple of (combined_image, combined_mask_uint16, combined_uncert)
    """
    # Combine bands with proper flux conversion using headers and variance maps
    combined_image, combined_uncert = combine_rizy_bands(bands_data, headers_data=headers_data, bands_weights=weights_data, headers_weight_data=headers_weight_data)

    # Combine masks if provided
    combined_mask = None
    if masks_data:
        combined_mask = combine_masks(masks_data)

    # Create dummy mask if none provided
    if combined_mask is None:
        combined_mask = np.zeros_like(combined_image, dtype=np.uint16)

    return combined_image, combined_mask, combined_uncert


def remove_background(data: np.ndarray, uncert: np.ndarray = None, sigma: float = 2.5, sigma_mask: float = 20) -> np.ndarray:
    """Remove background from image using SEP.

    Args:
        image: Input image array
        mask: Optional mask array

    Returns:
        Background-subtracted image
    """
    try:
        mask_bright_stars = data > np.nanmedian(uncert) * sigma_mask
        data_s = data.astype(data.dtype.newbyteorder("="))
        uncert_s = uncert.astype(uncert.dtype.newbyteorder("="))
        sep.set_extract_pixstack(1000000)
        objects, segmap = sep.extract(data_s, sigma, err=uncert_s, mask=mask_bright_stars, segmentation_map=True)
        data[np.logical_and(segmap == 0, ~mask_bright_stars)] = 0
    except Exception as e:
        logging.error(f"[Band] SEP extraction failed: {e}")
        return data

    return data
