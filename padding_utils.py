"""
Simple padding utilities for PS1 data processing.

Function-oriented approach for applying padding with neighbor cells.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def apply_simple_padding(image: np.ndarray, pad_size: int = 300) -> np.ndarray:
    """Apply simple edge padding to an image.

    Args:
        image: Input image array
        pad_size: Padding size in pixels

    Returns:
        Padded image array
    """
    return np.pad(image, pad_size, mode="edge")


def combine_row_with_padding(row_images: list[np.ndarray], padding_images: dict[str, np.ndarray], pad_size: int = 300) -> list[np.ndarray]:
    """Combine row images with padding from neighbor cells.

    Args:
        row_images: List of images in the current row
        padding_images: Dictionary of padding images by direction/cell
        pad_size: Padding size in pixels

    Returns:
        List of padded images for the row
    """
    padded_row = []

    for i, image in enumerate(row_images):
        # For now, use simple edge padding
        # TODO: Implement actual neighbor-based padding using padding_images
        padded = apply_simple_padding(image, pad_size)
        padded_row.append(padded)

        logger.debug(f"Padded cell {i}: {image.shape} -> {padded.shape}")

    return padded_row


def apply_neighbor_padding(center_image: np.ndarray, neighbor_images: dict[str, np.ndarray], pad_size: int = 300) -> np.ndarray:
    """Apply padding using actual neighbor cell data.

    Args:
        center_image: Main image to pad
        neighbor_images: Dict mapping directions to neighbor images
        pad_size: Padding size in pixels

    Returns:
        Padded image with neighbor data
    """
    h, w = center_image.shape
    padded_h, padded_w = h + 2 * pad_size, w + 2 * pad_size

    # Initialize with zeros
    padded = np.zeros((padded_h, padded_w), dtype=center_image.dtype)

    # Place center image
    padded[pad_size : pad_size + h, pad_size : pad_size + w] = center_image

    # Fill padding regions with neighbor data
    for direction, neighbor_img in neighbor_images.items():
        nh, nw = neighbor_img.shape

        if direction == "top":
            # Take bottom part of top neighbor
            if nh >= pad_size:
                src = neighbor_img[-pad_size:, : min(nw, w)]
                dst_h, dst_w = src.shape
                padded[0:dst_h, pad_size : pad_size + dst_w] = src

        elif direction == "bottom":
            # Take top part of bottom neighbor
            if nh >= pad_size:
                src = neighbor_img[:pad_size, : min(nw, w)]
                dst_h, dst_w = src.shape
                padded[pad_size + h : pad_size + h + dst_h, pad_size : pad_size + dst_w] = src

        elif direction == "left":
            # Take right part of left neighbor
            if nw >= pad_size:
                src = neighbor_img[: min(nh, h), -pad_size:]
                dst_h, dst_w = src.shape
                padded[pad_size : pad_size + dst_h, 0:dst_w] = src

        elif direction == "right":
            # Take left part of right neighbor
            if nw >= pad_size:
                src = neighbor_img[: min(nh, h), :pad_size]
                dst_h, dst_w = src.shape
                padded[pad_size : pad_size + dst_h, pad_size + w : pad_size + w + dst_w] = src

    # Fill any remaining padding with edge values
    # Top edge
    if padded[0, pad_size] == 0:  # Not filled by neighbor
        for i in range(pad_size):
            padded[i, pad_size : pad_size + w] = center_image[0, :]

    # Bottom edge
    if padded[padded_h - 1, pad_size] == 0:  # Not filled by neighbor
        for i in range(pad_size):
            padded[pad_size + h + i, pad_size : pad_size + w] = center_image[-1, :]

    # Left edge
    if padded[pad_size, 0] == 0:  # Not filled by neighbor
        for i in range(pad_size):
            padded[pad_size : pad_size + h, i] = center_image[:, 0]

    # Right edge
    if padded[pad_size, padded_w - 1] == 0:  # Not filled by neighbor
        for i in range(pad_size):
            padded[pad_size : pad_size + h, pad_size + w + i] = center_image[:, -1]

    logger.debug(f"Applied neighbor padding: {center_image.shape} -> {padded.shape}")
    return padded


def cross_projection_reproject(source_image: np.ndarray, source_wcs, target_wcs, target_shape: tuple) -> np.ndarray:
    """Reproject image from one projection to another.

    Args:
        source_image: Source image array
        source_wcs: Source WCS
        target_wcs: Target WCS
        target_shape: Target image shape

    Returns:
        Reprojected image
    """
    try:
        from reproject import reproject_interp

        reprojected, _ = reproject_interp((source_image, source_wcs), target_wcs, shape_out=target_shape)

        # Fill NaNs with zeros
        reprojected = np.nan_to_num(reprojected, nan=0.0)

        logger.debug(f"Reprojected {source_image.shape} -> {target_shape}")
        return reprojected

    except ImportError:
        logger.warning("reproject package not available, using simple interpolation")
        # Fallback to simple resizing
        from scipy.ndimage import zoom

        factor_y = target_shape[0] / source_image.shape[0]
        factor_x = target_shape[1] / source_image.shape[1]

        return zoom(source_image, (factor_y, factor_x), order=1)


def smart_padding_with_csv(center_image: np.ndarray, csv_padding_info: dict[str, list[str]], all_images: dict[str, np.ndarray], pad_size: int = 300) -> np.ndarray:
    """Apply smart padding using CSV-specified neighbor information.

    Args:
        center_image: Main image to pad
        csv_padding_info: Padding info from CSV (directions -> neighbor names)
        all_images: Dictionary of all available images by name
        pad_size: Padding size in pixels

    Returns:
        Padded image
    """
    neighbor_images = {}

    # Extract actual neighbor images based on CSV info
    for direction, neighbor_names in csv_padding_info.items():
        if neighbor_names and len(neighbor_names) > 0:
            # Use first available neighbor for this direction
            neighbor_name = neighbor_names[0]
            if neighbor_name in all_images:
                neighbor_images[direction] = all_images[neighbor_name]
                logger.debug(f"Using {neighbor_name} for {direction} padding")

    # Apply neighbor-based padding
    if neighbor_images:
        return apply_neighbor_padding(center_image, neighbor_images, pad_size)
    else:
        # Fallback to simple edge padding
        logger.debug("No neighbors available, using edge padding")
        return apply_simple_padding(center_image, pad_size)
