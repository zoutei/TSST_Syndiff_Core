"""
Simple convolution utilities for TESS PSF application.

Function-oriented approach for applying convolution to padded images.
"""

import logging

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


def apply_gaussian_convolution(image: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """Apply Gaussian convolution to simulate TESS PSF.

    Args:
        image: Input image array
        sigma: Gaussian sigma parameter

    Returns:
        Convolved image array
    """
    convolved = ndimage.gaussian_filter(image, sigma=sigma)
    logger.debug(f"Applied Gaussian convolution (sigma={sigma}): {image.shape}")
    return convolved


def create_tess_psf_kernel(size: int = 21, sigma: float = 4.0) -> np.ndarray:
    """Create a simple Gaussian PSF kernel for TESS.

    Args:
        size: Kernel size (should be odd)
        sigma: Gaussian sigma

    Returns:
        Normalized PSF kernel
    """
    if size % 2 == 0:
        size += 1  # Ensure odd size

    # Create coordinate grids
    center = size // 2
    y, x = np.ogrid[-center : center + 1, -center : center + 1]

    # Create Gaussian kernel
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    logger.debug(f"Created TESS PSF kernel: {size}x{size}, sigma={sigma}")
    return kernel


def apply_psf_convolution(image: np.ndarray, psf_kernel: np.ndarray = None) -> np.ndarray:
    """Apply PSF convolution using a kernel.

    Args:
        image: Input image array
        psf_kernel: PSF kernel (if None, creates default Gaussian)

    Returns:
        Convolved image array
    """
    if psf_kernel is None:
        psf_kernel = create_tess_psf_kernel()

    convolved = ndimage.convolve(image, psf_kernel, mode="constant", cval=0.0)
    logger.debug(f"Applied PSF convolution: {image.shape}")
    return convolved


def convolve_row_images(padded_images: list[np.ndarray], sigma: float = 4.0) -> list[np.ndarray]:
    """Apply convolution to a row of padded images.

    Args:
        padded_images: List of padded image arrays
        sigma: Gaussian sigma for convolution

    Returns:
        List of convolved image arrays
    """
    convolved_images = []

    for i, padded_img in enumerate(padded_images):
        convolved = apply_gaussian_convolution(padded_img, sigma)
        convolved_images.append(convolved)
        logger.debug(f"Convolved image {i}: range=[{convolved.min():.3f}, {convolved.max():.3f}]")

    logger.info(f"Convolved {len(padded_images)} images in row")
    return convolved_images


def apply_tess_psf_simulation(image: np.ndarray, psf_type: str = "gaussian", sigma: float = 4.0, kernel_size: int = 21) -> np.ndarray:
    """Apply TESS PSF simulation to an image.

    Args:
        image: Input image array
        psf_type: Type of PSF ('gaussian' or 'kernel')
        sigma: Gaussian sigma
        kernel_size: Size of PSF kernel

    Returns:
        Convolved image with TESS PSF applied
    """
    if psf_type == "gaussian":
        return apply_gaussian_convolution(image, sigma)
    elif psf_type == "kernel":
        kernel = create_tess_psf_kernel(kernel_size, sigma)
        return apply_psf_convolution(image, kernel)
    else:
        logger.warning(f"Unknown PSF type: {psf_type}, using Gaussian")
        return apply_gaussian_convolution(image, sigma)
