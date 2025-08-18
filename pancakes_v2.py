"""
PanCAKES v2.0 - Advanced Astronomical Image Processing Pipeline

This module provides a modern, high-performance implementation for processing
TESS (Transiting Exoplanet Survey Satellite) Full Frame Images and matching them
with PanSTARRS1 (PS1) SkyCell data using advanced computational techniques.

Author: Generated from optimization notebook analysis
Version: 2.0
"""

# Standard library imports
import argparse
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.wcs import WCS, FITSFixedWarning
from mocpy import MOC
from numba import jit
from shapely.geometry import Polygon
from tqdm import tqdm

# Suppress common warnings
warnings.simplefilter("ignore", category=VerifyWarning)
warnings.filterwarnings("ignore", category=FITSFixedWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# PADDING AND SKYCELL UTILITIES
# ============================================================================


@dataclass
class PaddingRequirements:
    """Flags indicating which sides/corners require padding based on TESS mapping."""

    top: bool = False
    bottom: bool = False
    left: bool = False
    right: bool = False
    top_left: bool = False
    top_right: bool = False
    bottom_left: bool = False
    bottom_right: bool = False

    def any_needed(self) -> bool:
        """Check if any side needs padding."""
        return any([self.top, self.bottom, self.left, self.right, self.top_left, self.top_right, self.bottom_left, self.bottom_right])

    def good_side_fail(self):
        """Check if padding is needed for sides that should never need it."""
        return any([self.bottom, self.left, self.top_left, self.bottom_left, self.bottom_right])

    def to_list(self):
        """Convert to list of boolean values."""
        return [self.top, self.right, self.top_right, self.bottom, self.left, self.bottom_left, self.bottom_right, self.top_left]

    def from_list(self, padding_list):
        """Set values from a list of booleans."""
        if len(padding_list) != 8:
            raise ValueError("Padding list must have exactly 8 elements")

        self.top = padding_list[0]
        self.right = padding_list[1]
        self.top_right = padding_list[2]
        self.bottom = padding_list[3]
        self.left = padding_list[4]
        self.bottom_left = padding_list[5]
        self.bottom_right = padding_list[6]
        self.top_left = padding_list[7]

        return self


def get_projection_cell_id(skycell_name):
    """
    Parse skycell name to extract projection, cell, y, and x coordinates.

    Args:
        skycell_name (str): Skycell name in format 'skycell.projection.cell'

    Returns:
        tuple: (projection, cell, y, x)
    """
    _, projection, cell = skycell_name.split(".")
    if _ != "skycell":
        print("Invalid skycell name format")
    projection = int(projection)
    y = int(cell[1])
    x = int(cell[2])
    return projection, cell, y, x


def check_tess_mapping_padding(mapping_data: np.ndarray, pad_distance: int = 500, edge_exclusion: int = 10) -> PaddingRequirements:
    """
    Analyze a TESS mapping FITS to determine which borders require padding.

    Expects that pixels mapped to this skycell are non-zero; edges with any non-zero within
    the first/last pad_distance (excluding a small inner edge) are flagged for padding.

    Args:
        mapping_data (np.ndarray): 2D mapping array
        pad_distance (int): Distance from edge to check for padding requirements
        edge_exclusion (int): Pixels to exclude from very edge

    Returns:
        PaddingRequirements: Object indicating which sides need padding
    """
    padding = PaddingRequirements()
    # Top
    padding.top = bool(np.any(mapping_data[-(pad_distance + edge_exclusion) : -edge_exclusion, edge_exclusion:-edge_exclusion] != -1))
    # Bottom
    padding.bottom = bool(np.any(mapping_data[edge_exclusion : pad_distance + edge_exclusion, edge_exclusion:-edge_exclusion] != -1))
    # Left
    padding.left = bool(np.any(mapping_data[edge_exclusion:-edge_exclusion, edge_exclusion : pad_distance + edge_exclusion] != -1))
    # Right
    padding.right = bool(np.any(mapping_data[edge_exclusion:-edge_exclusion, -(pad_distance + edge_exclusion) : -edge_exclusion] != -1))
    # Corners
    padding.top_left = bool(np.any(mapping_data[-(pad_distance + edge_exclusion) : -edge_exclusion, edge_exclusion : pad_distance + edge_exclusion] != -1))
    padding.top_right = bool(np.any(mapping_data[-(pad_distance + edge_exclusion) : -edge_exclusion, -(pad_distance + edge_exclusion) : -edge_exclusion] != -1))
    padding.bottom_left = bool(np.any(mapping_data[edge_exclusion : pad_distance + edge_exclusion, edge_exclusion : pad_distance + edge_exclusion] != -1))
    padding.bottom_right = bool(np.any(mapping_data[edge_exclusion : pad_distance + edge_exclusion, -(pad_distance + edge_exclusion) : -edge_exclusion] != -1))
    return padding


def get_padding_corners(skycell_row, ps1_wcs, padding_side, pad_size=500, edge_exclusion=10):
    """
    Get the corners of a padding region for a given skycell and padding side.

    Args:
        skycell_row: Row from the skycell DataFrame
        padding_side: One of 'top', 'right', 'top_right', etc.
        ps1_wcs: Optional WCS object (to avoid recreation)
        pad_size: Size of the padding in pixels
        edge_exclusion: Overlap with the original skycell in pixels

    Returns:
        list: List of [RA, DEC] corner coordinates for the padding region
    """
    # Get dimensions needed for padding calculations
    naxis1 = skycell_row["NAXIS1"]
    naxis2 = skycell_row["NAXIS2"]

    # Define coordinates for each padding region
    # Each region extends pad_size pixels outside the edge and edge_exclusion pixels inside
    if padding_side == "top":
        x_coords = [0, naxis1, naxis1, 0]
        y_coords = [naxis2 - edge_exclusion, naxis2 - edge_exclusion, naxis2 + pad_size, naxis2 + pad_size]
    elif padding_side == "right":
        x_coords = [naxis1 - edge_exclusion, naxis1 + pad_size, naxis1 + pad_size, naxis1 - edge_exclusion]
        y_coords = [0, 0, naxis2, naxis2]
    elif padding_side == "bottom":
        x_coords = [0, naxis1, naxis1, 0]
        y_coords = [-pad_size, -pad_size, edge_exclusion, edge_exclusion]
    elif padding_side == "left":
        x_coords = [-pad_size, edge_exclusion, edge_exclusion, -pad_size]
        y_coords = [0, 0, naxis2, naxis2]
    elif padding_side == "top_right":
        x_coords = [naxis1 - edge_exclusion, naxis1 + pad_size, naxis1 + pad_size, naxis1 - edge_exclusion]
        y_coords = [naxis2 - edge_exclusion, naxis2 - edge_exclusion, naxis2 + pad_size, naxis2 + pad_size]
    elif padding_side == "bottom_right":
        x_coords = [naxis1 - edge_exclusion, naxis1 + pad_size, naxis1 + pad_size, naxis1 - edge_exclusion]
        y_coords = [-pad_size, -pad_size, edge_exclusion, edge_exclusion]
    elif padding_side == "bottom_left":
        x_coords = [-pad_size, edge_exclusion, edge_exclusion, -pad_size]
        y_coords = [-pad_size, -pad_size, edge_exclusion, edge_exclusion]
    elif padding_side == "top_left":
        x_coords = [edge_exclusion, -pad_size, -pad_size, edge_exclusion]
        y_coords = [naxis2 - edge_exclusion, naxis2 - edge_exclusion, naxis2 + pad_size, naxis2 + pad_size]
        x_coords = [-pad_size, edge_exclusion, edge_exclusion, -pad_size]
        y_coords = [naxis2 - edge_exclusion, naxis2 - edge_exclusion, naxis2 + pad_size, naxis2 + pad_size]
    else:
        raise ValueError(f"Unknown padding side: {padding_side}")

    # Convert pixel coordinates to world coordinates (RA, DEC)
    world_coords = ps1_wcs.wcs_pix2world(np.vstack([x_coords, y_coords]).T, 0)

    # Return corners in [RA, DEC] format
    return [[ra, dec] for ra, dec in world_coords]


def get_padding_center(corners):
    """Calculate the center point of a padding region."""
    ra_avg = np.mean([corner[0] for corner in corners])
    dec_avg = np.mean([corner[1] for corner in corners])
    return ra_avg, dec_avg


def calculate_distance(ra1, dec1, ra2, dec2):
    """Calculate angular distance between two points in degrees."""
    c1 = SkyCoord(ra1 * u.degree, dec1 * u.degree, frame="icrs")
    c2 = SkyCoord(ra2 * u.degree, dec2 * u.degree, frame="icrs")
    return c1.separation(c2).degree


def calculate_overlap(region1, region2):
    """Calculate the percentage of region1 covered by region2."""
    try:
        intersection = region1.intersection(region2)
        if intersection.is_empty:
            return 0.0
        return (intersection.area / region1.area) * 100.0
    except Exception as e:
        print(f"Error calculating overlap: {e}")
        return 0.0


def find_best_padding_skycell(target_skycell, padding_corners, all_skycells):
    """
    Find the best skycell for padding based on coverage and proximity.

    Args:
        target_skycell: The skycell that needs padding
        padding_corners: The corners of the padding region
        all_skycells: DataFrame of all available skycells

    Returns:
        dict: Results containing best skycell(s) info and coverage analysis
    """
    # Create padding region polygon
    padding_region = Polygon(padding_corners)
    padding_center_ra, padding_center_dec = get_padding_center(padding_corners)

    # PS1 skycell width in degrees (approximate)
    ps1_width = 0.4  # degrees

    # Calculate search radius (diagonal of skycell * sqrt(2))
    search_radius = (ps1_width / 2) * np.sqrt(2)

    # Find all potentially overlapping skycells
    overlapping_candidates = []

    for _, candidate in all_skycells.iterrows():
        # Skip the target skycell itself
        if candidate["NAME"] == target_skycell["NAME"]:
            continue

        # Calculate center-to-center distance
        candidate_center_ra = np.mean([candidate[f"RA_Corner{i}"] for i in range(1, 5)])
        candidate_center_dec = np.mean([candidate[f"DEC_Corner{i}"] for i in range(1, 5)])
        distance = calculate_distance(padding_center_ra, padding_center_dec, candidate_center_ra, candidate_center_dec)

        # Filter by distance to reduce computation
        if distance < search_radius:
            # Create candidate polygon
            candidate_corners = [[candidate[f"RA_Corner{i}"], candidate[f"DEC_Corner{i}"]] for i in range(1, 5)]
            candidate_polygon = Polygon(candidate_corners)

            # Calculate overlap
            coverage = calculate_overlap(padding_region, candidate_polygon)

            if coverage > 0:
                overlapping_candidates.append({"skycell_id": candidate["NAME"], "projection": candidate["projection"], "coverage": coverage, "distance": distance, "polygon": candidate_polygon})

    # Sort by coverage (primary) and distance (secondary)
    overlapping_candidates.sort(key=lambda x: (-x["coverage"], x["distance"]))

    if not overlapping_candidates:
        return {"status": "no_overlap", "best_match": None, "coverage": 0, "combined_solutions": []}

    # Check if we have 100% coverage with the best candidate
    if overlapping_candidates[0]["coverage"] >= 99.9:  # Allow for small numerical errors
        return {"status": "full_coverage", "best_match": overlapping_candidates[0]["skycell_id"], "best_match_proj": overlapping_candidates[0]["projection"], "coverage": overlapping_candidates[0]["coverage"], "distance": overlapping_candidates[0]["distance"], "combined_solutions": []}

    # Try to find combinations that provide full coverage
    combined_solutions = []

    # Try all pairs of candidates
    for i in range(len(overlapping_candidates)):
        for j in range(i + 1, len(overlapping_candidates)):
            combined_polygon = overlapping_candidates[i]["polygon"].union(overlapping_candidates[j]["polygon"])
            combined_coverage = calculate_overlap(padding_region, combined_polygon)

            if combined_coverage >= 99.9:  # Allow for small numerical errors
                combined_solutions.append(
                    {
                        "skycells": [overlapping_candidates[i]["skycell_id"], overlapping_candidates[j]["skycell_id"]],
                        "projections": [overlapping_candidates[i]["projection"], overlapping_candidates[j]["projection"]],
                        "coverage": combined_coverage,
                        "avg_distance": (overlapping_candidates[i]["distance"] + overlapping_candidates[j]["distance"]) / 2,
                    }
                )

    # Sort combined solutions by average distance
    combined_solutions.sort(key=lambda x: x["avg_distance"])

    return {
        "status": "partial_coverage" if not combined_solutions else "combined_coverage",
        "best_match": overlapping_candidates[0]["skycell_id"],
        "best_match_proj": overlapping_candidates[0]["projection"],
        "coverage": overlapping_candidates[0]["coverage"],
        "distance": overlapping_candidates[0]["distance"],
        "combined_solutions": combined_solutions,
    }


def parse_special_padding_flags(flags_str):
    """Parse the special_padding_flags string into a list of booleans."""
    if not isinstance(flags_str, str):
        return [False] * 8

    try:
        flags = eval(flags_str)
        if isinstance(flags, list) and len(flags) == 8:
            return flags
        return [False] * 8
    except Exception:
        return [False] * 8


# ============================================================================
# NUMBA-ACCELERATED COORDINATE TRANSFORMATION FUNCTIONS
# ============================================================================


@jit(nopython=True)
def inverse_tan_projection(xi, eta, crval1, crval2):
    """
    Perform inverse tangential projection using Numba acceleration.

    Args:
        xi (float): Projected x coordinate in degrees
        eta (float): Projected y coordinate in degrees
        crval1 (float): Reference RA in degrees
        crval2 (float): Reference Dec in degrees

    Returns:
        tuple: (RA, Dec) in degrees
    """
    xi_rad = np.deg2rad(xi)
    eta_rad = np.deg2rad(eta)
    ra0 = np.deg2rad(crval1)
    dec0 = np.deg2rad(crval2)

    if np.allclose(xi, 0) and np.allclose(eta, 0):
        return (crval1, crval2)

    rho = np.sqrt(xi_rad**2 + eta_rad**2)
    c = np.arctan(rho)

    sin_c = np.sin(c)
    cos_c = np.cos(c)
    sin_dec0 = np.sin(dec0)
    cos_dec0 = np.cos(dec0)

    dec = np.arcsin(cos_c * sin_dec0 + (eta_rad * sin_c * cos_dec0) / rho)

    y_term = xi_rad * sin_c
    x_term = rho * cos_dec0 * cos_c - eta_rad * sin_dec0 * sin_c
    ra = ra0 + np.arctan2(y_term, x_term)

    return (np.rad2deg(ra), np.rad2deg(dec))


@jit(nopython=True)
def calculate_radec(x, y, crval1, crval2, crpix1, crpix2, pc1_1, pc1_2, pc2_1, pc2_2, cdelt1, cdelt2):
    """
    Calculate RA/Dec from pixel coordinates using WCS parameters.

    Args:
        x, y (float): Pixel coordinates
        crval1, crval2 (float): Reference world coordinates
        crpix1, crpix2 (float): Reference pixel coordinates
        pc1_1, pc1_2, pc2_1, pc2_2 (float): PC matrix elements
        cdelt1, cdelt2 (float): Coordinate deltas

    Returns:
        tuple: (RA, Dec) in degrees
    """
    u = (x - crpix1 + 1) * cdelt1
    v = (y - crpix2 + 1) * cdelt2

    xi = pc1_1 * u + pc1_2 * v
    eta = pc2_1 * u + pc2_2 * v

    ra, dec = inverse_tan_projection(xi, eta, crval1, crval2)
    return ra, dec


@jit(nopython=True)
def calculate_radec_corners_numba(buffer, naxis1, naxis2, crval1, crval2, crpix1, crpix2, pc1_1, pc1_2, pc2_1, pc2_2, cdelt1, cdelt2):
    """
    Calculate RA/Dec for corners of multiple images with buffer.

    Args:
        buffer (float): Buffer size in pixels
        naxis1, naxis2 (array): Image dimensions
        WCS parameters: Arrays of WCS transformation parameters

    Returns:
        ndarray: Shape (N, 4, 2) array of corner coordinates
    """
    if not (naxis1.shape == naxis2.shape == crval1.shape == crval2.shape == crpix1.shape == crpix2.shape == pc1_1.shape == pc1_2.shape == pc2_1.shape == pc2_2.shape == cdelt1.shape == cdelt2.shape):
        raise ValueError("All input arrays must have the same shape")

    ra_dec = np.empty((crval1.shape[0], 4, 2), dtype=np.float64)
    for i in range(crval1.shape[0]):
        x = np.array([buffer, buffer, naxis1[i] - buffer, naxis1[i] - buffer])
        y = np.array([buffer, naxis2[i] - buffer, naxis2[i] - buffer, buffer])
        for c in range(4):
            ra_dec[i, c, 0], ra_dec[i, c, 1] = calculate_radec(x[c], y[c], crval1[i], crval2[i], crpix1[i], crpix2[i], pc1_1[i], pc1_2[i], pc2_1[i], pc2_2[i], cdelt1[i], cdelt2[i])
    return ra_dec


@jit(nopython=True)
def calculate_radec_corners_shift_numba(buffer_large, buffer_small, buffer_normal, cell_x, cell_y, naxis1, naxis2, crval1, crval2, crpix1, crpix2, pc1_1, pc1_2, pc2_1, pc2_2, cdelt1, cdelt2):
    # Check array shapes
    if not (naxis1.shape == naxis2.shape == crval1.shape == crval2.shape == crpix1.shape == crpix2.shape == pc1_1.shape == pc1_2.shape == pc2_1.shape == pc2_2.shape == cdelt1.shape == cdelt2.shape):
        raise ValueError("All input arrays must have the same shape")

    ra_dec = np.empty((crval1.shape[0], 4, 2), dtype=np.float64)
    for i in range(crval1.shape[0]):
        if cell_x[i] == 0:
            x = np.array([buffer_normal, buffer_normal, naxis1[i] - buffer_small, naxis1[i] - buffer_small])
        elif cell_x[i] == 9:
            x = np.array([buffer_large, buffer_large, naxis1[i] - buffer_normal, naxis1[i] - buffer_normal])
        else:
            x = np.array([buffer_large, buffer_large, naxis1[i] - buffer_small, naxis1[i] - buffer_small])

        if cell_y[i] == 0:
            y = np.array([buffer_normal, naxis2[i] - buffer_small, naxis2[i] - buffer_small, buffer_normal])
        elif cell_y[i] == 9:
            y = np.array([buffer_large, naxis2[i] - buffer_normal, naxis2[i] - buffer_normal, buffer_large])
        else:
            y = np.array([buffer_large, naxis2[i] - buffer_small, naxis2[i] - buffer_small, buffer_large])

        for c in range(4):
            ra_dec[i, c, 0], ra_dec[i, c, 1] = calculate_radec(x[c], y[c], crval1[i], crval2[i], crpix1[i], crpix2[i], pc1_1[i], pc1_2[i], pc2_1[i], pc2_2[i], cdelt1[i], cdelt2[i])

    return ra_dec


@jit(nopython=True)
def calculate_radec_center_numba(naxis1, naxis2, crval1, crval2, crpix1, crpix2, pc1_1, pc1_2, pc2_1, pc2_2, cdelt1, cdelt2):
    """
    Calculate RA/Dec for centers of multiple images.

    Args:
        naxis1, naxis2 (array): Image dimensions
        WCS parameters: Arrays of WCS transformation parameters

    Returns:
        ndarray: Shape (N, 2) array of center coordinates
    """
    if not (naxis1.shape == naxis2.shape == crval1.shape == crval2.shape == crpix1.shape == crpix2.shape == pc1_1.shape == pc1_2.shape == pc2_1.shape == pc2_2.shape == cdelt1.shape == cdelt2.shape):
        raise ValueError("All input arrays must have the same shape")

    ra_dec = np.empty((crval1.shape[0], 2), dtype=np.float64)
    for i in range(crval1.shape[0]):
        ra_dec[i, 0], ra_dec[i, 1] = calculate_radec(naxis1[i] // 2, naxis2[i] // 2, crval1[i], crval2[i], crpix1[i], crpix2[i], pc1_1[i], pc1_2[i], pc2_1[i], pc2_2[i], cdelt1[i], cdelt2[i])
    return ra_dec


@jit(nopython=True)
def create_closest_center_array_numba(rust_result_flat, rust_result_lengths, projection_centers_x, projection_centers_y, pixel_coords, total_size):
    """
    Create array mapping each pixel to its closest skycell center.

    Args:
        rust_result_flat (array): Flattened pixel IDs from MOC filtering
        rust_result_lengths (array): Length of each skycell's pixel list
        projection_centers_x, projection_centers_y (array): Skycell projection centers
        pixel_coords (array): Pixel coordinates [y, x] for each pixel
        total_size (int): Total number of pixels

    Returns:
        ndarray: Array mapping pixel IDs to skycell indices
    """
    output_array = np.full(total_size, -1, dtype=np.int32)
    min_distances = np.full(total_size, np.inf, dtype=np.float64)

    start_idx = 0
    for list_idx in range(len(rust_result_lengths)):
        end_idx = start_idx + rust_result_lengths[list_idx]

        center_x = projection_centers_x[list_idx]
        center_y = projection_centers_y[list_idx]

        for i in range(start_idx, end_idx):
            pixel_id = rust_result_flat[i]
            pixel_x = pixel_coords[pixel_id, 1]  # x coordinate
            pixel_y = pixel_coords[pixel_id, 0]  # y coordinate

            # Calculate distance to projection center
            distance = np.sqrt((pixel_x - center_x) ** 2 + (pixel_y - center_y) ** 2)

            # Update if this is the closest center so far
            if distance < min_distances[pixel_id]:
                min_distances[pixel_id] = distance
                output_array[pixel_id] = list_idx

        start_idx = end_idx

    return output_array


@jit(nopython=True)
def create_skycell_pixel_lists_numba(tess_pix_skycell_id_remapped, num_skycells):
    """
    Create efficient pixel lists for each skycell using Numba acceleration.

    Args:
        tess_pix_skycell_id_remapped (array): Pixel to skycell mapping
        num_skycells (int): Number of skycells

    Returns:
        tuple: (flat_pixels, offsets) for efficient skycell pixel access
    """
    # Count pixels per skycell first
    counts = np.zeros(num_skycells, dtype=np.int32)
    for pixel_idx in range(len(tess_pix_skycell_id_remapped)):
        skycell_id = tess_pix_skycell_id_remapped[pixel_idx]
        if skycell_id != -1:
            counts[skycell_id] += 1

    # Calculate cumulative offsets
    offsets = np.zeros(num_skycells + 1, dtype=np.int32)
    for i in range(num_skycells):
        offsets[i + 1] = offsets[i] + counts[i]

    # Create flat array to store all pixel indices
    total_pixels = offsets[num_skycells]
    flat_pixels = np.zeros(total_pixels, dtype=np.int32)

    # Reset counts to use as current position trackers
    current_pos = np.copy(offsets[:-1])

    # Fill the flat array
    for pixel_idx in range(len(tess_pix_skycell_id_remapped)):
        skycell_id = tess_pix_skycell_id_remapped[pixel_idx]
        if skycell_id != -1:
            flat_pixels[current_pos[skycell_id]] = pixel_idx
            current_pos[skycell_id] += 1

    return flat_pixels, offsets


@jit(nopython=True)
def point_in_polygon(x, y, polygon):
    """
    Check if point (x, y) is inside polygon using ray casting algorithm.

    Args:
        x, y (float): Point coordinates
        polygon (array): Polygon vertices

    Returns:
        bool: True if point is inside polygon
    """
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


@jit(nopython=True)
def find_pixels_in_rectangles(coords_ps1_pix, ps1_shape):
    """
    Find integer pixel coordinates that lie within rectangles.

    Args:
        coords_ps1_pix (array): Shape (N, 4, 2) rectangle corner coordinates
        ps1_shape (tuple): (height, width) of PS1 image

    Returns:
        list: Arrays of pixel indices for each rectangle
    """
    height, width = ps1_shape
    result = []

    for rect_idx in range(coords_ps1_pix.shape[0]):
        # Get the 4 corners of the rectangle
        corners = coords_ps1_pix[rect_idx]

        # Find bounding box
        min_x = int(np.floor(np.min(corners[:, 0])))
        max_x = int(np.ceil(np.max(corners[:, 0])))
        min_y = int(np.floor(np.min(corners[:, 1])))
        max_y = int(np.ceil(np.max(corners[:, 1])))

        # Clip to image bounds
        min_x = max(0, min_x)
        max_x = min(width - 1, max_x)
        min_y = max(0, min_y)
        max_y = min(height - 1, max_y)

        # Collect pixels within the rectangle
        pixels_in_rect = []

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Check if point (x, y) is inside the rectangle
                if point_in_polygon(x, y, corners):
                    pixel_idx = y * width + x  # Convert 2D to 1D index
                    pixels_in_rect.append(pixel_idx)

        result.append(np.array(pixels_in_rect))

    return result


@jit(nopython=True)
def populate_array_numba(fll_1d_vec, ps1_pix_in_tess_result, tess_pix_in_skycell):
    """
    Populate output array with TESS pixel indices for each PS1 pixel.

    Args:
        fll_1d_vec (array): Output array to populate
        ps1_pix_in_tess_result (list): Lists of PS1 pixels for each TESS pixel
        tess_pix_in_skycell (array): TESS pixel indices
    """
    for i in range(len(ps1_pix_in_tess_result)):
        ps1_pix_in_tess = ps1_pix_in_tess_result[i]
        tess_ind = tess_pix_in_skycell[i]
        for j in range(len(ps1_pix_in_tess)):
            fll_1d_vec[ps1_pix_in_tess[j]] = tess_ind


# ============================================================================
# HIGH-LEVEL PROCESSING FUNCTIONS
# ============================================================================


def calculate_radec_corners(dataframe_skycells, buffer=120):
    """
    Calculate RA/Dec coordinates for corners of skycells with buffer.

    Args:
        dataframe_skycells (DataFrame): Skycell WCS information
        buffer (float): Buffer size in pixels

    Returns:
        ndarray: Corner coordinates for all skycells
    """
    return calculate_radec_corners_numba(
        buffer,
        dataframe_skycells["NAXIS1"].to_numpy(),
        dataframe_skycells["NAXIS2"].to_numpy(),
        dataframe_skycells["CRVAL1"].to_numpy(),
        dataframe_skycells["CRVAL2"].to_numpy(),
        dataframe_skycells["CRPIX1"].to_numpy(),
        dataframe_skycells["CRPIX2"].to_numpy(),
        dataframe_skycells["PC1_1"].to_numpy(),
        dataframe_skycells["PC1_2"].to_numpy(),
        dataframe_skycells["PC2_1"].to_numpy(),
        dataframe_skycells["PC2_2"].to_numpy(),
        dataframe_skycells["CDELT1"].to_numpy(),
        dataframe_skycells["CDELT2"].to_numpy(),
    )


def calculate_radec_corners_shift(dataframe_skycells, buffer_large=450, buffer_small=20, buffer_normal=200):
    return calculate_radec_corners_shift_numba(
        buffer_large,
        buffer_small,
        buffer_normal,
        dataframe_skycells["x"].to_numpy(),
        dataframe_skycells["y"].to_numpy(),
        dataframe_skycells["NAXIS1"].to_numpy(),
        dataframe_skycells["NAXIS2"].to_numpy(),
        dataframe_skycells["CRVAL1"].to_numpy(),
        dataframe_skycells["CRVAL2"].to_numpy(),
        dataframe_skycells["CRPIX1"].to_numpy(),
        dataframe_skycells["CRPIX2"].to_numpy(),
        dataframe_skycells["PC1_1"].to_numpy(),
        dataframe_skycells["PC1_2"].to_numpy(),
        dataframe_skycells["PC2_1"].to_numpy(),
        dataframe_skycells["PC2_2"].to_numpy(),
        dataframe_skycells["CDELT1"].to_numpy(),
        dataframe_skycells["CDELT2"].to_numpy(),
    )


def calculate_radec_center(dataframe_skycells):
    """
    Calculate RA/Dec coordinates for centers of skycells.

    Args:
        dataframe_skycells (DataFrame): Skycell WCS information

    Returns:
        ndarray: Center coordinates for all skycells
    """
    return calculate_radec_center_numba(
        dataframe_skycells["NAXIS1"].to_numpy(),
        dataframe_skycells["NAXIS2"].to_numpy(),
        dataframe_skycells["CRVAL1"].to_numpy(),
        dataframe_skycells["CRVAL2"].to_numpy(),
        dataframe_skycells["CRPIX1"].to_numpy(),
        dataframe_skycells["CRPIX2"].to_numpy(),
        dataframe_skycells["PC1_1"].to_numpy(),
        dataframe_skycells["PC1_2"].to_numpy(),
        dataframe_skycells["PC2_1"].to_numpy(),
        dataframe_skycells["PC2_2"].to_numpy(),
        dataframe_skycells["CDELT1"].to_numpy(),
        dataframe_skycells["CDELT2"].to_numpy(),
    )


def load_tess_image(tess_file):
    """
    Load TESS image and extract necessary information.

    Args:
        tess_file (str): Path to TESS FITS file

    Returns:
        tuple: (data_shape, wcs, ra_center, dec_center, header, sector, camera_id, ccd_id)
    """
    hdul = fits.open(tess_file)

    try:
        header = deepcopy(hdul[1].header)
        data = hdul[1].data
        wcs = WCS(hdul[1].header)
    except Exception:
        header = deepcopy(hdul[0].header)
        data = hdul[0].data
        wcs = WCS(hdul[0].header)

    data_shape = np.shape(data)
    ra_center, dec_center = wcs.all_pix2world(data_shape[1] / 2, data_shape[0] / 2, 0)

    sector = int(tess_file.split("/")[-1].split("-")[1][1:])
    camera = int(header["CAMERA"])  # camera_id (1-4)
    ccd = int(header["CCD"])  # ccd_id (1-4)

    hdul.close()
    return data_shape, wcs, ra_center, dec_center, header, sector, camera, ccd


def create_tess_pixel_coordinates(data_shape):
    """
    Create coordinate arrays for TESS pixels.

    Args:
        data_shape (tuple): Shape of TESS image

    Returns:
        tuple: (pixel_coordinates, ravelled_indices)
    """
    t_y, t_x = data_shape
    _ty, _tx = np.mgrid[:t_y, :t_x]

    ty_input = _ty.ravel()
    tx_input = _tx.ravel()

    tpix_coord_input = np.column_stack([ty_input, tx_input])
    ravelled_index = np.arange(data_shape[0] * data_shape[1])

    return tpix_coord_input, ravelled_index


def find_relevant_skycells(skycell_wcs_df, tess_wcs, data_shape, tess_buffer=150):
    """
    Find skycells that overlap with TESS image using MOC filtering.

    Args:
        skycell_wcs_df (DataFrame): Skycell WCS information
        tess_wcs (WCS): TESS image WCS
        data_shape (tuple): TESS image shape
        tess_buffer (float): Buffer around TESS image in pixels

    Returns:
        DataFrame: Filtered skycells that overlap with TESS image
    """
    # Create buffered TESS footprint
    tess_ffi_corner = tess_wcs.all_pix2world(
        np.array(
            [
                [-tess_buffer, 0],
                [-tess_buffer, data_shape[0]],
                [0, data_shape[0] + tess_buffer],
                [data_shape[1], data_shape[0] + tess_buffer],
                [data_shape[1] + tess_buffer, data_shape[0]],
                [data_shape[1] + tess_buffer, 0],
                [data_shape[1], -tess_buffer],
                [0, -tess_buffer],
            ]
        ),
        0,
    )

    tess_ffi_skycoord = SkyCoord(ra=tess_ffi_corner[:, 0] * u.deg, dec=tess_ffi_corner[:, 1] * u.deg, frame="icrs")

    tess_ffi_moc = MOC.from_polygon_skycoord(tess_ffi_skycoord, complement=False, max_depth=21)
    sc_mask = tess_ffi_moc.contains_lonlat(skycell_wcs_df["RA"].values * u.degree, skycell_wcs_df["DEC"].values * u.degree)

    return skycell_wcs_df[sc_mask].reset_index(drop=True)


def process_tess_to_skycell_mapping(tess_wcs, data_shape, tpix_coord_input, complete_wcs_skycells, edge_buffer_large=410, edge_buffer_small=70, buffer=200, n_threads=8):
    """
    Create optimized mapping from TESS pixels to skycells.

    Args:
        tess_wcs (WCS): TESS image WCS
        data_shape (tuple): TESS image shape
        tpix_coord_input (array): TESS pixel coordinates
        complete_wcs_skycells (DataFrame): Relevant skycells
        buffer (float): Buffer size for skycell edges
        n_threads (int): Number of threads for MOC processing

    Returns:
        tuple: (selected_skycells, tess_pixel_mapping)
    """
    # Calculate projection centers and skycell corners
    tess_proj_center_x, tess_proj_center_y = tess_wcs.all_world2pix(complete_wcs_skycells["CRVAL1"].to_numpy(), complete_wcs_skycells["CRVAL2"].to_numpy(), 0)
    sc_corners = calculate_radec_corners(complete_wcs_skycells, 50)
    complete_wcs_skycells["RA_Corner1"] = sc_corners[:, 0, 0]
    complete_wcs_skycells["DEC_Corner1"] = sc_corners[:, 0, 1]
    complete_wcs_skycells["RA_Corner2"] = sc_corners[:, 1, 0]
    complete_wcs_skycells["DEC_Corner2"] = sc_corners[:, 1, 1]
    complete_wcs_skycells["RA_Corner3"] = sc_corners[:, 2, 0]
    complete_wcs_skycells["DEC_Corner3"] = sc_corners[:, 2, 1]
    complete_wcs_skycells["RA_Corner4"] = sc_corners[:, 3, 0]
    complete_wcs_skycells["DEC_Corner4"] = sc_corners[:, 3, 1]

    if "projection" not in complete_wcs_skycells.columns:
        parsed = complete_wcs_skycells["NAME"].apply(get_projection_cell_id).tolist()
        cols = pd.DataFrame(parsed, columns=["projection", "cell", "y", "x"])
        complete_wcs_skycells[["projection", "y", "x", "cell"]] = cols[["projection", "y", "x", "cell"]]

    enc_sc_vertices = calculate_radec_corners_shift(complete_wcs_skycells, edge_buffer_large, edge_buffer_small, buffer)
    # enc_sc_vertices_noedge = calculate_radec_corners_shift(complete_wcs_skycells, edge_buffer_large, buffer)

    # Get TESS pixel RA/Dec coordinates
    _x_tess = tpix_coord_input[:, 1]
    _y_tess = tpix_coord_input[:, 0]
    _ra_tess, _dec_tess = tess_wcs.all_pix2world(_x_tess, _y_tess, 0)

    # Use MOC filtering for efficient polygon-point matching
    rust_result = MOC.filter_points_in_polygons(polygons=enc_sc_vertices, pix_ras=_ra_tess, pix_decs=_dec_tess, buffer=0.5, max_depth=21, n_threads=n_threads)
    # rust_result_noedge = MOC.filter_points_in_polygons(polygons=enc_sc_vertices_noedge, pix_ras=_ra_tess, pix_decs=_dec_tess, buffer=0.5, max_depth=21, n_threads=n_threads)

    # Create efficient pixel-to-skycell mapping
    rust_result_flat = np.concatenate([arr for arr in rust_result if len(arr) > 0])
    rust_result_lengths = np.array([len(arr) for arr in rust_result])

    tess_pix_skycell_id = create_closest_center_array_numba(rust_result_flat, rust_result_lengths, tess_proj_center_x, tess_proj_center_y, tpix_coord_input, data_shape[0] * data_shape[1])

    # rust_result_noedge_flat = np.concatenate([arr for arr in rust_result_noedge if len(arr) > 0])
    # rust_result_noedge_lengths = np.array([len(arr) for arr in rust_result_noedge])

    # tess_pix_skycell_id_no_edge = create_closest_center_array_numba(rust_result_noedge_flat, rust_result_noedge_lengths, tess_proj_center_x, tess_proj_center_y, tpix_coord_input, data_shape[0] * data_shape[1])

    # tess_pix_skycell_id = tess_pix_skycell_id_no_edge
    # tess_pix_skycell_id[tess_pix_skycell_id == -1] = tess_pix_skycell_id_all[tess_pix_skycell_id == -1]

    # Remap skycell IDs to consecutive integers
    unique_ids = np.unique(tess_pix_skycell_id[tess_pix_skycell_id != -1])
    id_mapping = np.full(np.max(unique_ids) + 1, -1, dtype=np.int32)
    id_mapping[unique_ids] = np.arange(len(unique_ids), dtype=np.int32)

    mask = tess_pix_skycell_id != -1
    tess_pix_skycell_id_remapped = np.full_like(tess_pix_skycell_id, -1)
    tess_pix_skycell_id_remapped[mask] = id_mapping[tess_pix_skycell_id[mask]]

    # Create selected skycells dataframe with pixel lists
    selected_skycells = complete_wcs_skycells.loc[unique_ids].reset_index(drop=True)

    flat_pixels, offsets = create_skycell_pixel_lists_numba(tess_pix_skycell_id_remapped, len(selected_skycells))

    skycell_pixel_arrays = []
    skycell_pixel_arrays_num_pix = np.zeros(len(selected_skycells), dtype=np.int32)
    for i in range(len(selected_skycells)):
        start_idx = offsets[i]
        end_idx = offsets[i + 1]
        skycell_pixel_arrays.append(flat_pixels[start_idx:end_idx])
        skycell_pixel_arrays_num_pix[i] = end_idx - start_idx

    selected_skycells["pixel_indices"] = skycell_pixel_arrays
    selected_skycells["pixel_indices_num_pix"] = skycell_pixel_arrays_num_pix

    return selected_skycells, tess_pix_skycell_id_remapped


def get_ps1_wcs_information(skycell_data):
    """
    Get WCS information from skycell data (Series, DataFrame row, or dict).

    Args:
        skycell_data (Series): Skycell WCS data

    Returns:
        tuple: (ps1_header, ps1_wcs, ps1_data_shape)
    """

    relevant_keys = ["NAXIS1", "NAXIS2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "PC1_1", "PC1_2", "PC2_1", "PC2_2", "CDELT1", "CDELT2", "RADESYS", "CTYPE1", "CTYPE2"]
    # only keep relevant keys
    header_dict = skycell_data[relevant_keys].to_dict()

    ps1_header = fits.Header(header_dict)
    ps1_data_shape = (int(header_dict["NAXIS2"]), int(header_dict["NAXIS1"]))

    temp_wcs = WCS(ps1_header)

    return ps1_header, temp_wcs, ps1_data_shape


def process_skycell_pixel_mapping(tess_wcs, tpix_coord_input, ps1_wcs, ps1_data_shape, tess_pix_in_skycell):
    """
    Process mapping between TESS pixels and PS1 pixels for a specific skycell.

    Args:
        tess_wcs (WCS): TESS image WCS
        tpix_coord_input (array): TESS pixel coordinates
        ps1_wcs (WCS): PS1 skycell WCS object
        ps1_data_shape (tuple): PS1 skycell image dimensions
        tess_pix_in_skycell (array): TESS pixel indices in this skycell

    Returns:
        ndarray: Mapping array from PS1 pixels to TESS pixels
    """
    # Get TESS pixel coordinates for this skycell
    coords = tpix_coord_input[tess_pix_in_skycell]
    x_coords = coords[:, 1]
    y_coords = coords[:, 0]

    # Calculate TESS pixel corners in world coordinates
    corners = np.array([np.column_stack([x_coords - 0.5, y_coords - 0.5]), np.column_stack([x_coords + 0.5, y_coords - 0.5]), np.column_stack([x_coords + 0.5, y_coords + 0.5]), np.column_stack([x_coords - 0.5, y_coords + 0.5])])  # upper_left  # upper_right  # lower_right  # lower_left

    corners_reshaped = corners.transpose(1, 0, 2).reshape(-1, 2)
    world_coords = tess_wcs.all_pix2world(corners_reshaped, 0)

    # Convert to PS1 pixel coordinates
    coords_ps1_pix = ps1_wcs.all_world2pix(world_coords, 0).reshape(len(tess_pix_in_skycell), 4, 2)

    # Find PS1 pixels within TESS pixel rectangles
    ps1_pix_in_tess_result = find_pixels_in_rectangles(coords_ps1_pix, ps1_data_shape)

    # Create output mapping array
    fll_1d_vec = np.full((ps1_data_shape[0] * ps1_data_shape[1]), -1, dtype=np.int32)
    populate_array_numba(fll_1d_vec, ps1_pix_in_tess_result, tess_pix_in_skycell)

    return fll_1d_vec.reshape(ps1_data_shape[0], ps1_data_shape[1])


def create_master_fits_header(tess_header, file_name):
    """
    Create a master FITS header for the output file.

    Args:
        tess_header (Header): Original TESS header
        file_name (str): Name of the output file

    Returns:
        Header: Processed FITS header
    """
    tess_header_master = deepcopy(tess_header)
    date_mod = datetime.now().strftime("%Y-%m-%d")

    tess_header_master["TESS_FFI"] = file_name
    tess_header_master["DATE-MOD"] = date_mod
    tess_header_master["SOFTWARE"] = "SynDiff"
    tess_header_master["CREATOR"] = "PanCAKES_v2"

    return tess_header_master


def create_fits_header(tess_header, skycell_name=None):
    """
    Create standardized FITS header for output files.

    Args:
        tess_header (Header): Original TESS header
        skycell_name (str, optional): Skycell name to include in header

    Returns:
        Header: Processed FITS header
    """
    dict_for_header = {}
    date_mod = datetime.now().strftime("%Y-%m-%d")

    cols = ["SECTOR", "CAMERA", "CCD", "TELESCOP", "INSTRUME"]
    defaults = ["N/A", 1, 1, "Not specified", "Not specified"]

    for c in range(len(cols)):
        col = cols[c]
        try:
            dict_for_header[col] = tess_header[col]
        except Exception:
            dict_for_header[col] = defaults[c]

    dict_for_header["DATE-MOD"] = date_mod
    dict_for_header["SOFTWARE"] = "SynDiff"
    dict_for_header["CREATOR"] = "PanCAKES_v2"

    if skycell_name:
        dict_for_header["SKYCELL"] = skycell_name

    return fits.Header(dict_for_header)


def save_skycell_mapping(mapping_array, skycell_name, tess_header, ps1_header, output_path, sector, camera_id, ccd_id, overwrite=True):
    """
    Save PS1-to-TESS pixel mapping as compressed FITS file.

    Args:
        mapping_array (ndarray): 2D mapping array
        skycell_name (str): Skycell name
        tess_header (Header): TESS image header
        ps1_header (Header): PS1 skycell header
        output_path (str): Output directory path
        sector (str): TESS sector identifier
        camera_id (int): TESS camera identifier
        ccd_id (int): TESS CCD identifier
        overwrite (bool): Whether to overwrite existing files
    """

    file_name = f"tess_s{sector}_{camera_id}_{ccd_id}_{skycell_name}.fits"

    # Create headers
    base_header = create_fits_header(tess_header, skycell_name)

    new_fits_header = fits.Header()
    new_fits_header["SIMPLE"] = "T"
    new_fits_header += base_header
    new_fits_header_extended = new_fits_header + ps1_header

    # Process mapping array
    mapping_array[mapping_array == -1] = -1  # Ensure -1 for unmapped pixels

    # Create FITS file
    file_path = os.path.join(output_path, f"sector_{sector:04d}", f"camera_{camera_id}", f"ccd_{ccd_id}", file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    primary_hdu = fits.PrimaryHDU(header=new_fits_header)
    image_hdu = fits.ImageHDU(data=np.int64(mapping_array), header=new_fits_header_extended)
    image_hdu.scale("int32", bscale=1.0, bzero=32768.0)
    image_hdu.header["EXTNAME"] = "TESS_PIXEL_MAP"
    image_hdu.header["BSCALE"] = 1.0
    image_hdu.header["BZERO"] = 32768.0

    hdul = fits.HDUList([primary_hdu, image_hdu])
    hdul.verify("fix")
    hdul.writeto(file_path, overwrite=overwrite)

    # Compress file
    compress_cmd = f"gzip -f {file_path}"
    os.system(compress_cmd)


def save_master_mapping(tess_pix_skycell_mapping, selected_skycells, ffi_file_name, tess_header, data_shape, output_path, sector, camera_id, ccd_id, overwrite=True):
    """
    Save master TESS-to-skycell mapping file.

    Args:
        tess_pix_skycell_mapping (ndarray): TESS pixel to skycell mapping
        selected_skycells (DataFrame): Selected skycells information
        tess_header (Header): TESS image header
        data_shape (tuple): TESS image shape
        output_path (str): Output directory path
        sector (str): TESS sector identifier
        camera_id (int): TESS camera identifier
        ccd_id (int): TESS CCD identifier
        overwrite (bool): Whether to overwrite existing files
    """
    # Create filename
    file_name = f"tess_s{sector:04d}_{camera_id}_{ccd_id}_master_pixels2skycells.fits"
    file_name_csv = f"tess_s{sector:04d}_{camera_id}_{ccd_id}_master_skycells_list.csv"

    file_path_csv = os.path.join(output_path, f"sector_{sector:04d}", f"camera_{camera_id}", f"ccd_{ccd_id}", file_name_csv)
    os.makedirs(os.path.dirname(file_path_csv), exist_ok=True)
    selected_skycells.to_csv(file_path_csv)

    # Create header
    master_header = create_master_fits_header(tess_header, ffi_file_name)
    master_header["DATE-MOD"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")

    # Create FITS file
    file_path = os.path.join(output_path, f"sector_{sector:04d}", f"camera_{camera_id}", f"ccd_{ccd_id}", file_name)
    primary_hdu = fits.PrimaryHDU()

    # Reshape mapping to 2D and save
    mapping_2d = tess_pix_skycell_mapping.reshape(data_shape)
    image_hdu = fits.ImageHDU(data=np.int16(mapping_2d), header=master_header)

    # Create skycell table
    table = fits.BinTableHDU.from_columns([fits.Column(name="SKYCELL", format="20A", array=selected_skycells["NAME"].values), fits.Column(name="SKYCIND", format="K", array=np.arange(len(selected_skycells)))])

    hdul = fits.HDUList([primary_hdu, image_hdu, table])
    hdul.writeto(file_path, overwrite=overwrite)

    # Compress file
    compress_cmd = f"gzip -f {file_path}"
    os.system(compress_cmd)


def process_single_skycell(args):
    """
    Process a single skycell mapping task for parallel execution.

    Args:
        args (tuple): Contains (skycell_row, tess_wcs, tpix_coord_input,
                      tess_header, output_path, sector, camera_id, ccd_id, overwrite, all_skycells)

    Returns:
        tuple: (success, skycell_name, padding_info, error_message)
    """
    try:
        (skycell_row, tess_wcs, tpix_coord_input, tess_header, output_path, sector, camera_id, ccd_id, pad_distance, edge_exclusion, overwrite, all_skycells) = args

        skycell_name = skycell_row["NAME"]
        tess_pix_in_skycell = skycell_row["pixel_indices"]

        if len(tess_pix_in_skycell) == 0:
            return (False, skycell_name, {}, "No TESS pixels in skycell")

        # Get PS1 header and WCS information directly from skycell_row
        ps1_header, ps1_wcs, ps1_data_shape = get_ps1_wcs_information(skycell_row)

        # Process skycell pixel mapping
        mapping_array = process_skycell_pixel_mapping(tess_wcs, tpix_coord_input, ps1_wcs, ps1_data_shape, tess_pix_in_skycell)

        # Analyze padding requirements immediately while we have the mapping array
        padding_info = analyze_single_skycell_padding(skycell_name, mapping_array, ps1_wcs, skycell_row, all_skycells, pad_distance=pad_distance, edge_exclusion=edge_exclusion)

        # Save mapping
        save_skycell_mapping(mapping_array, skycell_name, tess_header, ps1_header, output_path, sector, camera_id, ccd_id, overwrite)

        return (True, skycell_name, padding_info, None)

    except Exception as e:
        return (False, skycell_name, {}, str(e))


def analyze_single_skycell_padding(skycell_name, mapping_array, ps1_wcs, skycell_row, all_skycells, pad_distance=500, edge_exclusion=10):
    """
    Analyze padding requirements for a single skycell.

    Args:
        skycell_name (str): Name of the skycell
        mapping_array (ndarray): 2D mapping array for this skycell
        skycell_row (Series, optional): Row data for the skycell, needed for special padding
        all_skycells (DataFrame, optional): DataFrame with all skycells, needed for special padding
        pad_distance (int): Distance from edge to check for padding requirements
        edge_exclusion (int): Pixels to exclude from very edge

    Returns:
        dict: Dictionary with padding information for this skycell
    """
    # Initialize padding info with empty values
    padding_directions = ["top", "right", "top_right", "bottom", "left", "bottom_left", "bottom_right", "top_left"]
    padding_info = {f"pad_skycell_{direction}": "" for direction in padding_directions}
    padding_info.update({"special_padding_needed": False, "edge_pixels_used": False, "good_side_fail": False, "special_padding_flags": [False] * 8})

    # Parse skycell coordinates
    projection, cell, y, x = get_projection_cell_id(skycell_name)

    # Check for edge pixel usage (stricter check)
    check_no_edge_bad_pix = check_tess_mapping_padding(mapping_array, pad_distance=10, edge_exclusion=0)
    if check_no_edge_bad_pix.any_needed():
        padding_info["edge_pixels_used"] = True
        print(f"Warning: Edge pixels are being used for {skycell_name}. This is not good")

    # Check padding requirements
    pad_requirements = check_tess_mapping_padding(mapping_array, pad_distance=pad_distance, edge_exclusion=edge_exclusion)

    # Check if good sides fail
    if pad_requirements.good_side_fail():
        if x < 9 and y < 9 and x > 0 and y > 0:
            padding_info["good_side_fail"] = True
        else:
            if x == 0 and y == 0:
                pass
            elif y == 0:
                if not (pad_requirements.bottom or pad_requirements.bottom_left or pad_requirements.bottom_right):
                    padding_info["good_side_fail"] = True
            elif x == 0:
                if not (pad_requirements.left or pad_requirements.top_left or pad_requirements.bottom_left):
                    padding_info["good_side_fail"] = True
            elif y == 9:
                if not (pad_requirements.top or pad_requirements.top_left or pad_requirements.top_right):
                    padding_info["good_side_fail"] = True
            elif x == 9:
                if not (pad_requirements.right or pad_requirements.top_right or pad_requirements.bottom_right):
                    padding_info["good_side_fail"] = True

    # Track which directions need special padding from other projections
    special_padding_diff_projection = np.zeros(8, dtype=bool)  # top, right, top_right, bottom, left, bottom_left, bottom_right, top_left
    padding_list = pad_requirements.to_list()

    # Get padding directions
    padding_directions = ["top", "right", "top_right", "bottom", "left", "bottom_left", "bottom_right", "top_left"]

    # Get neighboring cells based on position in the grid
    for i, direction in enumerate(padding_directions):
        if padding_list[i]:
            # Default padding approach (within same projection)
            if direction == "top" and y < 9:
                padding_info[f"pad_skycell_{direction}"] = f"skycell.{projection}.0{y + 1}{x}"
            elif direction == "right" and x < 9:
                padding_info[f"pad_skycell_{direction}"] = f"skycell.{projection}.0{y}{x + 1}"
            elif direction == "top_right" and y < 9 and x < 9:
                padding_info[f"pad_skycell_{direction}"] = f"skycell.{projection}.0{y + 1}{x + 1}"
            elif direction == "bottom" and y > 0:
                padding_info[f"pad_skycell_{direction}"] = f"skycell.{projection}.0{y - 1}{x}"
            elif direction == "left" and x > 0:
                padding_info[f"pad_skycell_{direction}"] = f"skycell.{projection}.0{y}{x - 1}"
            elif direction == "bottom_left" and y > 0 and x > 0:
                padding_info[f"pad_skycell_{direction}"] = f"skycell.{projection}.0{y - 1}{x - 1}"
            elif direction == "bottom_right" and y > 0 and x < 9:
                padding_info[f"pad_skycell_{direction}"] = f"skycell.{projection}.0{y - 1}{x + 1}"
            elif direction == "top_left" and y < 9 and x > 0:
                padding_info[f"pad_skycell_{direction}"] = f"skycell.{projection}.0{y + 1}{x - 1}"
            else:
                # This side needs special padding (from different projection)
                special_padding_diff_projection[i] = True

    # If we need special padding and have the necessary inputs
    if np.any(special_padding_diff_projection) and skycell_row is not None and all_skycells is not None:
        padding_info["special_padding_needed"] = True
        padding_info["special_padding_flags"] = special_padding_diff_projection.tolist()

        # Process each direction that needs special padding
        for i, direction in enumerate(padding_directions):
            if special_padding_diff_projection[i]:
                try:
                    # Get padding corners for this direction
                    padding_corners = get_padding_corners(skycell_row, ps1_wcs, direction, pad_size=pad_distance, edge_exclusion=edge_exclusion)

                    # Find the best padding skycell
                    result = find_best_padding_skycell(skycell_row, padding_corners, all_skycells)

                    # Update padding info based on result
                    if result["status"] == "full_coverage":
                        # Single skycell with full coverage
                        padding_info[f"pad_skycell_{direction}"] = result["best_match"]
                    elif result["status"] == "combined_coverage" and result["combined_solutions"]:
                        # Combined solution (take the first one)
                        solution = result["combined_solutions"][0]
                        padding_info[f"pad_skycell_{direction}"] = "/".join(solution["skycells"])
                    elif result["status"] == "partial_coverage":
                        # Best partial coverage
                        padding_info[f"pad_skycell_{direction}"] = result["best_match"]
                except Exception as e:
                    print(f"Error processing {skycell_name} {direction} padding: {e}")
                    padding_info[f"pad_skycell_{direction}"] = "None"

    return padding_info


def update_skycells_with_padding_info(selected_skycells, padding_results):
    """
    Update selected_skycells dataframe with padding information collected from worker threads.

    Args:
        selected_skycells (DataFrame): DataFrame with skycell information
        padding_results (dict): Dictionary mapping skycell names to padding info

    Returns:
        DataFrame: Updated dataframe with padding information
    """

    # Initialize new columns for padding information
    padding_columns = ["pad_skycell_top", "pad_skycell_right", "pad_skycell_top_right", "pad_skycell_bottom", "pad_skycell_left", "pad_skycell_bottom_left", "pad_skycell_bottom_right", "pad_skycell_top_left", "special_padding_needed", "edge_pixels_used", "good_side_fail"]

    for col in padding_columns:
        if col not in selected_skycells.columns:
            selected_skycells[col] = ""

    # Parse projection, cell, y, x for all skycells if not already done
    if "projection" not in selected_skycells.columns:
        parsed = selected_skycells["NAME"].apply(get_projection_cell_id).tolist()
        cols = pd.DataFrame(parsed, columns=["projection", "cell", "y", "x"])
        selected_skycells[["projection", "y", "x", "cell"]] = cols[["projection", "y", "x", "cell"]]

    # Update dataframe with padding information from worker results
    for idx, skycell_row in selected_skycells.iterrows():
        skycell_name = skycell_row["NAME"]

        if skycell_name in padding_results:
            padding_info = padding_results[skycell_name]

            # Update dataframe with padding information
            for key, value in padding_info.items():
                if key == "special_padding_flags":
                    # Store as string representation for CSV compatibility
                    selected_skycells.at[idx, "special_padding_flags"] = str(value)
                else:
                    selected_skycells.at[idx, key] = value

    return selected_skycells


def save_updated_skycell_csv(selected_skycells, output_path, sector, camera_id, ccd_id):
    """
    Save updated CSV file with padding information for processed skycells.

    Args:
        selected_skycells (DataFrame): Processed skycells with padding info
        original_skycell_wcs_df (DataFrame): Original skycell WCS dataframe
        output_path (str): Output directory path
        sector (str): TESS sector identifier
        camera_id (int): TESS camera identifier
        ccd_id (int): TESS CCD identifier
    """
    # Start with the selected_skycells dataframe (only processed skycells)
    # Save updated CSV (only processed skycells)
    file_name_csv = f"tess_s{sector:04d}_{camera_id}_{ccd_id}_master_skycells_list.csv"
    file_path_csv = os.path.join(output_path, f"sector_{sector:04d}", f"camera_{camera_id}", f"ccd_{ccd_id}", file_name_csv)
    os.makedirs(os.path.dirname(file_path_csv), exist_ok=True)

    selected_skycells.to_csv(file_path_csv, index=False)
    print(f"Updated skycell CSV saved to: {file_path_csv}")


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================


def process_tess_image_optimized(tess_file, skycell_wcs_csv, output_path, pad_distance=480, edge_exclusion=10, edge_buffer_large=410, edge_buffer_small=70, buffer=200, tess_buffer=150, n_threads=8, overwrite=True, max_workers=None):
    """
    Main optimized pipeline for processing TESS images with PanSTARRS1 skycells.

    This function implements the complete optimized workflow:
    1. Load TESS image and skycell database
    2. Find relevant skycells using MOC filtering
    3. Create optimized TESS-to-skycell mapping
    4. Process each skycell for PS1-to-TESS pixel mapping
    5. Save all mapping files

    Args:
        tess_file (str): Path to TESS FITS file
        skycell_wcs_csv (str): Path to skycell WCS CSV file
        output_path (str): Output directory for mapping files
        buffer (int): Buffer size for PS1 skycells in pixels
        tess_buffer (int): Buffer size for TESS image in pixels
        n_threads (int): Number of threads for parallel processing
        overwrite (bool): Whether to overwrite existing files
        max_workers (int, optional): Maximum number of parallel workers for skycell processing.
                                   If None, uses min(32, len(skycells), cpu_count + 4)

    Returns:
        dict: Processing results and statistics
    """
    start_time = time.time()
    print("Starting optimized TESS image processing...")

    # Load data
    print("Loading TESS image and skycell database...")
    data_shape, tess_wcs, ra_center, dec_center, tess_header, sector, camera_id, ccd_id = load_tess_image(tess_file)

    skycell_wcs_df = pd.read_csv(skycell_wcs_csv)

    # Calculate skycell centers if not present
    if "RA" not in skycell_wcs_df.columns or "DEC" not in skycell_wcs_df.columns:
        print("Calculating skycell centers...")
        sc_center_ra_dec = calculate_radec_center(skycell_wcs_df.reset_index(drop=True))
        skycell_wcs_df["RA"] = sc_center_ra_dec[:, 0]
        skycell_wcs_df["DEC"] = sc_center_ra_dec[:, 1]

    # Create TESS pixel coordinates
    print("Creating TESS pixel coordinate arrays...")
    tpix_coord_input, ravelled_index = create_tess_pixel_coordinates(data_shape)

    # Find relevant skycells
    print("Finding relevant skycells using MOC filtering...")
    complete_wcs_skycells = find_relevant_skycells(skycell_wcs_df, tess_wcs, data_shape, tess_buffer)

    if len(complete_wcs_skycells) == 0:
        print("No relevant skycells found!")
        return {"status": "error", "message": "No relevant skycells found"}

    # print(f"Found {len(complete_wcs_skycells)} relevant skycells")

    # Create TESS-to-skycell mapping
    print("Creating optimized TESS-to-skycell mapping...")
    selected_skycells, tess_pix_skycell_mapping = process_tess_to_skycell_mapping(tess_wcs, data_shape, tpix_coord_input, complete_wcs_skycells, edge_buffer_large=edge_buffer_large, edge_buffer_small=edge_buffer_small, buffer=buffer, n_threads=n_threads)

    if np.any(tess_pix_skycell_mapping == -1):
        print("Warning: Some TESS pixels are not mapped to any skycell. This may affect the results.")

    # Save master mapping
    print("Saving master TESS-to-skycell mapping...")
    ffi_file_name = os.path.basename(tess_file)
    save_master_mapping(tess_pix_skycell_mapping, selected_skycells, ffi_file_name, tess_header, data_shape, output_path, sector, camera_id, ccd_id, overwrite)
    print(f"Processing time: {(time.time() - start_time):.2f} seconds")

    # Process each skycell
    print("Processing individual skycell mappings...")
    processed_skycells = 0
    skipped_skycells = 0

    # Determine number of workers for parallel processing
    if max_workers is None:
        import multiprocessing

        max_workers = min(32, len(selected_skycells), multiprocessing.cpu_count() + 4)

    # Prepare arguments for parallel processing
    task_args = []
    for _, skycell_row in selected_skycells.iterrows():
        args = (skycell_row, tess_wcs, tpix_coord_input, tess_header, output_path, sector, camera_id, ccd_id, pad_distance, edge_exclusion, overwrite, selected_skycells)
        task_args.append(args)

    # Process skycells in parallel with progress bar
    padding_results = {}  # Store padding info for each skycell

    if len(task_args) > 0:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_args = {executor.submit(process_single_skycell, args): args for args in task_args}

            # Process completed tasks with progress bar
            with tqdm(total=len(task_args), desc="Processing skycells") as pbar:
                for future in as_completed(future_to_args):
                    success, skycell_name, padding_info, error_message = future.result()

                    if success:
                        processed_skycells += 1
                        if padding_info:
                            padding_results[skycell_name] = padding_info
                    else:
                        skipped_skycells += 1
                        if error_message != "No TESS pixels in skycell":
                            print(f"Error processing skycell {skycell_name}: {error_message}")

                    pbar.update(1)

    # Update skycells with padding information collected from worker threads
    if padding_results:
        print("\nUpdating skycells with padding information...")
        selected_skycells = update_skycells_with_padding_info(selected_skycells, padding_results)

        # Save updated CSV with padding information
        save_updated_skycell_csv(selected_skycells, output_path, sector, camera_id, ccd_id)

    total_time = time.time() - start_time

    results = {
        "status": "success",
        "tess_file": tess_file,
        "total_skycells_found": len(complete_wcs_skycells),
        "selected_skycells": len(selected_skycells),
        "processed_skycells": processed_skycells,
        "skipped_skycells": skipped_skycells,
        "processing_time_seconds": total_time,
        "data_shape": data_shape,
        "ra_center": ra_center,
        "dec_center": dec_center,
    }

    print("\nProcessing complete!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Processed: {processed_skycells} skycells")
    print(f"Skipped: {skipped_skycells} skycells")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a TESS FITS file and generate PS1 skycell pixel mappings.")

    # Positional required argument
    parser.add_argument("tess_file", help="Path to the TESS FITS file (positional, required)")

    # Optional arguments (defaults chosen to match previous hard-coded values)
    parser.add_argument("--skycell_wcs_csv", default="./data/SkyCells/skycell_wcs.csv", help="Path to skycell WCS CSV file")
    parser.add_argument("--output_path", default="./data/skycell_pixel_mapping", help="Output directory for mapping files")
    parser.add_argument("--pad_distance", type=int, default=480, help="Pad distance in pixels for padding checks")
    parser.add_argument("--edge_exclusion", type=int, default=10, help="Edge exclusion in pixels for padding checks")
    parser.add_argument("--edge_buffer_large", type=int, default=410, help="Large edge buffer for WCS corner expansion")
    parser.add_argument("--edge_buffer_small", type=int, default=70, help="Small edge buffer for WCS corner expansion")
    parser.add_argument("--buffer", type=int, default=200, help="General buffer size in pixels")
    parser.add_argument("--tess_buffer", type=int, default=150, help="TESS footprint buffer in pixels used for MOC filtering")
    parser.add_argument("--n_threads", type=int, default=8, help="Number of threads to use in MOC filtering")
    parser.add_argument("--max_workers", type=int, default=None, help="Max workers for ProcessPoolExecutor (default: auto)")

    # Overwrite default preserved (default True). Provide flags to explicitly enable/disable.
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--overwrite", dest="overwrite", action="store_true", help="Overwrite existing output files (default)")
    group.add_argument("--no-overwrite", dest="overwrite", action="store_false", help="Do not overwrite existing output files")
    parser.set_defaults(overwrite=True)

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    # Call the processing function with parsed arguments
    results = process_tess_image_optimized(
        tess_file=args.tess_file,
        skycell_wcs_csv=args.skycell_wcs_csv,
        output_path=args.output_path,
        pad_distance=args.pad_distance,
        edge_exclusion=args.edge_exclusion,
        edge_buffer_large=args.edge_buffer_large,
        edge_buffer_small=args.edge_buffer_small,
        buffer=args.buffer,
        tess_buffer=args.tess_buffer,
        n_threads=args.n_threads,
        overwrite=args.overwrite,
        max_workers=args.max_workers,
    )

    print(f"Processing results: {results}")
