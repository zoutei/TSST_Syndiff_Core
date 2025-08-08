"""
PanCAKES v2.0 - Advanced Astronomical Image Processing Pipeline

This module provides a modern, high-performance implementation for processing
TESS (Transiting Exoplanet Survey Satellite) Full Frame Images and matching them
with PanSTARRS1 (PS1) SkyCell data using advanced computational techniques.

Author: Generated from optimization notebook analysis
Version: 2.0
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from mocpy import MOC
from numba import jit
import os
import time
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from astropy.wcs import FITSFixedWarning
from astropy.io.fits.verify import VerifyWarning

# Suppress common warnings
warnings.simplefilter("ignore", category=VerifyWarning)
warnings.filterwarnings("ignore", category=FITSFixedWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


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
        tuple: (data_shape, wcs, ra_center, dec_center, header, sector, ccd_id)
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

    sector = tess_file.split("/")[-1].split("-")[1][1:]
    camera = header["CAMERA"]
    ccd = header["CCD"]
    ccd_id = int(int(ccd) + 4 * (int(camera) - 1))

    hdul.close()
    return data_shape, wcs, ra_center, dec_center, header, sector, ccd_id


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


def process_tess_to_skycell_mapping(tess_wcs, data_shape, tpix_coord_input, complete_wcs_skycells, buffer=200, n_threads=8):
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

    enc_sc_vertices = calculate_radec_corners(complete_wcs_skycells, buffer)

    # Get TESS pixel RA/Dec coordinates
    _x_tess = tpix_coord_input[:, 1]
    _y_tess = tpix_coord_input[:, 0]
    _ra_tess, _dec_tess = tess_wcs.all_pix2world(_x_tess, _y_tess, 0)

    # Use MOC filtering for efficient polygon-point matching
    polygons = np.array(enc_sc_vertices)
    rust_result = MOC.filter_points_in_polygons(polygons=polygons, pix_ras=_ra_tess, pix_decs=_dec_tess, buffer=0.5, max_depth=21, n_threads=n_threads)

    # Create efficient pixel-to-skycell mapping
    rust_result_flat = np.concatenate([arr for arr in rust_result if len(arr) > 0])
    rust_result_lengths = np.array([len(arr) for arr in rust_result])

    tess_pix_skycell_id = create_closest_center_array_numba(rust_result_flat, rust_result_lengths, tess_proj_center_x, tess_proj_center_y, tpix_coord_input, data_shape[0] * data_shape[1])

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
    dict_for_header["SVERSION"] = 2.0
    dict_for_header["BSCALE"] = 1.0
    dict_for_header["BZERO"] = 32768.0

    if skycell_name:
        dict_for_header["SKYCELL"] = skycell_name

    return fits.Header(dict_for_header)


def save_skycell_mapping(mapping_array, skycell_name, tess_header, ps1_header, output_path, sector, ccd_id, overwrite=True):
    """
    Save PS1-to-TESS pixel mapping as compressed FITS file.

    Args:
        mapping_array (ndarray): 2D mapping array
        skycell_name (str): Skycell name
        tess_header (Header): TESS image header
        ps1_header (Header): PS1 skycell header
        output_path (str): Output directory path
        sector (str): TESS sector identifier
        ccd_id (int): Combined CCD identifier
        overwrite (bool): Whether to overwrite existing files
    """

    file_name = f"tess_s{sector}_{ccd_id}_{skycell_name}.fits"

    # Create headers
    base_header = create_fits_header(tess_header, skycell_name)

    new_fits_header = fits.Header()
    new_fits_header["SIMPLE"] = "T"
    new_fits_header += base_header
    new_fits_header_extended = new_fits_header + ps1_header

    # Process mapping array
    mapping_array[mapping_array == -1] = -1  # Ensure -1 for unmapped pixels

    # Create FITS file
    file_path = os.path.join(output_path, f"sector_{sector}", f"ccd_{ccd_id}", file_name)
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


def save_master_mapping(tess_pix_skycell_mapping, selected_skycells, tess_header, data_shape, output_path, sector, ccd_id, overwrite=True):
    """
    Save master TESS-to-skycell mapping file.

    Args:
        tess_pix_skycell_mapping (ndarray): TESS pixel to skycell mapping
        selected_skycells (DataFrame): Selected skycells information
        tess_header (Header): TESS image header
        data_shape (tuple): TESS image shape
        output_path (str): Output directory path
        sector (str): TESS sector identifier
        ccd_id (int): Combined CCD identifier
        overwrite (bool): Whether to overwrite existing files
    """
    # Create filename
    file_name = f"tess_s{sector}_{ccd_id}_master_pixels2skycells.fits"
    file_name_csv = f"tess_s{sector}_{ccd_id}_master_skycells_list.csv"

    file_path_csv = os.path.join(output_path, f"sector_{sector}", f"ccd_{ccd_id}", file_name_csv)
    os.makedirs(os.path.dirname(file_path_csv), exist_ok=True)
    selected_skycells.to_csv(file_path_csv)

    # Create header
    master_header = create_fits_header(tess_header)
    master_header["DATE-MOD"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")

    # Create FITS file
    file_path = os.path.join(output_path, f"sector_{sector}", f"ccd_{ccd_id}", file_name)
    primary_hdu = fits.PrimaryHDU(header=master_header)

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
                      tess_header, output_path, sector, ccd_id, overwrite)

    Returns:
        tuple: (success, skycell_name, error_message)
    """
    try:
        (skycell_row, tess_wcs, tpix_coord_input, tess_header, output_path, sector, ccd_id, overwrite) = args

        skycell_name = skycell_row["NAME"]
        tess_pix_in_skycell = skycell_row["pixel_indices"]

        if len(tess_pix_in_skycell) == 0:
            return (False, skycell_name, "No TESS pixels in skycell")

        # Get PS1 header and WCS information directly from skycell_row
        ps1_header, ps1_wcs, ps1_data_shape = get_ps1_wcs_information(skycell_row)

        # Process skycell pixel mapping
        mapping_array = process_skycell_pixel_mapping(tess_wcs, tpix_coord_input, ps1_wcs, ps1_data_shape, tess_pix_in_skycell)

        # Save mapping
        save_skycell_mapping(mapping_array, skycell_name, tess_header, ps1_header, output_path, sector, ccd_id, overwrite)

        return (True, skycell_name, None)

    except Exception as e:
        return (False, skycell_name, str(e))


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================


def process_tess_image_optimized(tess_file, skycell_wcs_csv, output_path, buffer=120, tess_buffer=150, n_threads=8, overwrite=True, max_workers=None):
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
    data_shape, tess_wcs, ra_center, dec_center, tess_header, sector, ccd_id = load_tess_image(tess_file)

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
    selected_skycells, tess_pix_skycell_mapping = process_tess_to_skycell_mapping(tess_wcs, data_shape, tpix_coord_input, complete_wcs_skycells, buffer, n_threads)

    if np.any(tess_pix_skycell_mapping == -1):
        print("Warning: Some TESS pixels are not mapped to any skycell. This may affect the results.")

    # Save master mapping
    print("Saving master TESS-to-skycell mapping...")
    save_master_mapping(tess_pix_skycell_mapping, selected_skycells, tess_header, data_shape, output_path, sector, ccd_id, overwrite)
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
        args = (skycell_row, tess_wcs, tpix_coord_input, tess_header, output_path, sector, ccd_id, overwrite)
        task_args.append(args)

    # Process skycells in parallel with progress bar
    if len(task_args) > 0:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_args = {executor.submit(process_single_skycell, args): args for args in task_args}

            # Process completed tasks with progress bar
            with tqdm(total=len(task_args), desc="Processing skycells") as pbar:
                for future in as_completed(future_to_args):
                    success, skycell_name, error_message = future.result()

                    if success:
                        processed_skycells += 1
                    else:
                        skipped_skycells += 1
                        if error_message != "No TESS pixels in skycell":
                            print(f"Error processing skycell {skycell_name}: {error_message}")

                    pbar.update(1)

    total_time = time.time() - start_time

    results = {"status": "success", "tess_file": tess_file, "total_skycells_found": len(complete_wcs_skycells), "selected_skycells": len(selected_skycells), "processed_skycells": processed_skycells, "skipped_skycells": skipped_skycells, "processing_time_seconds": total_time, "data_shape": data_shape, "ra_center": ra_center, "dec_center": dec_center}

    print("\nProcessing complete!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Processed: {processed_skycells} skycells")
    print(f"Skipped: {skipped_skycells} skycells")

    return results


if __name__ == "__main__":
    # Example usage
    tess_file = "./development/data/tess/tess2024066160824-s0076-2-4-0271-s_ffic.fits"
    # tess_file = "./development/data/tess/tess2022057231853-s0049-3-2-0221-s_ffic.fits"
    # tess_file = "./development/data/tess/tess2018207195942-s0001-4-1-0120-s_ffic.fits"
    skycell_wcs_csv = "./development/pancakes_opt/data/SkyCells/skycell_wcs.csv"
    output_path = "./development/data/mapping_output"

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Process TESS image
    results = process_tess_image_optimized(tess_file=tess_file, skycell_wcs_csv=skycell_wcs_csv, output_path=output_path, buffer=120, tess_buffer=150, n_threads=8, overwrite=True, max_workers=8)
 
    print(f"Processing results: {results}")
