#!/usr/bin/env python3
"""
Multi-Offset Downsampling Script

This script generates multiple downsampled images, each with a different
pixel offset. It handles mask bits and produces FITS output with proper headers.

Updated to use Zarr data from the convolved_results directory structure:
- data/convolved_results/sector_{sector:04d}/camera_{camera}/ccd_{ccd}/convolved_images.zarr
- data/convolved_results/sector_{sector:04d}/camera_{camera}/ccd_{ccd}/cell_metadata.json

The script loads PS1 convolved image data from Zarr stores instead of individual
FITS files, providing better performance and organization.
"""

import json
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from astropy.io import fits
from astropy.wcs import WCS
from joblib import Parallel, delayed
from tqdm import tqdm

# Import from existing script
from compute_ps1_skycell_shifts import RELEVANT_WCS_KEYS, build_ps1_wcs, compute_ps1_shift_for_skycell, load_tess_wcs


def load_zarr_metadata(sector: int, camera: int, ccd: int, convolved_data_path: Path) -> tuple[dict, Path]:
    """
    Load Zarr metadata once to avoid repeated file access.

    Returns:
        Tuple of (metadata_dict, zarr_path)
    """
    zarr_path = convolved_data_path / f"sector_{sector:04d}" / f"camera_{camera}" / f"ccd_{ccd}" / "convolved_images.zarr"
    metadata_path = convolved_data_path / f"sector_{sector:04d}" / f"camera_{camera}" / f"ccd_{ccd}" / "cell_metadata.json"

    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Load metadata once
    with open(metadata_path) as f:
        metadata = json.load(f)

    return metadata, zarr_path


def load_zarr_data_for_skycell(skycell_name: str, metadata: dict, zarr_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load PS1 convolved image and mask data from Zarr store for a specific skycell.

    Args:
        skycell_name: Name of the skycell (e.g., "rings.v3.skycell.1234.567")
        metadata: Pre-loaded metadata dictionary
        zarr_path: Path to the Zarr store

    Returns:
        Tuple of (image_data, mask_data) as numpy arrays
    """
    # Find the index for this skycell from pre-loaded metadata
    cell_index = None
    for idx, cell_info in enumerate(metadata["cells"]):
        if cell_info["name"] == skycell_name:
            cell_index = idx
            break

    if cell_index is None:
        raise ValueError(f"Skycell {skycell_name} not found in metadata")

    # Load the Zarr store (this is cached by Zarr internally)
    zarr_store = zarr.open(str(zarr_path), mode="r")

    # Load the convolved image data for this cell
    # Assuming the structure has separate arrays for images and masks
    if "convolved_images" in zarr_store:
        image_data = zarr_store["convolved_images"][cell_index]
    elif hasattr(zarr_store, "__len__") and cell_index < len(zarr_store):
        # If it's a flat array, just use the cell_index
        image_data = zarr_store[cell_index]
    else:
        # Try to find arrays by iterating through keys
        available_keys = list(zarr_store.keys()) if hasattr(zarr_store, "keys") else []
        raise ValueError(f"Cannot find image data for cell {cell_index}. Available keys: {available_keys}")

    # Load mask data (assuming similar structure)
    if "convolved_masks" in zarr_store:
        mask_data = zarr_store["convolved_masks"][cell_index]
    elif "masks" in zarr_store:
        mask_data = zarr_store["masks"][cell_index]
    else:
        # If no separate mask array, create a dummy mask of zeros
        mask_data = np.zeros_like(image_data, dtype=np.uint32)

    return image_data.astype(np.float32), mask_data.astype(np.uint32)


def precompute_shifts_for_offsets(tess_wcs: WCS, skycell_df: pd.DataFrame, offsets: np.ndarray) -> dict[tuple[float, float], pd.DataFrame]:
    """
    Precompute all PS1 shifts for each offset pair and skycell

    Returns:
        Dictionary mapping (dx, dy) to DataFrame with NAME, shift_x, shift_y
    """
    shift_results = {}

    for dx, dy in tqdm(offsets, desc="Computing shifts"):
        shift_x_list = []
        shift_y_list = []

        for _, row in skycell_df.iterrows():
            ps1_wcs, _ = build_ps1_wcs(row)
            sx, sy = compute_ps1_shift_for_skycell(
                tess_wcs,
                dx,
                dy,
                float(row["RA"]),
                float(row["DEC"]),
                ps1_wcs,
            )
            # Round to nearest integer (no interpolation)
            sx_int = int(round(sx))
            sy_int = int(round(sy))
            shift_x_list.append(sx_int)
            shift_y_list.append(sy_int)

        shift_df = pd.DataFrame(
            {
                "NAME": skycell_df["NAME"],
                "shift_x": shift_x_list,
                "shift_y": shift_y_list,
            }
        )

        shift_results[(dx, dy)] = shift_df

    return shift_results


def process_skycell_batch(batch_idx: int, reg_files: list[str], skycell_names: list[str], offsets: np.ndarray, shifts_dict: dict[tuple[float, float], pd.DataFrame], tess_shape: tuple[int, int], zarr_metadata: dict, zarr_path: Path, padding: int = 500, ignore_mask_bits: list[int] = []) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a batch of skycells using sparse arrays for memory efficiency

    Returns:
        Tuple of (indices, sums, counts, mask_counts) where:
        - indices: Array of TESS pixel indices (1D linearized from y,x)
        - sums: Array of shape (len(indices), num_offsets) with sum values
        - counts: Array of shape (len(indices), num_offsets) with count values
        - mask_counts: Array of shape (len(indices), num_offsets) with mask count values
    """
    t_y, t_x = tess_shape
    num_offsets = len(offsets)

    # Lists to collect values (will convert to arrays later)
    all_indices = []
    all_sums = []
    all_counts = []
    all_mask_counts = []

    # Create mask for ignoring specific bits
    ignore_mask = 0
    for bit in ignore_mask_bits:
        ignore_mask |= 1 << bit

    # Process each skycell in the batch
    for sc_idx, (reg_file, skycell_name) in enumerate(zip(reg_files, skycell_names)):
        try:
            # Load registration mapping
            with fits.open(reg_file) as hdul:
                ps1_assignment_x = hdul[1].data.astype(int)
                ps1_assignment_x[ps1_assignment_x == 65535] = -1
                ps1_assignment_y = hdul[2].data.astype(int)
                ps1_assignment_y[ps1_assignment_y == 65535] = 0

            # Create the pixel mapping
            ps1_assignment = ps1_assignment_x + ps1_assignment_y * t_x

            # Prepare for binning
            pind = ps1_assignment.ravel()
            sort_ind = np.argsort(pind)

            # Get valid TESS pixels
            tess_pixels = np.unique(pind[np.isfinite(pind)]).astype(int)
            tess_pixels = tess_pixels[tess_pixels >= 0]

            if len(tess_pixels) == 0:
                continue

            # Calculate breaks for binning
            breaks = np.where(np.diff(pind[sort_ind]) > 0)[0] + 1
            breaks = np.append(breaks, len(sort_ind))

            # Try to load PS1 data from Zarr store
            try:
                # Load PS1 data and mask from Zarr
                ps1_data, ps1_mask = load_zarr_data_for_skycell(skycell_name, zarr_metadata, zarr_path)

                # Extract the core region without padding (if padding exists in the data)
                if ps1_data.shape[0] > 2 * padding and ps1_data.shape[1] > 2 * padding:
                    ps1_base = ps1_data[padding:-padding, padding:-padding]
                    ps1_mask_base = ps1_mask[padding:-padding, padding:-padding]
                else:
                    # If no padding or insufficient padding, use the full data
                    ps1_base = ps1_data
                    ps1_mask_base = ps1_mask

                # Initialize arrays for each skycell's results
                pixel_sums = np.zeros((len(tess_pixels), num_offsets), dtype=np.float32)
                pixel_counts = np.zeros((len(tess_pixels), num_offsets), dtype=np.int32)
                pixel_mask_counts = np.zeros((len(tess_pixels), num_offsets), dtype=np.int32)

                # Process each offset
                for offset_idx, (dx, dy) in enumerate(offsets):
                    # Get shifts from precomputed values
                    shift_df = shifts_dict[(dx, dy)]
                    row_idx = shift_df.index[shift_df["NAME"] == skycell_name].tolist()

                    if not row_idx:
                        continue

                    sx = shift_df.loc[row_idx[0], "shift_x"]
                    sy = shift_df.loc[row_idx[0], "shift_y"]

                    # Apply the shift (integer pixel shifts only - no interpolation)
                    ps1_shifted = np.roll(ps1_base, (sy, sx), axis=(0, 1))
                    ps1_mask_shifted = np.roll(ps1_mask_base, (sy, sx), axis=(0, 1))

                    # Sort the shifted data
                    ps1_rav = ps1_shifted.ravel()[sort_ind]
                    ps1_mask_rav = ps1_mask_shifted.ravel()[sort_ind]

                    # Compute sums for each TESS pixel
                    sums = np.zeros(len(breaks) - 1, dtype=np.float32)
                    counts = np.zeros(len(breaks) - 1, dtype=np.int32)
                    mask_counts = np.zeros(len(breaks) - 1, dtype=np.int32)

                    for i in range(len(breaks) - 1):
                        slice_data = ps1_rav[breaks[i] : breaks[i + 1]]
                        slice_mask = ps1_mask_rav[breaks[i] : breaks[i + 1]]

                        # Count pixels that should be ignored based on mask bits
                        ignored_pixels = (slice_mask & ignore_mask) > 0

                        # Count all pixels for denominator
                        counts[i] = len(slice_data)

                        # Sum only non-masked pixels
                        sums[i] = np.sum(slice_data[~ignored_pixels])

                        # Count masked pixels for reference
                        mask_counts[i] = np.sum(slice_mask != 0)

                    # Store the results for this offset
                    pixel_sums[:, offset_idx] = sums
                    pixel_counts[:, offset_idx] = counts
                    pixel_mask_counts[:, offset_idx] = mask_counts

                # Add all valid pixels from this skycell to our results
                # Filter out pixels outside the image bounds
                valid_mask = (tess_pixels // t_x < t_y) & (tess_pixels % t_x < t_x)

                if np.any(valid_mask):
                    all_indices.append(tess_pixels[valid_mask])
                    all_sums.append(pixel_sums[valid_mask])
                    all_counts.append(pixel_counts[valid_mask])
                    all_mask_counts.append(pixel_mask_counts[valid_mask])

            except Exception as e:
                print(f"Error processing PS1 data for skycell {skycell_name}: {e}")
                continue

        except Exception as e:
            print(f"Error processing registration for skycell {skycell_name}: {e}")

    print(f"Completed batch {batch_idx + 1}")

    # Convert lists to arrays
    if all_indices:
        indices = np.concatenate(all_indices)
        sums = np.vstack(all_sums)
        counts = np.vstack(all_counts)
        mask_counts = np.vstack(all_mask_counts)
    else:
        # Return empty arrays if no data
        indices = np.array([], dtype=int)
        sums = np.zeros((0, num_offsets), dtype=np.float32)
        counts = np.zeros((0, num_offsets), dtype=np.int32)
        mask_counts = np.zeros((0, num_offsets), dtype=np.int32)

    return indices, sums, counts, mask_counts


def create_syndiff_header(tess_header):
    """
    Create a header for the syndiff output based on the TESS header.
    """
    # First, copy specific keywords
    keys_to_copy = ["TELESCOP", "INSTRUME", "CAMERA", "CCD"]
    syndiff_header = fits.Header()

    for key in keys_to_copy:
        if key in tess_header:
            syndiff_header.set(key, tess_header[key], tess_header.comments.get(key, ""))

    # Set PS1 date information
    syndiff_header.set("MJD-OBS", "55197.00000", "TSTART of PS1")
    syndiff_header.set("DATE-OBS", "2010-01-01T00:00:00.000", "TSTART of PS1")
    syndiff_header.set("DATE-END", "2015-01-01T00:00:00.000", "TSTOP of PS1")

    # Copy WCS and quality information
    keys_to_copy = ["RADESYS", "EQUINOX", "WCSAXES", "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2", "DQUALITY", "IMAGTYPE"]

    for key in tess_header:
        if key.startswith(("A_", "B_", "AP_", "BP_", "RA_", "DEC_", "ROLL_")) or key in keys_to_copy:
            syndiff_header.set(key, tess_header[key], tess_header.comments.get(key, ""))

    # Add syndiff tag
    syndiff_header.set("SYNDIFF", True, "Syndiff template")

    return syndiff_header


def save_fits_outputs(output_dir: Path, results: np.ndarray, offsets: np.ndarray, tess_header: fits.Header, save_extensions: bool = True):
    """
    Save the results as FITS files.

    Args:
        output_dir: Directory to save outputs
        results: Array of shape (num_offsets, 3, ny, nx) with:
            [0] = sum of PS1 pixel values
            [1] = count of PS1 pixels
            [2] = count of masked PS1 pixels
        offsets: Array of (dx, dy) pairs
        tess_header: Header from TESS file to use as a base
        save_extensions: Whether to save data, count and mask as HDU extensions
    """
    # Create syndiff header based on TESS header
    syndiff_header = create_syndiff_header(tess_header)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # # Save a CSV with the offsets
    # offset_df = pd.DataFrame(offsets, columns=['dx', 'dy'])
    # offset_df.to_csv(output_dir / "offsets.csv", index=False)

    # Save each offset result as a FITS file
    for idx, (dx, dy) in enumerate(offsets):
        # Calculate final image (handle division by zero)
        mask = results[idx, 1] > 0  # Pixels with counts
        final_image = np.zeros_like(results[idx, 0])
        final_image[mask] = results[idx, 0][mask] / results[idx, 1][mask]

        # Update header with offset information
        offset_header = syndiff_header.copy()
        offset_header["DX_SHIFT"] = (dx, "TESS pixel x shift")
        offset_header["DY_SHIFT"] = (dy, "TESS pixel y shift")

        # File with just the data
        primary_hdu = fits.PrimaryHDU(data=final_image, header=offset_header)
        hdu_list = fits.HDUList([primary_hdu])
        output_filename = output_dir / f"_dx{dx:.3f}_dy{dy:.3f}.fits"
        hdu_list.writeto(output_filename, overwrite=True)

        if save_extensions:
            # File with data, count, and mask as extensions
            primary_hdu = fits.PrimaryHDU(header=offset_header)
            hdu1 = fits.ImageHDU(data=final_image, header=offset_header, name="FLUX")
            hdu2 = fits.ImageHDU(data=results[idx, 1].astype(np.int32), header=offset_header, name="COUNT")
            hdu3 = fits.ImageHDU(data=results[idx, 2].astype(np.int32), header=offset_header, name="MASK")

            hdu_list = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3])
            output_filename = output_dir / f"downsampled_dx{dx:.3f}_dy{dy:.3f}_extended.fits"
            hdu_list.writeto(output_filename, overwrite=True)


def main():
    # Set sector, camera, and ccd
    sector = 20
    camera = 3
    ccd = 3

    # Generate paths based on parameters
    TESS_FITS_PATH = Path(f"/home/kshukawa/syndiff/data/tess/{sector}_{camera}_{ccd}/tess2020019135923-s00{sector}-{camera}-{ccd}-0165-s_ffic.fits")
    SKYCELL_CSV_PATH = Path(f"/home/kshukawa/syndiff/data/skycell_pixel_mapping/sector_00{sector}/camera_{camera}/ccd_{ccd}/tess_s00{sector}_{camera}_{ccd}_master_skycells_list.csv")
    CONVOLVED_DATA_PATH = Path("/home/kshukawa/syndiff/data/convolved_results/")
    REG_FILES_PATTERN = f"/home/kshukawa/syndiff/data/skycell_pixel_mapping/sector_00{sector}/camera_{camera}/ccd_{ccd}/*.fits.gz"
    OUTPUT_DIR = Path(f"/home/kshukawa/syndiff/data/shifted_downsamples/sector{sector}_camera{camera}_ccd{ccd}")

    # Processing parameters - adjusted for better memory efficiency
    N_JOBS = 16  # Reduced number of parallel jobs to lower memory pressure
    SKYCELLS_PER_BATCH = 20  # Adjusted for memory usage
    padding = 500

    """Main function to run the multi-offset downsampling process."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Define offsets to process (could be loaded from a file)
    offsets = np.array(
        [
            [0.1, 0.1],  # dx, dy
            [0.2, 0.0],
            [0.0, 0.2],
            [-0.1, 0.1],
            [-0.1, -0.1],
        ]
    )

    # Set mask bits to ignore (0-indexed)
    # For example, to ignore bits 0, 2, and 5:
    ignore_mask_bits = [5]

    # Load TESS data and WCS
    print("Loading TESS data and WCS...")
    start_time = time.time()
    tess_wcs, tess_dims = load_tess_wcs(TESS_FITS_PATH)
    with fits.open(TESS_FITS_PATH) as hdul:
        # Find HDU with data
        hdu_idx = 1 if len(hdul) > 1 and getattr(hdul[1], "data", None) is not None else 0
        tess_data = hdul[hdu_idx].data.astype(np.float32)
        tess_header = hdul[hdu_idx].header

    # Load skycell info
    print("Loading skycell info...")
    usecols = ["NAME", "RA", "DEC"] + RELEVANT_WCS_KEYS
    skycell_df = pd.read_csv(SKYCELL_CSV_PATH, usecols=usecols)

    # Load Zarr metadata once for efficient access
    print("Loading Zarr metadata...")
    zarr_metadata, zarr_path = load_zarr_metadata(sector, camera, ccd, CONVOLVED_DATA_PATH)
    print(f"Found {len(zarr_metadata['cells'])} cells in Zarr store")

    # Precompute shifts for all offsets
    print("Precomputing shifts for all offsets...")
    shifts_dict = precompute_shifts_for_offsets(tess_wcs, skycell_df, offsets)

    # Get registration files
    print("Getting registration files...")
    reg_files = sorted(glob(REG_FILES_PATTERN))
    skycell_names = [Path(f).stem.split("_")[0] for f in reg_files]  # Extract skycell names

    # Split into batches
    num_batches = (len(reg_files) + SKYCELLS_PER_BATCH - 1) // SKYCELLS_PER_BATCH
    print(f"Processing {len(reg_files)} skycells in {num_batches} batches...")

    reg_batches = np.array_split(reg_files, num_batches)
    name_batches = np.array_split(skycell_names, num_batches)

    # Process batches in parallel
    results = Parallel(n_jobs=N_JOBS)(delayed(process_skycell_batch)(i, reg_batch, name_batch, offsets, shifts_dict, tess_data.shape, zarr_metadata, zarr_path, padding=padding, ignore_mask_bits=ignore_mask_bits) for i, (reg_batch, name_batch) in enumerate(zip(reg_batches, name_batches)))

    # Combine results using the sparse array approach
    print("Combining results...")
    all_indices = []
    all_sums = []
    all_counts = []
    all_mask_counts = []

    for indices, sums, counts, mask_counts in results:
        if len(indices) > 0:
            all_indices.append(indices)
            all_sums.append(sums)
            all_counts.append(counts)
            all_mask_counts.append(mask_counts)

    # Concatenate all results
    if all_indices:
        combined_indices = np.concatenate(all_indices)
        combined_sums = np.vstack(all_sums)
        combined_counts = np.vstack(all_counts)
        combined_mask_counts = np.vstack(all_mask_counts)

        # Handle duplicate pixels (from different skycells)
        if len(combined_indices) > len(np.unique(combined_indices)):
            # Find unique indices and their positions
            unique_indices, inverse_indices = np.unique(combined_indices, return_inverse=True)

            # Initialize arrays for the consolidated results
            unique_sums = np.zeros((len(unique_indices), len(offsets)), dtype=np.float32)
            unique_counts = np.zeros((len(unique_indices), len(offsets)), dtype=np.int32)
            unique_mask_counts = np.zeros((len(unique_indices), len(offsets)), dtype=np.int32)

            # Use np.add.at for efficient aggregation by index
            np.add.at(unique_sums, inverse_indices, combined_sums)
            np.add.at(unique_counts, inverse_indices, combined_counts)
            np.add.at(unique_mask_counts, inverse_indices, combined_mask_counts)

            # Replace with deduplicated arrays
            combined_indices = unique_indices
            combined_sums = unique_sums
            combined_counts = unique_counts
            combined_mask_counts = unique_mask_counts

        # Now convert from sparse representation to full array
        combined_results = np.zeros((len(offsets), 3, *tess_data.shape), dtype=np.float32)

        for i, idx in enumerate(combined_indices):
            y = idx // tess_data.shape[1]
            x = idx % tess_data.shape[1]

            if 0 <= y < tess_data.shape[0] and 0 <= x < tess_data.shape[1]:
                for offset_idx in range(len(offsets)):
                    combined_results[offset_idx, 0, y, x] = combined_sums[i, offset_idx]
                    combined_results[offset_idx, 1, y, x] = combined_counts[i, offset_idx]
                    combined_results[offset_idx, 2, y, x] = combined_mask_counts[i, offset_idx]
    else:
        # Create empty results if no data
        combined_results = np.zeros((len(offsets), 3, *tess_data.shape), dtype=np.float32)

    # Save outputs as FITS files
    print("Saving outputs...")
    save_fits_outputs(OUTPUT_DIR, combined_results, offsets, tess_header, save_extensions=True)

    # Record processing time
    total_time = time.time() - start_time
    print(f"Done! Total processing time: {total_time / 60:.2f} minutes")

    # Print summary information
    print(f"Processing completed at: {time.ctime()}")
    print(f"Total time: {total_time / 60:.2f} minutes")
    print(f"Processed {len(reg_files)} skycells in {num_batches} batches")
    print(f"Generated {len(offsets)} shifted images")
    print(f"Ignored mask bits: {ignore_mask_bits}")
    print("Shifts processed:")
    for dx, dy in offsets:
        print(f"  dx={dx:.3f}, dy={dy:.3f}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
