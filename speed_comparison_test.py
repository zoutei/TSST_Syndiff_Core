#!/usr/bin/env python3
"""
Speed comparison tests for zarr vs FITS data loading and processing.

This script compares the performance of:
1. Loading data from zarr files and combining bands (10 iterations)
2. Loading the same data from FITS files and combining bands (10 iterations)
3. Saving results to both zarr and FITS formats

Author: Test script for TSST_Syndiff_Core
"""

import logging
import os
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def time_zarr_loading_and_combining(n_iterations: int = 10) -> tuple[float, dict]:
    """Time zarr loading and band combination for multiple iterations.

    Args:
        n_iterations: Number of iterations to run

    Returns:
        Tuple of (average_time, sample_results)
    """
    logger.info(f"Testing zarr loading and combining for {n_iterations} skycells...")

    try:
        from band_utils import process_skycell_bands
        from zarr_utils import get_available_projections, get_projection_skycells, load_skycell_bands, load_skycell_masks

        zarr_path = "data/ps1_skycells_zarr/ps1_skycells.zarr"

        if not os.path.exists(zarr_path):
            logger.error(f"Zarr file not found: {zarr_path}")
            return 0.0, {}

        projections = get_available_projections(zarr_path)
        if not projections:
            logger.error("No projections found in zarr")
            return 0.0, {}

        # Gather up to n_iterations skycells across projections
        skycell_list = []
        for proj in projections:
            skycells = get_projection_skycells(zarr_path, proj)
            for skycell_full in skycells:
                # Convert full format "skycell.2556.080" to short format "080"
                if skycell_full.startswith("skycell."):
                    skycell_short = skycell_full.split(".")[-1]  # Get the last part
                else:
                    skycell_short = skycell_full
                skycell_list.append((proj, skycell_short))
                if len(skycell_list) >= n_iterations:
                    break
            if len(skycell_list) >= n_iterations:
                break

        if not skycell_list:
            logger.error("No skycells found in any projection")
            return 0.0, {}

        times = []
        sample_results = None
        all_results = []

        for i, (proj, skycell) in enumerate(skycell_list):
            start_time = time.time()
            bands_data = load_skycell_bands(zarr_path, proj, skycell)
            masks_data = load_skycell_masks(zarr_path, proj, skycell)
            combined_image, combined_mask = process_skycell_bands(bands_data, masks_data)
            end_time = time.time()
            iteration_time = end_time - start_time
            times.append(iteration_time)
            result = {"combined_image": combined_image, "combined_mask": combined_mask, "projection": proj, "skycell": skycell, "bands_loaded": list(bands_data.keys()), "image_shape": combined_image.shape, "mask_shape": combined_mask.shape}
            all_results.append(result)
            if i == 0:
                sample_results = result
            logger.debug(f"Skycell {i + 1}: {proj}/{skycell} - {iteration_time:.4f}s")

        avg_time = np.mean(times)
        std_time = np.std(times)
        logger.info(f"Zarr loading average: {avg_time:.4f}s ± {std_time:.4f}s for {len(times)} skycells")

        return avg_time, {"sample": sample_results, "all": all_results}

    except Exception as e:
        logger.error(f"Zarr loading test failed: {e}")
        return 0.0, {}


def time_fits_loading_and_combining(projection: str, skycell: str, n_iterations: int = 10) -> tuple[float, dict]:
    """Time FITS loading and band combination for multiple iterations.

    Args:
        projection: PS1 projection ID (for finding corresponding FITS files)
        skycell: Skycell ID
        n_iterations: Number of iterations to run

    Returns:
        Tuple of (average_time, sample_results)
    """
    logger.info(f"Testing FITS loading and combining for {n_iterations} skycells...")

    try:
        from band_utils import process_skycell_bands
        from zarr_utils import get_available_projections, get_projection_skycells

        fits_base_path = Path("data/ps1_skycells")
        bands = ["r", "i", "z", "y"]

        # Gather up to n_iterations skycells using zarr_utils for consistency
        zarr_path = "data/ps1_skycells_zarr/ps1_skycells.zarr"
        projections = get_available_projections(zarr_path)
        skycell_list = []
        for proj in projections:
            skycells = get_projection_skycells(zarr_path, proj)
            for skycell in skycells:
                skycell_list.append((proj, skycell))
                if len(skycell_list) >= n_iterations:
                    break
            if len(skycell_list) >= n_iterations:
                break

        if not skycell_list:
            logger.error("No skycells found for FITS test")
            return 0.0, {}

        times = []
        sample_results = None
        all_results = []

        for i, (target_proj, target_sky) in enumerate(skycell_list):
            found_fits_dir = None
            # Find the FITS directory for this skycell
            for proj_dir in fits_base_path.iterdir():
                if proj_dir.is_dir() and proj_dir.name == target_proj:
                    for sky_dir in proj_dir.iterdir():
                        if sky_dir.is_dir() and sky_dir.name == target_sky:
                            sample_file = sky_dir / f"rings.v3.skycell.{proj_dir.name}.{sky_dir.name}.stk.r.unconv.fits"
                            if sample_file.exists():
                                found_fits_dir = sky_dir
                                break
                    if found_fits_dir:
                        break
            if not found_fits_dir:
                logger.warning(f"No FITS directory for {target_proj}/{target_sky}, skipping...")
                continue

            fits_files = {}
            mask_files = {}
            proj_id = found_fits_dir.parent.name
            sky_id = found_fits_dir.name
            for band in bands:
                fits_file = found_fits_dir / f"rings.v3.skycell.{proj_id}.{sky_id}.stk.{band}.unconv.fits"
                mask_file = found_fits_dir / f"rings.v3.skycell.{proj_id}.{sky_id}.stk.{band}.unconv.mask.fits"
                if fits_file.exists():
                    fits_files[band] = fits_file
                if mask_file.exists():
                    mask_files[band] = mask_file
            if not fits_files:
                logger.warning(f"No FITS files found in {found_fits_dir}, skipping...")
                continue

            start_time = time.time()
            bands_data = {}
            masks_data = {}
            for band, fits_file in fits_files.items():
                with fits.open(fits_file) as hdul:
                    bands_data[band] = hdul[0].data.astype(np.float32)
                if band in mask_files:
                    with fits.open(mask_files[band]) as hdul:
                        masks_data[band] = hdul[0].data.astype(np.float32)
            combined_image, combined_mask = process_skycell_bands(bands_data, masks_data)
            end_time = time.time()
            iteration_time = end_time - start_time
            times.append(iteration_time)
            result = {"combined_image": combined_image, "combined_mask": combined_mask, "projection": proj_id, "skycell": sky_id, "bands_loaded": list(bands_data.keys()), "image_shape": combined_image.shape, "mask_shape": combined_mask.shape, "fits_dir": str(found_fits_dir)}
            all_results.append(result)
            if i == 0:
                sample_results = result
            logger.debug(f"FITS Skycell {i + 1}: {proj_id}/{sky_id} - {iteration_time:.4f}s")

        if not times:
            logger.error("No FITS skycells processed.")
            return 0.0, {}

        avg_time = np.mean(times)
        std_time = np.std(times)
        logger.info(f"FITS loading average: {avg_time:.4f}s ± {std_time:.4f}s for {len(times)} skycells")

        return avg_time, {"sample": sample_results, "all": all_results}

    except Exception as e:
        logger.error(f"FITS loading test failed: {e}")
        return 0.0, {}


def time_save_zarr(combined_image: np.ndarray, combined_mask: np.ndarray, n_iterations: int = 10) -> float:
    """Time zarr save operations for multiple iterations.

    Args:
        combined_image: Combined image data
        combined_mask: Combined mask data
        n_iterations: Number of iterations to run

    Returns:
        Average time taken to save
    """
    logger.info(f"Testing zarr save performance for {n_iterations} iterations...")

    try:
        import zarr

        times = []

        for i in range(n_iterations):
            output_path = f"data/speed_test_output_{i}.zarr"
            start_time = time.time()

            # Create zarr store
            store = zarr.open(output_path, mode="w")

            # Save data using create_array
            store.create_array("combined_image", data=combined_image.astype(np.float32))
            store.create_array("combined_mask", data=combined_mask.astype(np.float32))

            # Add metadata
            store.attrs["created_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            store.attrs["data_shape"] = combined_image.shape
            store.attrs["data_type"] = "speed_test_results"
            store.attrs["iteration"] = i

            end_time = time.time()
            iteration_time = end_time - start_time
            times.append(iteration_time)

            logger.debug(f"Zarr save iteration {i + 1}: {iteration_time:.4f}s")

        avg_time = np.mean(times)
        std_time = np.std(times)
        logger.info(f"Zarr save average: {avg_time:.4f}s ± {std_time:.4f}s")
        return avg_time

    except Exception as e:
        logger.error(f"Failed to time zarr save: {e}")
        return 0.0


def time_save_fits(combined_image: np.ndarray, combined_mask: np.ndarray, n_iterations: int = 10) -> float:
    """Time FITS save operations for multiple iterations.

    Args:
        combined_image: Combined image data
        combined_mask: Combined mask data
        n_iterations: Number of iterations to run

    Returns:
        Average time taken to save
    """
    logger.info(f"Testing FITS save performance for {n_iterations} iterations...")

    try:
        times = []

        for i in range(n_iterations):
            output_path = f"data/speed_test_output_{i}.fits"
            start_time = time.time()

            # Create FITS HDU list
            primary_hdu = fits.PrimaryHDU(data=combined_image.astype(np.float32))
            mask_hdu = fits.ImageHDU(data=combined_mask.astype(np.float32), name="MASK")

            # Add metadata
            primary_hdu.header["CREATED"] = time.strftime("%Y-%m-%d %H:%M:%S")
            primary_hdu.header["DATATYPE"] = "SPEED_TEST"
            primary_hdu.header["ITER"] = i
            primary_hdu.header["COMMENT"] = "Speed test results - combined image"
            mask_hdu.header["COMMENT"] = "Speed test results - combined mask"

            hdul = fits.HDUList([primary_hdu, mask_hdu])
            hdul.writeto(output_path, overwrite=True)

            end_time = time.time()
            iteration_time = end_time - start_time
            times.append(iteration_time)

            logger.debug(f"FITS save iteration {i + 1}: {iteration_time:.4f}s")

        avg_time = np.mean(times)
        std_time = np.std(times)
        logger.info(f"FITS save average: {avg_time:.4f}s ± {std_time:.4f}s")
        return avg_time

    except Exception as e:
        logger.error(f"Failed to time FITS save: {e}")
        return 0.0


def save_results_zarr(combined_image: np.ndarray, combined_mask: np.ndarray, output_path: str = "data/speed_test_output.zarr") -> float:
    """Save results to zarr format and measure time.

    Args:
        combined_image: Combined image data
        combined_mask: Combined mask data
        output_path: Output zarr path

    Returns:
        Time taken to save
    """
    logger.info("Saving results to zarr...")

    try:
        import zarr

        start_time = time.time()

        # Create zarr store
        store = zarr.open(output_path, mode="w")

        # Save data using create_array
        store.create_array("combined_image", data=combined_image.astype(np.float32))
        store.create_array("combined_mask", data=combined_mask.astype(np.float32))

        # Add metadata
        store.attrs["created_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        store.attrs["data_shape"] = combined_image.shape
        store.attrs["data_type"] = "speed_test_results"

        end_time = time.time()
        save_time = end_time - start_time

        logger.info(f"Zarr save time: {save_time:.4f}s")
        return save_time

    except Exception as e:
        logger.error(f"Failed to save zarr: {e}")
        return 0.0


def save_results_fits(combined_image: np.ndarray, combined_mask: np.ndarray, output_path: str = "data/speed_test_output.fits") -> float:
    """Save results to FITS format and measure time.

    Args:
        combined_image: Combined image data
        combined_mask: Combined mask data
        output_path: Output FITS path

    Returns:
        Time taken to save
    """
    logger.info("Saving results to FITS...")

    try:
        start_time = time.time()

        # Create FITS HDU list
        primary_hdu = fits.PrimaryHDU(data=combined_image.astype(np.float32))
        mask_hdu = fits.ImageHDU(data=combined_mask.astype(np.float32), name="MASK")

        # Add metadata
        primary_hdu.header["CREATED"] = time.strftime("%Y-%m-%d %H:%M:%S")
        primary_hdu.header["DATATYPE"] = "SPEED_TEST"
        primary_hdu.header["COMMENT"] = "Speed test results - combined image"
        mask_hdu.header["COMMENT"] = "Speed test results - combined mask"

        hdul = fits.HDUList([primary_hdu, mask_hdu])
        hdul.writeto(output_path, overwrite=True)

        end_time = time.time()
        save_time = end_time - start_time

        logger.info(f"FITS save time: {save_time:.4f}s")
        return save_time

    except Exception as e:
        logger.error(f"Failed to save FITS: {e}")
        return 0.0


def compare_data_consistency(zarr_results: dict, pipeline_results: dict) -> bool:
    """Compare results from zarr and pipeline loading to ensure consistency.

    Args:
        zarr_results: Results from zarr loading
        pipeline_results: Results from pipeline loading

    Returns:
        True if data is reasonably consistent
    """
    logger.info("Comparing data consistency between zarr and pipeline...")

    try:
        zarr_image = zarr_results["combined_image"]
        zarr_mask = zarr_results["combined_mask"]

        # Pipeline results have different structure: {"image": ..., "mask": ...}
        pipeline_image = pipeline_results["image"]
        pipeline_mask = pipeline_results["mask"]

        # Check shapes
        if zarr_image.shape != pipeline_image.shape:
            logger.warning(f"Image shape mismatch: zarr {zarr_image.shape} vs pipeline {pipeline_image.shape}")
            return False

        if zarr_mask.shape != pipeline_mask.shape:
            logger.warning(f"Mask shape mismatch: zarr {zarr_mask.shape} vs pipeline {pipeline_mask.shape}")
            return False

        # Check data ranges (allowing for some differences due to processing)
        zarr_valid = zarr_image[~np.isnan(zarr_image)]
        pipeline_valid = pipeline_image[~np.isnan(pipeline_image)]

        if len(zarr_valid) > 0 and len(pipeline_valid) > 0:
            zarr_range = [np.min(zarr_valid), np.max(zarr_valid)]
            pipeline_range = [np.min(pipeline_valid), np.max(pipeline_valid)]

            logger.info(f"Zarr image range: [{zarr_range[0]:.3f}, {zarr_range[1]:.3f}]")
            logger.info(f"Pipeline image range: [{pipeline_range[0]:.3f}, {pipeline_range[1]:.3f}]")

            # Check if ranges are reasonably similar (within order of magnitude)
            if abs(np.log10(abs(zarr_range[1]) + 1e-10) - np.log10(abs(pipeline_range[1]) + 1e-10)) > 2:
                logger.warning("Image ranges differ significantly")
                return False

        logger.info("Data consistency check passed")
        return True

    except Exception as e:
        logger.error(f"Data consistency check failed: {e}")
        return False


def main():
    """Run the speed comparison tests."""
    n_iterations = 10

    print("🚀 PS1 Data Loading Speed Comparison")
    print("=" * 50)

    # Gather skycells for all tests
    print("\n📊 Gathering skycells for testing...")
    from zarr_utils import get_available_projections, get_projection_skycells

    zarr_path = "data/ps1_skycells_zarr/ps1_skycells.zarr"
    projections = get_available_projections(zarr_path)
    skycell_list = []
    for proj in projections:
        skycells = get_projection_skycells(zarr_path, proj)
        for skycell_full in skycells:
            # Convert full format "skycell.2556.080" to short format "080"
            if skycell_full.startswith("skycell."):
                skycell_short = skycell_full.split(".")[-1]  # Get the last part
            else:
                skycell_short = skycell_full
            skycell_list.append((proj, skycell_short))
            if len(skycell_list) >= n_iterations:
                break
        if len(skycell_list) >= n_iterations:
            break

    if not skycell_list:
        print("❌ No skycells found for testing")
        return 1

    # Test Zarr loading on the gathered skycells
    print("\n📊 Testing Zarr Loading Performance on 10 skycells...")
    zarr_time, zarr_results = time_zarr_loading_and_combining(n_iterations=n_iterations)

    if not zarr_results:
        print("❌ Zarr test failed, cannot continue")
        return 1

    # Test FITS loading on the same skycells
    print("\n📁 Testing FITS Loading Performance on 10 skycells...")
    fits_time, fits_results = time_fits_loading_and_combining(zarr_results["sample"]["projection"], zarr_results["sample"]["skycell"], n_iterations=n_iterations)

    if not fits_results:
        print("❌ FITS test failed")
        return 1

    # Test load_and_combine pipeline on the same skycells
    print("\n🧵 Testing load_and_combine pipeline on 10 skycells...")
    pipeline_time, pipeline_results = time_load_and_combine_pipeline(n_iterations=n_iterations)

    # Compare data consistency for the first skycell
    print("\n🔍 Comparing Data Consistency (first skycell)...")
    is_consistent = compare_data_consistency(zarr_results["sample"], pipeline_results["sample"])

    # Test saving performance (using first skycell's data)
    print("\n💾 Testing Save Performance...")
    zarr_save_time = time_save_zarr(zarr_results["sample"]["combined_image"], zarr_results["sample"]["combined_mask"], n_iterations=n_iterations)
    fits_save_time = time_save_fits(zarr_results["sample"]["combined_image"], zarr_results["sample"]["combined_mask"], n_iterations=n_iterations)

    # Print summary
    print("\n📈 Performance Summary")
    print("=" * 30)
    print(f"Zarr Loading (avg, 10 skycells): {zarr_time:.4f}s")
    # print(f"FITS Loading (avg, 10 skycells): {fits_time:.4f}s")
    print(f"load_and_combine pipeline (avg per skycell): {pipeline_time:.4f}s")

    if zarr_time > 0 and pipeline_time > 0:
        speedup = pipeline_time / zarr_time
        print(f"Zarr speedup: {speedup:.2f}x")

    print(f"\nZarr Save (avg): {zarr_save_time:.4f}s")
    print(f"FITS Save (avg): {fits_save_time:.4f}s")

    if zarr_save_time > 0 and fits_save_time > 0:
        save_speedup = fits_save_time / zarr_save_time
        print(f"Zarr save speedup: {save_speedup:.2f}x")

    print(f"\nData Consistency (first skycell): {'✅ PASS' if is_consistent else '❌ FAIL'}")

    # Detailed results for first skycell
    print("\nTest Details (first skycell):")
    print(f"- Zarr source: {zarr_results['sample']['projection']}/{zarr_results['sample']['skycell']}")
    # print(f"- FITS source: {fits_results['sample'].get('fits_dir', 'N/A')}")
    print(f"- Bands tested: {zarr_results['sample']['bands_loaded']}")
    print(f"- Image shape: {zarr_results['sample']['image_shape']}")
    print(f"- Iterations: {n_iterations} skycells")
    print(f"- Pipeline sample keys: {list(pipeline_results['sample'].keys()) if pipeline_results['sample'] else 'N/A'}")

    print("\n🎯 Test completed successfully!")
    return 0


def time_load_and_combine_pipeline(n_iterations: int = 10) -> tuple[float, dict]:
    """Time the load_and_combine pipeline for multiple skycells."""
    logger.info(f"Testing load_and_combine pipeline for {n_iterations} skycells...")
    try:
        import importlib

        load_and_combine = importlib.import_module("load_and_combine")
        from zarr_utils import get_available_projections, get_projection_skycells

        zarr_path = "data/ps1_skycells_zarr/ps1_skycells.zarr"
        projections = get_available_projections(zarr_path)
        skycell_list = []
        for proj in projections:
            skycells = get_projection_skycells(zarr_path, proj)
            for skycell_full in skycells:
                # Convert full format "skycell.2556.080" to short format "080"
                if skycell_full.startswith("skycell."):
                    skycell_short = skycell_full.split(".")[-1]  # Get the last part
                else:
                    skycell_short = skycell_full
                skycell_list.append((proj, skycell_short))
                if len(skycell_list) >= n_iterations:
                    break
            if len(skycell_list) >= n_iterations:
                break
        if not skycell_list:
            logger.error("No skycells found for pipeline test")
            return 0.0, {}
        start_time = time.time()
        results = load_and_combine.run_processing_pipeline(zarr_path, skycell_list, num_producers=2, num_consumers=4)
        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"Pipeline processed {len(results)} skycells in {elapsed:.4f}s")
        # Pick a sample result for summary
        sample_key = next(iter(results)) if results else None
        sample = results[sample_key] if sample_key else {}
        return elapsed / len(results) if results else 0.0, {"sample": sample, "all": results}
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        return 0.0, {}


if __name__ == "__main__":
    import sys

    exit_code = main()
    sys.exit(exit_code)
