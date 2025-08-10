#!/usr/bin/env python3
"""
Simple PS1 processing pipeline - Function-oriented approach.

Main pipeline for processing PS1 data with row-based processing,
zarr I/O, padding, and convolution for TESS synthetic difference imaging.
"""

import gc
import logging
import os

# Set thread limits
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Import our simple utilities
from band_utils import process_skycell_bands
from convolution_utils import apply_tess_psf_simulation
from csv_utils import find_csv_file, get_all_padding_cells, get_projection_rows, get_projections_from_csv
from padding_utils import smart_padding_with_csv
from zarr_utils import load_skycell_bands, load_skycell_masks, save_convolved_results

logger = logging.getLogger(__name__)


def process_single_projection(projection: str, zarr_path: str, csv_path: str, output_path: str, pad_size: int = 300, psf_sigma: float = 4.0) -> dict:
    """Process a complete projection with all its rows.

    Args:
        projection: PS1 projection ID
        zarr_path: Input zarr path
        csv_path: CSV file with mapping info
        output_path: Output zarr path
        pad_size: Padding size in pixels
        psf_sigma: PSF sigma for convolution

    Returns:
        Processing statistics
    """
    logger.info(f"Processing projection {projection}")

    # Get rows for this projection
    projection_rows = get_projection_rows(csv_path, projection)
    logger.info(f"Found {len(projection_rows)} rows in projection {projection}")

    processed_rows = 0
    total_cells = 0

    # Process each row
    for row_id, row_skycells in projection_rows.items():
        try:
            stats = process_row_with_padding(row_skycells, projection, row_id, zarr_path, csv_path, output_path, pad_size, psf_sigma)
            processed_rows += 1
            total_cells += stats["cells_processed"]

            # Clean up memory after each row
            gc.collect()

        except Exception as e:
            logger.error(f"Failed to process row {row_id} in projection {projection}: {e}")
            continue

    logger.info(f"Completed projection {projection}: {processed_rows}/{len(projection_rows)} rows, {total_cells} cells")

    return {"projection": projection, "rows_processed": processed_rows, "total_rows": len(projection_rows), "cells_processed": total_cells}


def process_row_with_padding(row_skycells: list[str], projection: str, row_id: int, zarr_path: str, csv_path: str, output_path: str, pad_size: int = 300, psf_sigma: float = 4.0) -> dict:
    """Process a single row with padding and convolution.

    Args:
        row_skycells: List of skycells in this row
        projection: PS1 projection ID
        row_id: Row identifier
        zarr_path: Input zarr path
        csv_path: CSV file path
        output_path: Output zarr path
        pad_size: Padding size in pixels
        psf_sigma: PSF sigma for convolution

    Returns:
        Processing statistics
    """
    logger.info(f"Processing row {row_id} in projection {projection}: {len(row_skycells)} cells")

    # Step 1: Get all padding cells needed for this row
    all_padding = get_all_padding_cells(csv_path, row_skycells)
    unique_padding_cells = set()
    for padding_list in all_padding.values():
        unique_padding_cells.update(padding_list)

    logger.info(f"Need to load {len(unique_padding_cells)} unique padding cells")

    # Step 2: Load all required images (row + padding)
    all_needed_cells = [(projection, cell) for cell in row_skycells]

    # Add padding cells (could be from different projections)
    for padding_cell in unique_padding_cells:
        # For now, assume padding cells are from same projection
        # TODO: Handle cross-projection padding
        all_needed_cells.append((projection, padding_cell))

    # Load all cell images
    logger.info(f"Loading {len(all_needed_cells)} cells total...")
    all_images = {}
    all_masks = {}

    for proj, cell in all_needed_cells:
        try:
            # Load bands and combine
            bands_data = load_skycell_bands(zarr_path, proj, cell)
            masks_data = load_skycell_masks(zarr_path, proj, cell)

            # Combine bands
            combined_image, combined_mask = process_skycell_bands(bands_data, masks_data)

            all_images[cell] = combined_image
            all_masks[cell] = combined_mask

            # Clean up individual band data
            del bands_data, masks_data

        except Exception as e:
            logger.warning(f"Failed to load {proj}/{cell}: {e}")
            continue

    logger.info(f"Successfully loaded {len(all_images)} cells")

    # Step 3: Process each cell in the row with padding
    row_results = {}

    for cell in row_skycells:
        if cell not in all_images:
            logger.warning(f"Skipping {cell} - not loaded")
            continue

        try:
            # Get CSV padding info for this cell
            from csv_utils import get_padding_info

            padding_info = get_padding_info(csv_path, cell)

            # Apply padding
            padded_image = smart_padding_with_csv(all_images[cell], padding_info, all_images, pad_size)

            # Apply convolution
            convolved_image = apply_tess_psf_simulation(padded_image, psf_type="gaussian", sigma=psf_sigma)

            row_results[cell] = convolved_image

            logger.debug(f"Processed {cell}: {all_images[cell].shape} -> {convolved_image.shape}")

        except Exception as e:
            logger.error(f"Failed to process {cell}: {e}")
            continue

    # Step 4: Save results
    cells_processed = len(row_results)
    if row_results:
        save_convolved_results(output_path, projection, row_id, row_results)

    # Step 5: Clean up
    del all_images, all_masks, row_results
    gc.collect()

    logger.info(f"Completed row {row_id}: {cells_processed} cells processed")

    return {"row_id": row_id, "cells_processed": cells_processed, "cells_in_row": len(row_skycells)}


def run_simple_pipeline(sector: int, camera: int, ccd: int, data_root: str = "data", projections_limit: int = None) -> dict:
    """Run the complete simple pipeline for a sector/camera/ccd.

    Args:
        sector: TESS sector number
        camera: TESS camera number
        ccd: TESS CCD number
        data_root: Root data directory
        projections_limit: Limit number of projections (for testing)

    Returns:
        Processing statistics
    """
    logger.info(f"Starting simple pipeline for sector {sector}, camera {camera}, ccd {ccd}")

    # Set up paths
    zarr_path = f"{data_root}/ps1_skycells_zarr/ps1_skycells.zarr"
    output_path = f"{data_root}/convolved_results/sector_{sector:04d}_camera_{camera}_ccd_{ccd}.zarr"

    try:
        csv_path = find_csv_file(data_root, sector, camera, ccd)
    except FileNotFoundError as e:
        logger.error(f"CSV file not found: {e}")
        return {"error": str(e)}

    logger.info("Using paths:")
    logger.info(f"  Zarr input: {zarr_path}")
    logger.info(f"  CSV file: {csv_path}")
    logger.info(f"  Output: {output_path}")

    # Get projections to process
    try:
        projections = get_projections_from_csv(csv_path)
        if projections_limit:
            projections = projections[:projections_limit]
            logger.info(f"Limited to {len(projections)} projections for testing")
    except Exception as e:
        logger.error(f"Failed to get projections: {e}")
        return {"error": str(e)}

    # Process each projection
    results = []
    for i, projection in enumerate(projections):
        logger.info(f"Progress: {i + 1}/{len(projections)} projections")

        try:
            result = process_single_projection(projection, zarr_path, csv_path, output_path)
            results.append(result)

        except Exception as e:
            logger.error(f"Failed to process projection {projection}: {e}")
            continue

    # Summary
    total_rows = sum(r["rows_processed"] for r in results)
    total_cells = sum(r["cells_processed"] for r in results)

    logger.info("Pipeline completed!")
    logger.info(f"  Processed {len(results)}/{len(projections)} projections")
    logger.info(f"  Total rows: {total_rows}")
    logger.info(f"  Total cells: {total_cells}")

    return {"sector": sector, "camera": camera, "ccd": ccd, "projections_processed": len(results), "total_projections": len(projections), "total_rows": total_rows, "total_cells": total_cells, "results": results}


if __name__ == "__main__":
    import argparse

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Simple PS1 processing pipeline")
    parser.add_argument("sector", type=int, help="TESS sector number")
    parser.add_argument("camera", type=int, help="TESS camera number")
    parser.add_argument("ccd", type=int, help="TESS CCD number")
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument("--limit", type=int, help="Limit projections for testing")

    args = parser.parse_args()

    results = run_simple_pipeline(args.sector, args.camera, args.ccd, data_root=args.data_root, projections_limit=args.limit)

    if "error" not in results:
        print("\n✅ Pipeline completed successfully!")
        print(f"Processed {results['projections_processed']} projections")
        print(f"Total cells processed: {results['total_cells']}")
    else:
        print(f"\n❌ Pipeline failed: {results['error']}")
        exit(1)
