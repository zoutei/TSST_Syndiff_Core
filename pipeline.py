"""
End-to-end pipeline runner:
  1) Run Pancakes v2 to generate TESS↔PS1 mapping files and skycell list CSV
  2) Download PS1 skycells (images and masks) to Zarr format for listed skycells
  3) Combine PS1 bands and convolve using modern sliding window pipeline (process_ps1)
  4) Multi-offset downsample combined images into TESS grid

Only this file uses CLI args. Individual step modules expose functions.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from astropy.io import fits

from download_and_store_zarr import download_and_store_ps1_data
from multi_offset_downsampling import main as run_multi_offset_downsampling
from pancakes_v2 import process_tess_image_optimized
from process_ps1 import run_modern_sliding_window_pipeline


def extract_tess_metadata(tess_fits: str) -> tuple[int, int, int]:
    """
    Extract sector, camera, and CCD information from TESS FITS file.

    Args:
        tess_fits: Path to TESS FFI FITS file

    Returns:
        Tuple of (sector, camera, ccd)
    """
    with fits.open(tess_fits) as hdul:
        header = hdul[1].header

        # Extract sector from filename (e.g., tess2020123456-s0020-...)
        sector = int(tess_fits.split("/")[-1].split("-")[1][1:])

        # Extract camera and CCD from header
        camera = int(header["CAMERA"])  # camera_id (1-4)
        ccd = int(header["CCD"])  # ccd_id (1-4)

    return sector, camera, ccd


def derive_paths(sector: int, camera: int, ccd_id: int, data_root: str) -> dict:
    sector4 = f"{sector:04d}"
    mapping_root = Path(data_root) / "mapping_output"
    mapping_dir = mapping_root / f"sector_{sector4}" / f"camera_{camera}" / f"ccd_{ccd_id}"
    skycells_csv = mapping_dir / f"tess_s{sector4}_{camera}_{ccd_id}_master_skycells_list.csv"
    ps1_skycells_zarr = Path(data_root) / "ps1_skycells_zarr"
    convolved_results_dir = Path(data_root) / "convolved_results" / f"sector_{sector4}" / f"camera_{camera}" / f"ccd_{ccd_id}"
    registrations_glob = str(mapping_dir / f"TESS_s{sector4}_{camera}_{ccd_id}_skycell.*.fits.gz")
    return {
        "mapping_root": str(mapping_root),
        "mapping_dir": str(mapping_dir),
        "skycells_csv": str(skycells_csv),
        "ps1_skycells_zarr": str(ps1_skycells_zarr),
        "convolved_results_dir": str(convolved_results_dir),
        "registrations_glob": registrations_glob,
    }


def run_pipeline(tess_fits: str, skycell_wcs_csv: str = "data/SkyCells/skycell_wcs.csv", data_root: str = "data", cores: int = 8, jobs: int = 60, overwrite: bool = False, verbose: int = 0, multi_offset_array: str = "0.0,0.0", ignore_mask_bits: str = "12"):
    """
    Run the complete PS1→TESS pipeline.

    Args:
        tess_fits: Path to TESS FFI FITS file
        skycell_wcs_csv: Path to skycell WCS catalog CSV for Pancakes (default: data/SkyCells/skycell_wcs.csv)
        data_root: Root directory for data storage
        cores: Number of CPU cores to use
        jobs: Number of parallel jobs for downloading
        overwrite: Whether to overwrite existing files
        verbose: Verbosity level (0=INFO, 1+=DEBUG)
        multi_offset_array: Comma-separated dx,dy pairs for downsampling (e.g., "0.0,0.0,0.5,0.0,0.0,0.5,0.5,0.5")
        ignore_mask_bits: Comma-separated list of mask bits to ignore (e.g., "12,13")
    """
    logging.basicConfig(level=logging.INFO if verbose == 0 else logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

    # Extract sector, camera, and CCD automatically from TESS FITS file
    sector, camera, ccd_id = extract_tess_metadata(tess_fits)
    logging.info(f"Extracted metadata from TESS file: sector={sector}, camera={camera}, ccd={ccd_id}")

    # Parse multi-offset array
    offset_values = [float(x) for x in multi_offset_array.split(",")]
    if len(offset_values) % 2 != 0:
        raise ValueError("multi_offset_array must contain an even number of values (dx,dy pairs)")
    offsets = np.array([(offset_values[i], offset_values[i + 1]) for i in range(0, len(offset_values), 2)])

    # Parse ignore mask bits
    ignore_bits = [int(x) for x in ignore_mask_bits.split(",") if x.strip()]

    paths = derive_paths(sector, camera, ccd_id, data_root)
    Path(paths["mapping_root"]).mkdir(parents=True, exist_ok=True)
    Path(paths["ps1_skycells_zarr"]).mkdir(parents=True, exist_ok=True)
    Path(paths["convolved_results_dir"]).mkdir(parents=True, exist_ok=True)

    logging.info(f"Starting pipeline for TESS sector {sector}, camera {camera}, CCD {ccd_id}")

    # 1) Pancakes v2: Generate TESS↔PS1 mapping files and skycell list CSV
    logging.info("Step 1: Running Pancakes v2 for TESS↔PS1 mapping...")
    process_tess_image_optimized(
        tess_file=tess_fits,
        skycell_wcs_csv=skycell_wcs_csv,
        output_path=paths["mapping_root"],
        buffer=120,
        tess_buffer=150,
        n_threads=8,
        overwrite=overwrite,
        max_workers=cores,
    )

    # 2) Download PS1 skycells to Zarr format
    logging.info("Step 2: Downloading PS1 skycells to Zarr format...")
    download_result = download_and_store_ps1_data(
        sector=sector,
        camera=camera,
        ccd=ccd_id,
        num_workers=jobs,
        zarr_output_dir=paths["ps1_skycells_zarr"],
        use_local_files=False,
        log_level="DEBUG" if verbose > 0 else "INFO",
        overwrite=overwrite,
    )

    if download_result["status"] != "completed":
        raise RuntimeError(f"PS1 download failed: {download_result['message']}")
    logging.info(f"PS1 data successfully stored in: {download_result['zarr_path']}")

    # 3) Combine PS1 bands and convolve using modern sliding window pipeline
    logging.info("Step 3: Running modern sliding window pipeline for PS1 processing...")
    run_modern_sliding_window_pipeline(
        sector=sector,
        camera=camera,
        ccd=ccd_id,
        data_root=data_root,
        projections_limit=None,  # Process all projections
        psf_sigma=60.0,  # Default PSF sigma
    )

    # 4) Multi-offset downsample to TESS grid
    logging.info("Step 4: Running multi-offset downsampling to TESS grid...")
    run_multi_offset_downsampling(
        sector=sector,
        camera=camera,
        ccd=ccd_id,
        offsets=offsets,
        ignore_mask_bits=ignore_bits,
        data_root=data_root,
        convolved_dir=paths["convolved_results_dir"],
        output_base=None,  # Use default output path
    )

    logging.info(f"Pipeline completed successfully for sector {sector}, camera {camera}, CCD {ccd_id}")
    return {
        "status": "completed",
        "sector": sector,
        "camera": camera,
        "ccd": ccd_id,
        "paths": paths,
        "offsets": offsets.tolist(),
        "ignore_mask_bits": ignore_bits,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full PS1→TESS pipeline")
    p.add_argument("tess_fits", help="Path to TESS FFI FITS file")
    p.add_argument("skycell_wcs_csv", nargs="?", default="data/SkyCells/skycell_wcs.csv", help="Path to skycell WCS catalog CSV for Pancakes (default: data/SkyCells/skycell_wcs.csv)")
    p.add_argument("--data-root", default="data", help="Root directory for data storage")
    p.add_argument("--cores", type=int, default=8, help="Number of CPU cores to use")
    p.add_argument("--jobs", type=int, default=60, help="Number of parallel jobs for downloading")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    p.add_argument("--verbose", action="count", default=0, help="Increase verbosity (use -v for INFO, -vv for DEBUG)")
    p.add_argument("--multi-offset-array", default="0.0,0.0", help="Comma-separated dx,dy pairs for downsampling (e.g., '0.0,0.0,0.5,0.0,0.0,0.5,0.5,0.5')")
    p.add_argument("--ignore-mask-bits", default="12", help="Comma-separated list of mask bits to ignore (e.g., '12,13')")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_pipeline(
        tess_fits=args.tess_fits,
        skycell_wcs_csv=args.skycell_wcs_csv,
        data_root=args.data_root,
        cores=args.cores,
        jobs=args.jobs,
        overwrite=args.overwrite,
        verbose=args.verbose,
        multi_offset_array=args.multi_offset_array,
        ignore_mask_bits=args.ignore_mask_bits,
    )
    print(f"Pipeline completed: {result}")
