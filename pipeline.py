"""
End-to-end pipeline runner:
  1) Run Pancakes v2 to generate TESS↔PS1 mapping files and skycell list CSV
  2) Download PS1 skycells (images and masks) for listed skycells
  3) Combine PS1 bands and convolve (process_ps1)
  4) Downsample combined images into TESS grid

Only this file uses CLI args. Individual step modules expose functions.
"""

import argparse
import logging
from pathlib import Path

from download_ps1_skycells import download_from_csv
from downsample import run_downsample
from pancakes_v2 import process_tess_image_optimized
from process_ps1 import ProcessingConfig, run_ps1_processing_pipeline


def derive_paths(sector: int, ccd_id: int, data_root: str) -> dict:
    sector4 = f"{sector:04d}"
    ccd = int(ccd_id)
    mapping_root = Path(data_root) / "mapping_output"
    mapping_dir = mapping_root / f"sector_{sector4}" / f"ccd_{ccd}"
    skycells_csv = mapping_dir / f"tess_s{sector4}_{ccd}_master_skycells_list.csv"
    ps1_skycells_dir = Path(data_root) / "ps1_skycells"
    save_dir = Path(data_root) / "tess_comb_skycells" / f"s{sector4}_ccd{ccd}"
    registrations_glob = str(mapping_dir / f"TESS_s{sector4}_{ccd}_skycell.*.fits.gz")
    return {
        "mapping_root": str(mapping_root),
        "mapping_dir": str(mapping_dir),
        "skycells_csv": str(skycells_csv),
        "ps1_skycells_dir": str(ps1_skycells_dir),
        "save_dir": str(save_dir),
        "registrations_glob": registrations_glob,
    }


def run_pipeline(sector: int, ccd_id: int, tess_fits: str, skycell_wcs_csv: str, data_root: str = "data", cores: int = 8, jobs: int = 60, overwrite: bool = False, verbose: int = 0):
    logging.basicConfig(level=logging.INFO if verbose == 0 else logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

    paths = derive_paths(sector, ccd_id, data_root)
    Path(paths["mapping_root"]).mkdir(parents=True, exist_ok=True)
    Path(paths["ps1_skycells_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["save_dir"]).mkdir(parents=True, exist_ok=True)

    # 1) Pancakes v2
    process_tess_image_optimized(
        tess_file=tess_fits,
        skycell_wcs_csv=skycell_wcs_csv,
        output_path=paths["mapping_root"],
        buffer=120,
        tess_buffer=150,
        n_threads=8,
        overwrite=True,
        max_workers=cores,
    )

    # 2) Download PS1 skycells for listed entries
    download_from_csv(
        csv_path=paths["skycells_csv"],
        save_path=paths["ps1_skycells_dir"],
        jobs=jobs,
        filters="rizy",
        download_masks=True,
    )

    # 3) Combine PS1 bands and convolve
    config = ProcessingConfig(
        sector=sector,
        ccd_id=ccd_id,
        data_root=data_root,
        cores=cores,
        overwrite=overwrite,
        verbose=verbose,
        use_mask=True,
    )
    run_ps1_processing_pipeline(config)

    # 4) Downsample to TESS grid
    skycell_conv_dir = str(Path(paths["save_dir"]) / "final_conv/")
    run_downsample(
        tess_filename=tess_fits,
        skycell_path=skycell_conv_dir,
        registrations_glob=paths["registrations_glob"],
        n_jobs=jobs,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full PS1→TESS pipeline")
    p.add_argument("--sector", type=int, required=True)
    p.add_argument("--ccd-id", type=int, required=True)
    p.add_argument("--tess-fits", required=True, help="Path to TESS FFI FITS file")
    p.add_argument("--skycell-wcs-csv", required=True, help="Path to skycell WCS catalog CSV for Pancakes")
    p.add_argument("--data-root", default="data")
    p.add_argument("--cores", type=int, default=8)
    p.add_argument("--jobs", type=int, default=60)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="count", default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        sector=args.sector,
        ccd_id=args.ccd_id,
        tess_fits=args.tess_fits,
        skycell_wcs_csv=args.skycell_wcs_csv,
        data_root=args.data_root,
        cores=args.cores,
        jobs=args.jobs,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
