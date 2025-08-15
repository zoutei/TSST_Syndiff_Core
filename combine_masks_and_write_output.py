"""Combine PS1 per-band masks for all skycells listed in a master CSV and
write the combined mask into a sector/camera/ccd output Zarr store.

Behavior (exactly):
- Take sector, camera, ccd as positional arguments.
- Find the master CSV using `csv_utils.find_csv_file`.
- For every skycell listed in the CSV (column `NAME` if present, fallback to `skycell`),
  load its band masks from the main PS1 Zarr store and combine them.
- Save the combined mask into the output zarr store at
  `data/convolved_results/sector_{sector:04d}_camera_{camera}_ccd_{ccd}.zarr`
  using the key `<skycell_name>_mask`. Overwrite any existing array with the same key.

This script does nothing else.
"""

import argparse
import logging
from collections.abc import Iterable

import numpy as np
import zarr

from band_utils import combine_masks
from csv_utils import find_csv_file, load_csv_data
from zarr_utils import load_skycell_masks

logger = logging.getLogger("combine_masks")


def iter_skycells_from_csv(csv_path: str) -> Iterable[str]:
    df = load_csv_data(csv_path)

    # Prefer column NAME (used elsewhere in this repo). Fallbacks for other common names.
    if "NAME" in df.columns:
        names = df["NAME"].astype(str).unique()
    elif "skycell" in df.columns:
        names = df["skycell"].astype(str).unique()
    else:
        raise ValueError(f"CSV at {csv_path} does not contain a 'NAME' or 'skycell' column")

    for n in names:
        n = str(n).strip()
        if n:
            yield n


def combine_and_save(sector: int, camera: int, ccd: int, data_root: str = "data") -> None:
    # Locate CSV
    csv_path = find_csv_file(data_root, sector, camera, ccd)
    logger.info(f"Using skycell CSV: {csv_path}")

    # Input Zarr store containing the original PS1 skycells
    input_zarr = f"{data_root}/ps1_skycells_zarr/ps1_skycells.zarr"

    # Output Zarr store for this sector/camera/ccd
    output_zarr = f"{data_root}/convolved_results/sector_{sector:04d}_camera_{camera}_ccd_{ccd}.zarr"
    store_out = zarr.open(output_zarr, mode="a")
    store_read = zarr.open(input_zarr, mode="r")
    count = 0
    errors = 0

    for skycell_name in iter_skycells_from_csv(csv_path):
        try:
            # Expect skycell_name like 'skycell.2556.080' or similar
            parts = skycell_name.split(".")
            if len(parts) >= 3 and parts[0].lower().startswith("skycell"):
                projection = parts[1]
            elif len(parts) >= 2 and parts[0].isdigit():
                # sometimes CSVs can list projection then cell; be conservative
                projection = parts[0]
            else:
                # Fallback: try to find projection by searching for digits in the string
                projection = next((p for p in parts if p.isdigit()), None)

            if projection is None:
                logger.warning(f"Could not determine projection for skycell '{skycell_name}', skipping")
                errors += 1
                continue

            # Load masks for this skycell from the input store
            masks_data = load_skycell_masks(store_read, projection, skycell_name)

            # Combine masks (returns uint16 or None)
            combined = combine_masks(masks_data) if masks_data else None

            # Ensure dtype uint16
            if combined.dtype != np.uint16:
                combined = combined.astype(np.uint16)

            # Save to output store with key '<skycell_name>_mask' (overwrite if exists)
            mask_key = f"{skycell_name}_mask"
            if mask_key in store_out:
                del store_out[mask_key]

            store_out[mask_key] = combined
            count += 1
            logger.info(f"Wrote combined mask for {skycell_name} -> {output_zarr}:{mask_key}")

        except Exception as exc:
            logger.warning(f"Failed to process {skycell_name}: {exc}")
            errors += 1
            continue

    logger.info(f"Completed: wrote {count} masks, {errors} errors")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Combine per-band PS1 masks for skycells and save into sector output zarr")
    parser.add_argument("sector", type=int)
    parser.add_argument("camera", type=int)
    parser.add_argument("ccd", type=int)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")

    combine_and_save(args.sector, args.camera, args.ccd, data_root=args.data_root)


if __name__ == "__main__":
    main()
