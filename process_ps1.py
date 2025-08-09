"""Minimal PS1 processing pipeline.

Two clear steps for each skycell:
- Combine raw PS1 r,i,z,y into rizy combined image (+ mask)
- Pad combined image (if mapping indicates) and convolve to final outputs

Set sector/camera/ccd to auto-derive inputs/outputs; or override paths directly.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
from typing import List, Tuple, Optional
import logging
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from astropy.modeling import models
from astropy.io import fits
from astropy.wcs import WCS
from scipy.signal import fftconvolve
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from smart_padding import (
    smart_pad_ps1_image_and_mask,
    PaddingConfig,
    PaddingRequirements,
    check_tess_mapping_padding,
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class ProcessingConfig:
    """Configuration for PS1 processing pipeline.

    Prefer setting `sector`, `camera_id`, and `ccd_id`; paths will be derived automatically.
    You may override any derived paths explicitly.
    """

    # Canonical high-level inputs
    sector: int
    camera_id: int
    ccd_id: int
    data_root: str = "data"

    # Data paths (auto-derived if not supplied)
    ps1_skycells_path: Optional[str] = None
    savepath: Optional[str] = None
    linear_comb_savepath: Optional[str] = None
    tess_conv_savepath: Optional[str] = None
    catalog_path: Optional[str] = None  # directory containing per-skycell csv files
    mapping_path: Optional[str] = None  # directory containing mapping FITS
    skycell_list_csv: Optional[str] = None 

    # Processing parameters
    psf_std: float = 60.0
    psf_kernal_size: float = 2001
    combine_weights: Optional[List[float]] = None  # r, i, z, y
    pad_distance: int = 500
    edge_exclusion: int = 50
    suffix: str = "rizy.conv"

    # Execution parameters
    cores: int = 5
    overwrite: bool = False
    use_mask: bool = True
    verbose: int = 0

    def __post_init__(self):
        # Weights default
        if self.combine_weights is None:
            self.combine_weights = [0.238, 0.344, 0.283, 0.135]

        # Defaults for PS1 inputs
        if self.ps1_skycells_path is None:
            self.ps1_skycells_path = str(Path(self.data_root) / "ps1_skycells")

        # Savepath and subfolders
        sector4 = f"{self.sector:04d}"

        if self.savepath is None:
            self.savepath = str(Path(self.data_root)/ "tess_ps1_skycells"/ f"sector_{sector4}"/ f"camera_{self.camera_id}"/ f"ccd_{self.ccd_id}")

        if self.linear_comb_savepath is None:
            self.linear_comb_savepath = str(Path(self.savepath) / "linear_comb")

        if self.tess_conv_savepath is None:
            # Keep for compatibility; final outputs go under savepath/final_conv
            self.tess_conv_savepath = str(Path(self.savepath) / "tess_conv")

        if self.mapping_path is None:
            self.mapping_path = str(Path(self.data_root)/ "skycell_pixel_mapping"/ f"sector_{sector4}"/ f"camera_{self.camera_id}"/ f"ccd_{self.ccd_id}")

        if self.skycell_list_csv is None:
            self.skycell_list_csv = str(Path(self.mapping_path)/ f"tess_s{sector4}_{self.camera_id}_{self.ccd_id}_master_skycells_list.csv")


@dataclass
class ProcessedData:
    combined_image: np.ndarray
    combined_mask: Optional[np.ndarray]
    header: fits.Header


def load_and_combine_ps1_bands(
    file_pattern: str,
    bands: List[str],
    weights: List[float],
    use_mask: bool = True,
) -> ProcessedData:
    combined_image = None
    combined_mask = None
    reference_header = None

    weights_array = np.array(weights)

    for i, (band, weight) in enumerate(zip(bands, weights_array)):
        filepath = f"{file_pattern}{band}.unconv.fits"

        with fits.open(filepath) as hdul:
            hdu_index = 1 if len(hdul) > 1 else 0
            raw_data = hdul[hdu_index].data.astype(np.float32)
            header = hdul[hdu_index].header
            hdul.close()

            # Convert from log scale to flux using BOFFSET/BSOFTEN
            if "BOFFSET" in header and "BSOFTEN" in header:
                a = 2.5 / np.log(10)
                x = raw_data / a
                flux = header["BOFFSET"] + header["BSOFTEN"] * 2 * np.sinh(x)
                data = flux / header["EXPTIME"]
            else:
                data = raw_data

        if i == 0:
            reference_header = header
            combined_image = np.zeros_like(data, dtype=np.float32)
            if use_mask:
                combined_mask = np.zeros_like(data, dtype=np.uint32)

        combined_image += data * weight

        if use_mask:
            mask_filepath = filepath.replace(".fits", ".mask.fits")
            if Path(mask_filepath).exists():
                try:
                    with fits.open(mask_filepath) as mask_hdul:
                        mask_hdu_index = 1 if len(mask_hdul) > 1 else 0
                        mask_data = mask_hdul[mask_hdu_index].data.astype(np.uint32)
                        combined_mask = combined_mask | mask_data
                except Exception as exc:
                    logger.warning(f"Could not load mask {mask_filepath}: {exc}")

    return ProcessedData(
        combined_image=combined_image,
        combined_mask=combined_mask,
        header=reference_header,
    )


def save_fits_image(data: np.ndarray, header: fits.Header, filepath: Path, overwrite: bool = False) -> None:
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdul = fits.HDUList([hdu])
    filepath.parent.mkdir(parents=True, exist_ok=True)
    hdul.writeto(filepath, overwrite=overwrite)


def create_gaussian_psf(size: int = 2001, std: float = 60.0) -> np.ndarray:
    y, x = np.mgrid[:size, :sizey, x = np.mgrid[:size, :size]]
    x = x - size / 2
    y = y - size / 2
    psf = models.Gaussian2D(x_stddev=std, y_stddev=std)(x, y)
    return psf / np.nansum(psf)


def create_output_header(header: fits.Header, config: ProcessingConfig, conv: bool = False) -> fits.Header:
    output_header = header.copy()
    bad_keys = ["HISTORY", "INP_*", "SCL_*", "ZPT_*", "EXP_*", "AIR_*", "HIERARCH*", "BSOFTEN", "BOFFSET"]
    for key in list(output_header.keys()):
        for bad_pattern in bad_keys:
            if bad_pattern.endswith("*") and key.startswith(bad_pattern[:-1]):
                try:
                    del output_header[key]
                except KeyError:
                    pass
                break
            elif key == bad_pattern:
                try:
                    del output_header[key]
                except KeyError:
                    pass
                break

    output_header["FILTER"] = ("rizy", "Filter used")
    output_header["COMBINE"] = (True, "Combined image")
    output_header["FRACR"] = (config.combine_weights[0], "Fraction of r band used")
    output_header["FRACI"] = (config.combine_weights[1], "Fraction of i band used")
    output_header["FRACZ"] = (config.combine_weights[2], "Fraction of z band used")
    output_header["FRACY"] = (config.combine_weights[3], "Fraction of y band used")
    if conv:
        output_header["PSFTYPE"] = ("Gaussian", "Type of PSF used in convolution")
        output_header["PSFSTD"] = (config.psf_std, "Standard deviation of Gaussian PSF")
        output_header["PSF_KERNAL"] = (config.psf_kernal_size, "Kernal size of Gaussian PSF")
    return output_header


def setup_output_directories(config: ProcessingConfig) -> None:
    for d in [
        Path(config.savepath),
        Path(config.linear_comb_savepath),
        Path(config.tess_conv_savepath),
    ]:
        d.mkdir(parents=True, exist_ok=True)


def _extract_projection_and_cell_from_skycell_name(skycell_name: str) -> Tuple[str, str]:
    parts = skycell_name.split('.')
    if len(parts) >= 3 and parts[0] == 'skycell':
        return parts[1], parts[2]
    raise ValueError(f"Unexpected skycell name format: {skycell_name}")


def _build_raw_file_pattern(datapath: str, skycell_name: str) -> str:
    projection, cell = _extract_projection_and_cell_from_skycell_name(skycell_name)
    base = f"rings.v3.{skycell_name}.stk."
    return str(Path(datapath) / projection / cell / base)


def _combined_output_paths(config: ProcessingConfig, skycell_name: str) -> Tuple[Path, Path]:
    base = f"rings.v3.{skycell_name}.stk.rizy.unconv_combined.fits.gz"
    img_path = Path(config.linear_comb_savepath) / base
    mask_path = Path(str(img_path).replace('.fits.gz', '.mask.fits.gz'))
    return img_path, mask_path


def _final_output_paths(config: ProcessingConfig, skycell_name: str) -> Tuple[Path, Path]:
    base_prefix = f"rings.v3.{skycell_name}.stk."
    final_dir = Path(config.tess_conv_savepath)
    img_path = final_dir / f"{base_prefix}{config.suffix}.fits.gz"
    mask_path = Path(str(img_path).replace('.fits.gz', '.mask.fits.gz'))
    return img_path, mask_path


def _load_skycell_names(skycell_list_csv: str) -> List[str]:
    df = pd.read_csv(skycell_list_csv)
    name_col = 'NAME' if 'NAME' in df.columns else ('Name' if 'Name' in df.columns else None)
    if name_col is None:
        raise ValueError("Skycell list CSV must contain a 'NAME' or 'Name' column")
    return list(df[name_col].values)


def combine_one_skycell(config: ProcessingConfig, skycell_name: str) -> bool:
    """Combine raw PS1 r,i,z,y bands into rizy combined image (and mask) for one skycell."""
    try:
        bands = ["r", "i", "z", "y"]
        combined_img_path, combined_mask_path = _combined_output_paths(config, skycell_name)
        if combined_img_path.exists() and not config.overwrite:
            logger.info(f"Combine exists, skipping: {combined_img_path.name}")
            return True

        file_pattern = _build_raw_file_pattern(config.ps1_skycells_path, skycell_name)
        processed = load_and_combine_ps1_bands(
            file_pattern, bands, config.combine_weights, use_mask=config.use_mask
        )
        header = create_output_header(processed.header, config, conv=False)
        save_fits_image(processed.combined_image, header, combined_img_path, config.overwrite)
        if processed.combined_mask is not None:
            mask_header = header.copy()
            mask_header["MASK"] = (True, "PS1 bitmask used in combination")
            mask_header["MASKTYPE"] = ("PS1_BITMASK", "PS1 multi-bit binary flag system")
            mask_header["MASKFMT"] = ("UINT32", "Mask data format preserving all flags")
            save_fits_image(processed.combined_mask.astype(np.uint32), mask_header, combined_mask_path, config.overwrite)
        return True
    except Exception as exc:
        logger.error(f"Failed to combine {skycell_name}: {exc}")
        return False


def pad_and_convolve_one_skycell(config: ProcessingConfig, skycell_name: str) -> bool:
    """Pad a combined skycell image (if needed) and convolve to final outputs."""
    try:
        combined_img_path, combined_mask_path = _combined_output_paths(config, skycell_name)
        if not combined_img_path.exists():
            logger.warning(f"Missing combined image for {skycell_name}, skipping")
            return False

        final_img_path, final_mask_path = _final_output_paths(config, skycell_name)
        if final_img_path.exists() and not config.overwrite:
            logger.info(f"Final exists, skipping: {final_img_path.name}")
            return True

        with fits.open(combined_img_path) as hdul:
            hdu_index = 1 if len(hdul) > 1 else 0
            combined_image = hdul[hdu_index].data.astype(np.float32)
            header = hdul[hdu_index].header
        wcs = WCS(header)

        combined_mask = None
        if combined_mask_path.exists():
            try:
                with fits.open(combined_mask_path) as mhdul:
                    mhdu_index = 1 if len(mhdul) > 1 else 0
                    combined_mask = mhdul[mhdu_index].data.astype(np.uint32)
            except Exception as exc:
                logger.warning(f"Could not load combined mask {combined_mask_path.name}: {exc}")

        # Determine mapping file and padding requirement
        sector4 = f"{config.sector:04d}"
        mapping_dir = Path(config.mapping_path)
        projection, cell = _extract_projection_and_cell_from_skycell_name(skycell_name)
        mapping_glob = f"tess_s{sector4}_{config.ccd_id}_skycell.{projection}.{cell}*.fits.gz"
        mapping_files = list(mapping_dir.glob(mapping_glob))

        if mapping_files:
            padding_req = check_tess_mapping_padding(mapping_files[0], config.pad_distance, config.edge_exclusion)
        else:
            logger.warning(
                f"No mapping file found for {skycell_name} in {mapping_dir} with pattern {mapping_glob}, assuming padding everywhere"
            )
            padding_req = PaddingRequirements(top=True, bottom=True, left=True, right=True,
                                             top_left=True, top_right=True, bottom_left=True, bottom_right=True)

        # Apply smart padding to combined image and mask if needed, using neighbor combined masks
        if padding_req.any_needed():
            padding_config = PaddingConfig(
                pad_distance=config.pad_distance,
                datapath=config.ps1_skycells_path,
                skycells_path=config.skycell_list_csv,
                combined_dir=str(Path(config.linear_comb_savepath)),
            )

            reference_filename = f"rings.v3.{skycell_name}.stk.r.unconv.fits"
            combined_image, combined_mask = smart_pad_ps1_image_and_mask(
                combined_image,
                combined_mask,
                wcs,
                reference_filename,
                "rizy",
                padding_config,
            )
        else:
            combined_image = np.pad(combined_image, config.pad_distance)
            if combined_mask is None:
                combined_mask = np.pad(combined_mask, config.pad_distance)
            # Add convolution-edge mask bit
            edge_thickness = psf.shape[0] // 2 + config.pad_distance
            if edge_thickness > 0:
                if combined_mask is None:
                    combined_mask = np.zeros_like(final_image, dtype=np.uint32)
                SYNDIFF_CONV_BAD = np.uint32(1 << 30)
                edge_bit = SYNDIFF_CONV_BAD
                combined_mask[:edge_thickness, :] |= edge_bit
                combined_mask[-edge_thickness:, :] |= edge_bit
                combined_mask[:, :edge_thickness] |= edge_bit
                combined_mask[:, -edge_thickness:] |= edge_bit


        # Convolve and save
        psf = create_gaussian_psf(size=config.psf_kernal_size, std=config.psf_std)
        final_image = fftconvolve(combined_image, psf, mode="same")

        final_header = create_output_header(header, config, conv=True)
        save_fits_image(final_image, final_header, final_img_path, config.overwrite)


        if combined_mask is not None:
            mask_header = final_header.copy()
            mask_header["MASK"] = (True, "PS1 bitmask used in combination")
            mask_header["MASKTYPE"] = ("PS1_BITMASK", "PS1 multi-bit binary flag system")
            mask_header["MASKFMT"] = ("UINT32", "Mask data format preserving all flags")
            save_fits_image(combined_mask.astype(np.uint32), mask_header, final_mask_path, config.overwrite)

        return True
    except Exception as exc:
        logger.error(f"Failed to pad+convolve {skycell_name}: {exc}")
        return False


def run_ps1_processing_pipeline(config: ProcessingConfig) -> None:
    logger.info("Starting PS1 processing pipeline")

    if config.verbose > 0:
        logging.getLogger().setLevel(logging.DEBUG)

    setup_output_directories(config)

    if not config.skycell_list_csv:
        raise ValueError("'skycell_list_csv' not resolved; provide sector+camera_id+ccd_id or set it explicitly.")
    skycell_names = _load_skycell_names(config.skycell_list_csv)
    if len(skycell_names) == 0:
        logger.warning("No skycells to process")
        return

    t0 = time.time()

    # Step 1: Combine all skycells (parallel)
    t1 = time.time()
    combine_ok = 0
    with ThreadPoolExecutor(max_workers=max(1, int(config.cores))) as executor:
        futures = [executor.submit(combine_one_skycell, config, name) for name in skycell_names]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Combining"):
            if fut.result():
                combine_ok += 1
    t2 = time.time()
    logger.info(f"Combine pass: {combine_ok}/{len(skycell_names)} in {t2 - t1:.1f}s")

    # Step 2: Pad and convolve (parallel)
    pad_ok = 0
    t3 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, int(config.cores))) as executor:
        futures = [executor.submit(pad_and_convolve_one_skycell, config, name) for name in skycell_names]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Pad+Convolve"):
            if fut.result():
                pad_ok += 1
    t4 = time.time()
    logger.info(f"Pad+Convolve pass: {pad_ok}/{len(skycell_names)} in {t4 - t3:.1f}s")

    logger.info(f"Total pipeline time: {t4 - t0:.1f}s")


if __name__ == "__main__":
    pipe_config = ProcessingConfig(
        sector=20,
        camera_id=3,
        ccd_id=3,
        psf_std=60.0,
        psf_kernal_size=2001,
        pad_distance=500,
        edge_exclusion=50,
        data_root="data",
        cores=10,
        overwrite=False,
        verbose=1,
        use_mask=True,
    )
    run_ps1_processing_pipeline(pipe_config)


