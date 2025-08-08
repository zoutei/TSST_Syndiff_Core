"""
PS1 Image Processing Pipeline (modernized, canonical)

Workflow:
- Load and linearly combine 4 PS1 filters (r, i, z, y)
- Optional saturation correction (currently disabled; see notes)
- Save intermediate combined image
- Analyze TESS pixel mapping to determine smart padding requirements
- Apply padding only where needed using smart reprojection
- Convolve with PSF and save final results (and masks)

Primary user inputs: sector and ccd_id. Paths are auto-derived from these.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
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

from correct_saturation import saturated_stars  # noqa: F401 (kept for production toggle)
from smart_padding import smart_pad_ps1_image, PaddingConfig
from tools import _save_space


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class ProcessingConfig:
    """Configuration for PS1 processing pipeline.

    Prefer setting `sector` and `ccd_id`; paths will be derived automatically.
    You may override any derived paths explicitly.
    """

    # Canonical high-level inputs
    sector: Optional[int] = None
    ccd_id: Optional[int] = None
    data_root: str = "data"

    # Data paths (auto-derived if not supplied)
    ps1_skycells_path: Optional[str] = None  # alias of datapath
    datapath: Optional[str] = None
    savepath: Optional[str] = None
    catalog_path: Optional[str] = None  # directory containing per-skycell csv files
    mapping_path: Optional[str] = None  # directory containing mapping FITS
    skycells_path: Optional[str] = None  # CSV listing skycells for padding
    smart_padding_skycells: Optional[str] = None  # alias of skycells_path

    # Processing parameters
    psf_std: float = 70.0
    combine_weights: Optional[List[float]] = None  # r, i, z, y
    pad_distance: int = 500
    edge_exclusion: int = 10
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

        # Normalize datapath alias
        if self.ps1_skycells_path is not None and self.datapath is None:
            self.datapath = self.ps1_skycells_path

        # Derive paths from sector/ccd if provided
        if self.sector is not None and self.ccd_id is not None:
            sector4 = f"{self.sector:04d}"
            ccd2 = f"{self.ccd_id:02d}"

            if self.datapath is None:
                self.datapath = str(Path(self.data_root) / "ps1_skycells")

            if self.mapping_path is None:
                self.mapping_path = str(Path(self.data_root) / "mapping_output" / f"sector_{sector4}" / f"ccd_{ccd2}")

            if self.skycells_path is None:
                self.skycells_path = str(Path(self.mapping_path) / f"tess_s{sector4}_{ccd2}_master_skycells_list.csv")

            if self.catalog_path is None:
                # Directory containing per-skycell CSVs
                self.catalog_path = str(Path(self.data_root) / f"Sector_{self.sector}_star_cat")

            if self.savepath is None:
                self.savepath = str(Path(self.data_root) / "tess_comb_skycells" / f"s{sector4}_ccd{ccd2}")

        # Fallbacks if still not set
        if self.datapath is None:
            self.datapath = str(Path(self.data_root) / "ps1_skycells")
        if self.savepath is None:
            self.savepath = str(Path(self.data_root) / "tess_comb_skycells")

        # Default smart padding CSV path alias
        if self.smart_padding_skycells is None:
            self.smart_padding_skycells = self.skycells_path


@dataclass
class PaddingRequirements:
    top: bool = False
    bottom: bool = False
    left: bool = False
    right: bool = False
    top_left: bool = False
    top_right: bool = False
    bottom_left: bool = False
    bottom_right: bool = False

    def any_needed(self) -> bool:
        return any([self.top, self.bottom, self.left, self.right,
                    self.top_left, self.top_right, self.bottom_left, self.bottom_right])


@dataclass
class ProcessedData:
    combined_image: np.ndarray
    combined_mask: Optional[np.ndarray]
    header: fits.Header
    wcs: WCS


def _resolve_skycells_name_column(skycells: pd.DataFrame) -> str:
    cols = {c.upper(): c for c in skycells.columns}
    if "NAME" in cols:
        return cols["NAME"]
    if "Name" in skycells.columns:
        return "Name"
    raise KeyError("Skycells CSV must contain a 'NAME' or 'Name' column")


def load_and_combine_ps1_bands(
    file_pattern: str,
    bands: List[str],
    weights: List[float],
    use_mask: bool = True,
    smart_padding: bool = False,
    padding_config: Optional[PaddingConfig] = None,
) -> ProcessedData:
    combined_image = None
    combined_mask = None
    reference_header = None
    reference_wcs = None

    weights_array = np.array(weights)

    for i, (band, weight) in enumerate(zip(bands, weights_array)):
        filepath = f"{file_pattern}{band}.unconv.fits"

        with fits.open(filepath) as hdul:
            hdu_index = 1 if len(hdul) > 1 else 0
            raw_data = hdul[hdu_index].data.astype(np.float32)
            header = hdul[hdu_index].header

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
            reference_wcs = WCS(header)
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

    if smart_padding and padding_config:
        reference_filepath = f"{file_pattern}{bands[0]}.unconv.fits"
        combined_image = smart_pad_ps1_image(
            combined_image, reference_wcs, reference_filepath, "rizy", padding_config
        )
        if combined_mask is not None:
            pad_amount = padding_config.pad_distance
            combined_mask = np.pad(combined_mask, pad_amount, constant_values=0)

    return ProcessedData(
        combined_image=combined_image,
        combined_mask=combined_mask,
        header=reference_header,
        wcs=reference_wcs,
    )


def save_fits_image(data: np.ndarray, header: fits.Header, filepath: Path, overwrite: bool = False) -> None:
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdul = fits.HDUList([hdu])
    filepath.parent.mkdir(parents=True, exist_ok=True)
    hdul.writeto(filepath, overwrite=overwrite)


def create_gaussian_psf(size: int = 2000, std: float = 70.0) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    x = x - size / 2
    y = y - size / 2
    psf = models.Gaussian2D(x_stddev=std, y_stddev=std)(x, y)
    return psf / np.nansum(psf)


def check_tess_mapping_padding(mapping_file: Path, pad_distance: int = 500, edge_exclusion: int = 10) -> PaddingRequirements:
    try:
        with fits.open(mapping_file) as hdul:
            mapping_data = hdul[1].data if hdul[1].data is not None else hdul[2].data

        padding = PaddingRequirements()
        # Top
        padding.top = np.any(mapping_data[edge_exclusion:pad_distance + edge_exclusion, edge_exclusion:-edge_exclusion] != 0)
        # Bottom
        padding.bottom = np.any(mapping_data[-(pad_distance + edge_exclusion):-edge_exclusion, edge_exclusion:-edge_exclusion] != 0)
        # Left
        padding.left = np.any(mapping_data[edge_exclusion:-edge_exclusion, edge_exclusion:pad_distance + edge_exclusion] != 0)
        # Right
        padding.right = np.any(mapping_data[edge_exclusion:-edge_exclusion, -(pad_distance + edge_exclusion):-edge_exclusion] != 0)
        # Corners
        padding.top_left = np.any(mapping_data[edge_exclusion:pad_distance + edge_exclusion, edge_exclusion:pad_distance + edge_exclusion] != 0)
        padding.top_right = np.any(mapping_data[edge_exclusion:pad_distance + edge_exclusion, -(pad_distance + edge_exclusion):-edge_exclusion] != 0)
        padding.bottom_left = np.any(mapping_data[-(pad_distance + edge_exclusion):-edge_exclusion, edge_exclusion:pad_distance + edge_exclusion] != 0)
        padding.bottom_right = np.any(mapping_data[-(pad_distance + edge_exclusion):-edge_exclusion, -(pad_distance + edge_exclusion):-edge_exclusion] != 0)
        return padding
    except Exception as exc:
        logger.warning(f"Error analyzing mapping file {mapping_file}: {exc}")
        return PaddingRequirements(top=True, bottom=True, left=True, right=True,
                                   top_left=True, top_right=True, bottom_left=True, bottom_right=True)


def create_output_header(header: fits.Header, config: ProcessingConfig) -> fits.Header:
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
    output_header["PSFTYPE"] = ("Gaussian", "Type of PSF used in convolution")
    output_header["PSFSTD"] = (config.psf_std, "Standard deviation of Gaussian PSF")
    weights = np.array(config.combine_weights)
    output_header["FRACR"] = (weights[0], "Fraction of r band used")
    output_header["FRACI"] = (weights[1], "Fraction of i band used")
    output_header["FRACZ"] = (weights[2], "Fraction of z band used")
    output_header["FRACY"] = (weights[3], "Fraction of y band used")
    return output_header


def create_ps1_compatible_object(processed_data: ProcessedData) -> object:
    class PS1Compatible:
        def __init__(self, data, mask, header, wcs, band="rizy"):
            self.data = data
            self.padded = data
            self.mask = mask
            self.header = header
            self.wcs = wcs
            self.band = band

    return PS1Compatible(
        data=processed_data.combined_image,
        mask=processed_data.combined_mask,
        header=processed_data.header,
        wcs=processed_data.wcs,
        band="rizy",
    )


def process_single_field(
    file_pattern: str,
    skycell_id: str,
    config: ProcessingConfig,
    skycells: pd.DataFrame,
    catalog: pd.DataFrame,
) -> bool:
    try:
        base_name = Path(file_pattern).name
        final_output = Path(config.savepath) / "final_conv" / f"{base_name}{config.suffix}.fits"

        if final_output.exists() and not config.overwrite:
            logger.info(f"Skipping {base_name} - already processed")
            return True

        bands = ["r", "i", "z", "y"]

        name_parts = base_name.split(".")
        projection, cell = name_parts[3], name_parts[4]

        processed_data = load_and_combine_ps1_bands(
            file_pattern, bands, config.combine_weights, use_mask=config.use_mask
        )

        header = create_output_header(processed_data.header, config)

        sat_corrected_ps1 = create_ps1_compatible_object(processed_data)

        combined_output_dir = Path(config.savepath) / "combined"
        combined_output_path = combined_output_dir / f"{base_name}rizy.unconv_combined.fits"
        save_fits_image(sat_corrected_ps1.padded, header, combined_output_path, config.overwrite)

        # Derive mapping filename pattern dynamically
        sector4 = f"{config.sector:04d}" if config.sector is not None else "*"
        ccd2 = f"{config.ccd_id:02d}" if config.ccd_id is not None else "*"
        mapping_glob = f"TESS_s{sector4}_{ccd2}_skycell.{projection}.{cell}_*.fits.gz"

        mapping_dir = Path(config.mapping_path) if config.mapping_path else Path(config.data_root) / "mapping_output"
        mapping_files = list(mapping_dir.glob(mapping_glob))

        if mapping_files:
            padding_req = check_tess_mapping_padding(mapping_files[0], config.pad_distance, config.edge_exclusion)
        else:
            logger.warning(f"No mapping file found for {base_name} in {mapping_dir} with pattern {mapping_glob}, assuming padding needed everywhere")
            padding_req = PaddingRequirements(top=True, bottom=True, left=True, right=True,
                                             top_left=True, top_right=True, bottom_left=True, bottom_right=True)

        if padding_req.any_needed():
            padding_config = PaddingConfig(
                pad_distance=config.pad_distance,
                datapath=config.datapath,
                skycells_path=config.smart_padding_skycells,
            )

            # Apply smart padding directly to the already combined image (single-pass)
            reference_filepath = f"{file_pattern}{bands[0]}.unconv.fits"
            padded_image = smart_pad_ps1_image(
                sat_corrected_ps1.padded,
                sat_corrected_ps1.wcs,
                reference_filepath,
                "rizy",
                padding_config,
            )
            sat_corrected_ps1.padded = padded_image

            if sat_corrected_ps1.mask is not None:
                pad_amount = padding_config.pad_distance
                sat_corrected_ps1.mask = np.pad(sat_corrected_ps1.mask, pad_amount, constant_values=0)

            final_ps1_obj = sat_corrected_ps1
        else:
            final_ps1_obj = sat_corrected_ps1

        psf = create_gaussian_psf(size=2000, std=config.psf_std)
        final_image = fftconvolve(final_ps1_obj.padded, psf, mode="same")

        final_header = create_output_header(final_ps1_obj.header, config)
        mask_header = final_header.copy()
        mask_header["MASK"] = (True, "PS1 bitmask used in combination")
        mask_header["MASKTYPE"] = ("PS1_BITMASK", "PS1 multi-bit binary flag system")
        mask_header["MASKFMT"] = ("UINT32", "Mask data format preserving all flags")

        final_conv_dir = Path(config.savepath) / "final_conv"
        save_fits_image(final_image, final_header, final_conv_dir / f"{base_name}{config.suffix}.fits", config.overwrite)

        if final_ps1_obj.mask is not None:
            save_fits_image(final_ps1_obj.mask.astype(np.uint32), mask_header,
                            final_conv_dir / f"{base_name}{config.suffix}.mask.fits", config.overwrite)

        logger.info(f"Successfully processed {base_name}")
        return True
    except Exception as exc:
        logger.error(f"Error processing {file_pattern}: {exc}")
        return False


def discover_ps1_fields(config: ProcessingConfig, skycells: pd.DataFrame) -> List[Tuple[str, str]]:
    logger.info("Discovering PS1 fields...")

    files = list(Path(config.datapath).rglob("*.unconv.fits"))
    field_patterns: Dict[str, Dict[str, object]] = {}

    name_col = _resolve_skycells_name_column(skycells)
    csv_skycells = set(skycells[name_col].astype(str).str.split("skycell.").str[-1])

    for file_path in files:
        file_str = str(file_path)
        if ".stk." in file_str and ".unconv.fits" in file_str:
            parts = file_str.split(".stk.")
            if len(parts) >= 2:
                base_pattern = parts[0] + ".stk."

            filename = file_path.name
            if "skycell." in filename:
                skycell_id = filename.split("skycell.")[1].split(".stk")[0]

                if skycell_id in csv_skycells:
                    if base_pattern not in field_patterns:
                        field_patterns[base_pattern] = {"count": 0, "skycell_id": skycell_id}
                    field_patterns[base_pattern]["count"] += 1

    complete_fields = [
        (pattern, info["skycell_id"]) for pattern, info in field_patterns.items() if info["count"] >= 4
    ]

    # Skip already processed if not overwriting
    if not config.overwrite:
        filtered_fields: List[Tuple[str, str]] = []
        for pattern, skycell_id in complete_fields:
            base_name = Path(pattern).name
            final_output = Path(config.savepath) / "final_conv" / f"{base_name}{config.suffix}.fits"
            if not final_output.exists():
                filtered_fields.append((pattern, skycell_id))
        complete_fields = filtered_fields

    logger.info(f"Found {len(complete_fields)} fields to process")
    return complete_fields


def setup_output_directories(config: ProcessingConfig) -> None:
    for d in [Path(config.savepath), Path(config.savepath) / "combined", Path(config.savepath) / "final_conv"]:
        d.mkdir(parents=True, exist_ok=True)


def run_ps1_processing_pipeline(config: ProcessingConfig) -> None:
    logger.info("Starting PS1 processing pipeline")

    if config.verbose > 0:
        logging.getLogger().setLevel(logging.DEBUG)

    setup_output_directories(config)

    if not config.skycells_path:
        raise ValueError("'skycells_path' not resolved; provide sector+ccd_id or set it explicitly.")
    skycells = pd.read_csv(config.skycells_path)

    fields_to_process = discover_ps1_fields(config, skycells)
    if not fields_to_process:
        logger.warning("No fields found to process")
        return

    success_count = 0

    for file_pattern, skycell_id in tqdm(fields_to_process, desc="Processing fields"):
        name_parts = Path(file_pattern).name.split(".")
        projection, cell = name_parts[3], name_parts[4]

        # Catalog path per field
        sector = config.sector if config.sector is not None else 0
        ccd = config.ccd_id if config.ccd_id is not None else 0
        catalog_dir = Path(config.catalog_path) if config.catalog_path else Path(config.data_root)
        # Accept both legacy and new naming conventions
        candidate_paths = [
            catalog_dir / f"Sector{sector}_ccd{ccd}_skycell.{projection}.{cell}_ps1.csv",
            catalog_dir / f"Sector_{sector}_star_cat" / f"Sector{sector}_ccd{ccd}_skycell.{projection}.{cell}_ps1.csv",
        ]
        field_catalog_path = next((p for p in candidate_paths if p.exists()), candidate_paths[0])

        try:
            if field_catalog_path.exists():
                catalog = pd.read_csv(field_catalog_path)
            else:
                logger.warning(f"Catalog not found: {field_catalog_path}, using dummy catalog (testing)")
                catalog = pd.DataFrame({"ra": [0], "dec": [0]})

            if process_single_field(file_pattern, skycell_id, config, skycells, catalog):
                success_count += 1
        except Exception as exc:
            logger.error(f"Failed to process {file_pattern}: {exc}")

    logger.info(f"Processing complete: {success_count}/{len(fields_to_process)} fields processed successfully")


def run_combine_only(config: ProcessingConfig, limit: Optional[int] = None) -> None:
    """Combine PS1 bands for available fields and save combined images only.

    - No smart padding
    - No convolution
    - Saves combined image (and mask if available) under savepath/combined
    """
    logger.info("Starting combine-only run (no padding, no convolution)")
    if config.verbose > 0:
        logging.getLogger().setLevel(logging.DEBUG)

    setup_output_directories(config)

    if not config.skycells_path:
        raise ValueError("'skycells_path' not resolved; provide sector+ccd_id or set it explicitly.")
    skycells = pd.read_csv(config.skycells_path)

    fields_to_process = discover_ps1_fields(config, skycells)
    if limit is not None:
        fields_to_process = fields_to_process[:limit]
    if not fields_to_process:
        logger.warning("No fields found to process for combine-only")
        return

    bands = ["r", "i", "z", "y"]
    combined_output_dir = Path(config.savepath) / "combined"
    combined_output_dir.mkdir(parents=True, exist_ok=True)

    for file_pattern, skycell_id in tqdm(fields_to_process, desc="Combining fields"):
        base_name = Path(file_pattern).name
        output_path = combined_output_dir / f"{base_name}rizy.unconv_combined.fits"
        if output_path.exists() and not config.overwrite:
            logger.info(f"Skipping {base_name} - combined file exists")
            continue

        try:
            processed = load_and_combine_ps1_bands(
                file_pattern, bands, config.combine_weights, use_mask=config.use_mask
            )
            header = create_output_header(processed.header, config)
            save_fits_image(processed.combined_image, header, output_path, config.overwrite)
            if processed.combined_mask is not None:
                mask_header = header.copy()
                mask_header['MASK'] = (True, 'PS1 bitmask used in combination')
                save_fits_image(processed.combined_mask.astype(np.uint32), mask_header, output_path.with_suffix('.mask.fits'), config.overwrite)
            logger.info(f"Combined {base_name}")
        except Exception as exc:
            logger.error(f"Failed to combine {file_pattern}: {exc}")


if __name__ == "__main__":
    # Simple local test: combine-only for a small subset
    test_config = ProcessingConfig(
        sector=20,
        ccd_id=11,
        data_root="data",
        cores=1,
        overwrite=True,
        verbose=1,
        use_mask=True,
    )
    run_combine_only(test_config, limit=2)


