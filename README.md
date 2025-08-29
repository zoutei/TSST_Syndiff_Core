# PS1 Image Processing Pipeline

Modernized and canonical implementation for combining PS1 images, analyzing TESS pixel mappings, applying smart padding, and producing final convolved outputs.

## Quick Start

Run the pipeline by specifying sector and CCD. All paths are auto-derived.

```bash
python pipeline.py --sector 20 --ccd-id 11 -v
```

This resolves the following default paths (override with flags if needed):
- `datapath`: `data/ps1_skycells`
- `mapping_path`: `data/mapping_output/sector_0020/ccd_11`
- `skycells_path`: `data/mapping_output/sector_0020/ccd_11/tess_s0020_11_master_skycells_list.csv`
- `catalog_path`: `data/Sector_20_star_cat`
- `savepath`: `data/tess_comb_skycells/s0020_ccd11`

## What it does
- Loads PS1 r, i, z, y bands and linearly combines them
- Saves the combined image
- Analyzes TESS pixel mapping to determine where padding is needed
- Applies smart padding via skycell reprojection only where needed
- Convolves with a Gaussian PSF and writes final science image and mask

## Configuration

In-code configuration is defined by `process_ps1.ProcessingConfig`. Prefer setting `sector` and `ccd_id`, which derive all other paths. You can override any path explicitly.

Key options:
- `sector` (int), `ccd_id` (int): Primary inputs
- `data_root` (str): Root directory for data (default `data`)
- `combine_weights` (list[float]): Linear combination weights for r, i, z, y
- `pad_distance` (int): Edge distance for padding decision
- `psf_std` (float): Gaussian PSF sigma for convolution
- `cores` (int), `overwrite` (bool), `verbose` (int)

## Outputs
Inside `savepath`:
- `combined/` — pre-convolution combined images
- `final_conv/` — convolved images and masks

## Notes
- Saturation correction hooks are in place but currently disabled by default.
- Legacy code remains under `old/` for reference; the upgraded variants have been consolidated and renamed as the main implementation.


