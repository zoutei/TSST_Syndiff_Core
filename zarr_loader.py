"""
Zarr loading utilities for PS1 skycell data.

Handles the hierarchical zarr structure and provides efficient loading
of individual cells and bands with thread-safe access.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr
from astropy.wcs import WCS

logger = logging.getLogger(__name__)


class ZarrManager:
    """Thread-safe zarr array manager for PS1 skycell data."""
    
    def __init__(self, zarr_root_path: str):
        """Initialize zarr manager.
        
        Args:
            zarr_root_path: Path to ps1_skycells.zarr root
        """
        self.zarr_root_path = Path(zarr_root_path)
        self.zarr_store = zarr.open(str(self.zarr_root_path), mode='r')
        self._lock = threading.Lock()
        self._projection_cache = {}
        
        logger.info(f"Initialized ZarrManager with {self.zarr_root_path}")
        
    def _get_projection_group(self, projection: str) -> zarr.Group:
        """Get zarr group for a specific projection with caching."""
        if projection not in self._projection_cache:
            with self._lock:
                if projection not in self._projection_cache:
                    # Direct projection access based on actual structure
                    self._projection_cache[projection] = self.zarr_store[projection]
        return self._projection_cache[projection]
    
    def load_skycell_band(self, skycell_name: str, band: str, is_mask: bool = False) -> np.ndarray:
        """Load a single band from a skycell.
        
        Args:
            skycell_name: Full skycell name (e.g., "skycell.2556.080")
            band: Band name ("r", "i", "z", "y")
            is_mask: Whether to load mask data
            
        Returns:
            Numpy array with band data
        """
        try:
            # Parse skycell name to get projection and cell
            parts = skycell_name.split('.')
            if len(parts) != 3 or parts[0] != 'skycell':
                raise ValueError(f"Invalid skycell name format: {skycell_name}")
            
            projection = parts[1]
            # Note: cell variable removed as it's not used
            
            # Get projection group
            proj_group = self._get_projection_group(projection)
            
            # Build path to band data
            band_suffix = "_mask" if is_mask else ""
            band_key = f"{band}{band_suffix}"
            
            # Load data
            data = proj_group[skycell_name][band_key][:]
            
            logger.debug(f"Loaded {skycell_name}/{band_key}: shape={data.shape}, dtype={data.dtype}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load {skycell_name}/{band}: {e}")
            raise
    
    def load_skycell_all_bands(self, skycell_name: str, bands: list[str] = None) -> dict[str, np.ndarray]:
        """Load all bands for a skycell.
        
        Args:
            skycell_name: Full skycell name
            bands: List of bands to load (default: ["r", "i", "z", "y"])
            
        Returns:
            Dictionary mapping band names to numpy arrays
        """
        if bands is None:
            bands = ["r", "i", "z", "y"]
        
        results = {}
        for band in bands:
            results[band] = self.load_skycell_band(skycell_name, band, is_mask=False)
        
        return results
    
    def load_skycell_all_masks(self, skycell_name: str, bands: list[str] = None) -> dict[str, np.ndarray]:
        """Load all mask bands for a skycell.
        
        Args:
            skycell_name: Full skycell name
            bands: List of bands to load (default: ["r", "i", "z", "y"])
            
        Returns:
            Dictionary mapping band names to mask arrays
        """
        if bands is None:
            bands = ["r", "i", "z", "y"]
        
        results = {}
        for band in bands:
            try:
                results[band] = self.load_skycell_band(skycell_name, band, is_mask=True)
            except Exception as e:
                logger.warning(f"Failed to load mask {skycell_name}/{band}: {e}")
                # Create dummy mask if not available
                image_data = self.load_skycell_band(skycell_name, band, is_mask=False)
                results[band] = np.zeros_like(image_data, dtype=np.uint32)
        
        return results
    
    def get_skycell_wcs(self, skycell_name: str, reference_band: str = "r") -> WCS:
        """Get WCS for a skycell from zarr metadata or derive from reference.
        
        For now, creates a dummy WCS. In production, this should extract
        WCS from zarr metadata or load from a reference FITS file.
        
        Args:
            skycell_name: Full skycell name
            reference_band: Band to use for WCS reference
            
        Returns:
            WCS object
        """
        # TODO: Extract WCS from zarr metadata when available
        # For now, create a basic WCS structure
        
        # Load a sample array to get dimensions
        sample_data = self.load_skycell_band(skycell_name, reference_band)
        ny, nx = sample_data.shape
        
        # Create basic WCS (this should be replaced with actual WCS from metadata)
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [nx/2, ny/2]
        wcs.wcs.crval = [200.0, 70.0]  # Dummy coordinates
        wcs.wcs.cdelt = [6.9444446e-05, 6.9444446e-05]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.array_shape = (ny, nx)
        
        logger.warning(f"Using dummy WCS for {skycell_name} - should be replaced with actual metadata")
        return wcs
    
    def batch_load_cells(self, skycell_names: list[str], bands: list[str] = None) -> dict[str, dict[str, np.ndarray]]:
        """Load multiple skycells in parallel.
        
        Args:
            skycell_names: List of skycell names to load
            bands: List of bands to load
            
        Returns:
            Nested dict: {skycell_name: {band: array}}
        """
        if bands is None:
            bands = ["r", "i", "z", "y"]
        
        results = {}
        
        def load_one_cell(skycell_name):
            return skycell_name, self.load_skycell_all_bands(skycell_name, bands)
        
        # Use ThreadPoolExecutor for parallel loading
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(load_one_cell, name) for name in skycell_names]
            for future in futures:
                cell_name, cell_data = future.result()
                results[cell_name] = cell_data
        
        return results
    
    def batch_load_masks(self, skycell_names: list[str], bands: list[str] = None) -> dict[str, dict[str, np.ndarray]]:
        """Load masks for multiple skycells in parallel.
        
        Args:
            skycell_names: List of skycell names to load
            bands: List of bands to load
            
        Returns:
            Nested dict: {skycell_name: {band: mask_array}}
        """
        if bands is None:
            bands = ["r", "i", "z", "y"]
        
        results = {}
        
        def load_one_mask(skycell_name):
            return skycell_name, self.load_skycell_all_masks(skycell_name, bands)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(load_one_mask, name) for name in skycell_names]
            for future in futures:
                cell_name, cell_masks = future.result()
                results[cell_name] = cell_masks
        
        return results
    
    def list_available_skycells(self, projection: str = None) -> list[str]:
        """List all available skycells, optionally filtered by projection.
        
        Args:
            projection: Optional projection filter
            
        Returns:
            List of skycell names
        """
        skycells = []
        
        if projection:
            projections = [projection]
        else:
            # Get all projections from zarr store
            projections = list(self.zarr_store.group_keys())
        
        for proj in projections:
            try:
                proj_group = self._get_projection_group(proj)
                for skycell_name in proj_group.group_keys():
                    skycells.append(skycell_name)
            except Exception as e:
                logger.warning(f"Failed to list skycells for projection {proj}: {e}")
        
        return sorted(skycells)


def create_zarr_output_structure(output_path: str, n_cells: int, cell_height: int, cell_width: int) -> tuple[zarr.Array, zarr.Array]:
    """Create zarr arrays for convolved output.
    
    Args:
        output_path: Path to output zarr directory
        n_cells: Number of cells to store
        cell_height: Height of each cell
        cell_width: Width of each cell
        
    Returns:
        Tuple of (images_array, masks_array)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create zarr store
    store = zarr.open(str(output_path), mode='w')
    
    # Create arrays for convolved images and masks
    images = store.create_dataset(
        'convolved_images',
        shape=(n_cells, cell_height, cell_width),
        dtype=np.float32,
        chunks=(1, cell_height, cell_width),
        compression='blosc'
    )
    
    masks = store.create_dataset(
        'convolved_masks', 
        shape=(n_cells, cell_height, cell_width),
        dtype=np.uint32,
        chunks=(1, cell_height, cell_width),
        compression='blosc'
    )
    
    logger.info(f"Created output zarr structure at {output_path}")
    logger.info(f"Images shape: {images.shape}, Masks shape: {masks.shape}")
    
    return images, masks
