# Process PS1 - Sliding Window Pipeline Documentation

## Overview

The `process_ps1.py` module implements a high-performance, memory-efficient PS1 (PanSTARRS1) image processing pipeline using a sliding window approach. This pipeline processes PS1 skycell data in parallel workers feeding into a sequential assembler that manages the sliding window state correctly.

## Architecture

The pipeline uses a **four-stage architecture** designed for maximum throughput while maintaining correct sliding window semantics:

```
Stage 1: Reader Workers (Parallel)
    ↓
Stage 2: Pre-processor Workers (Parallel)  
    ↓
Stage 3: Sequential Assembler (Single Thread)
    ↓
Stage 4: Saver Worker (Single Thread)
```

### Why This Architecture?

1. **Parallel Upstream Workers**: Data loading and preprocessing can be parallelized safely
2. **Sequential Assembler**: The sliding window state must be managed sequentially to ensure correct row-by-row processing
3. **Single Saver**: File I/O is serialized to avoid conflicts

## Key Concepts

### Sliding Window Processing

The pipeline processes PS1 data using a "sliding window" approach where:

- **Current Array**: Contains the row currently being processed and convolved
- **Next Array**: Contains the next row, used for padding the current row
- **Window Advancement**: After processing, next becomes current, and a new next row is loaded

### Master Arrays

Master arrays are large memory buffers sized to hold entire rows of skycells:

```python
Width = (max_cells_in_projection × cell_width) + (2 × PAD_SIZE)
Height = cell_height + (2 × PAD_SIZE)
```

### Cell Overlap Handling

Adjacent PS1 skycells naturally overlap by 480 pixels. The pipeline:

1. Places first cell normally at position 0 (accounting for padding)
2. Places subsequent cells with 10-pixel edge exclusion
3. Tracks cell positions to correctly extract non-overlapping regions

## Worker Stages

### Stage 1: Reader Workers

**Function**: `reader_worker(task_queue, raw_cell_queue, zarr_store)`

**Purpose**: Load raw PS1 skycell data from Zarr storage

**Input**: Task queue containing `(projection, skycell_name, row_id)` tuples

**Output**: Raw cell bundles with data and metadata

**Key Operations**:
- Loads image and mask data from Zarr store
- Extracts WCS information and headers
- Packages data for downstream processing
- Handles both "skycell.XXX.YYY" and "rings.v3.skycell.XXX.YYY" naming conventions

**Parallelization**: Multiple reader workers run in parallel, each processing different tasks

### Stage 2: Pre-processor Workers  

**Function**: `pre_processor_worker(raw_cell_queue, combined_cell_queue)`

**Purpose**: Combine PS1 bands and prepare data for assembly

**Input**: Raw cell bundles from reader workers

**Output**: Combined cell bundles ready for assembly

**Key Operations**:
- Band combination using `process_skycell_bands()` from `band_utils.py`
- Applies band weights and processing
- Preserves all metadata and spatial information

**Parallelization**: Multiple pre-processor workers run in parallel

### Stage 3: Sequential Assembler

**Function**: `sequential_processor(projections, df, combined_cell_queue, results_queue, psf_sigma, zarr_path)`

**Purpose**: Assemble rows and manage sliding window state

**Input**: Combined cell bundles and projection metadata

**Output**: Convolved results for each row

**Key Operations**:
1. **Row Assembly**: Gathers all cells for a row from the queue
2. **Sliding Window Management**: Maintains current and next arrays
3. **Cross-Row Padding**: Uses next row cells to pad current row where possible
4. **Remaining Padding**: Loads additional cells for complete padding
5. **Convolution**: Applies Gaussian PSF convolution
6. **Cell Extraction**: Extracts individual cell results with correct overlap handling

**Sequential Nature**: Must run in a single thread to maintain sliding window state consistency

### Stage 4: Saver Worker

**Function**: `saver_worker(results_queue, output_path)`

**Purpose**: Save convolved results to disk

**Input**: Convolved cell results from assembler

**Output**: Zarr files and metadata on disk

**Key Operations**:
- Writes image and mask data to Zarr store
- Saves cell metadata as JSON
- Handles file locking for thread safety

## Data Structures

### MasterArrayConfig

Configuration for master array dimensions and cell placement:

```python
@dataclass
class MasterArrayConfig:
    width: int           # Total width including padding
    height: int          # Total height including padding  
    cell_width: int      # Individual cell width
    cell_height: int     # Individual cell height
    pad_size: int        # Padding size (480 pixels)
    max_cells_per_row: int # Maximum cells in any row
```

### ProcessingState

Current state of the sliding window:

```python
@dataclass
class ProcessingState:
    current_array: np.ndarray    # Array being processed
    next_array: np.ndarray       # Array for next row
    current_masks: dict          # Masks for current row cells
    next_masks: dict             # Masks for next row cells
```

## I/O Methods

### Input Data Sources

1. **CSV Metadata**: Projection and cell information from `find_csv_file()` and `load_csv_data()`
2. **Zarr Storage**: PS1 skycell data loaded via `load_skycell_bands_masks_and_headers()`
3. **WCS Information**: Coordinate system data extracted from PS1 headers

### Output Data Formats

1. **Zarr Arrays**: Efficient storage for convolved images and masks
2. **JSON Metadata**: Cell positioning and processing information
3. **Directory Structure**: Organized by sector/camera/ccd hierarchy

### File Organization

```
data/
└── convolved_results/
    └── sector_XXXX/
        └── camera_X/
            └── ccd_X/
                ├── convolved_images.zarr    # Main data store
                └── cell_metadata.json       # Processing metadata
```

## Memory Management

### Efficient Buffer Usage

- **Master Arrays**: Pre-allocated to maximum required size
- **Cell Buffers**: Reused across rows to minimize allocations
- **Zarr Chunking**: Data stored in optimized chunks for streaming access

### Padding Strategy

1. **Cross-Row Optimization**: Use already-loaded next row cells for current row padding
2. **On-Demand Loading**: Only load additional padding cells when needed
3. **Memory Cleanup**: Clear processed data promptly to manage memory usage

## Error Handling and Robustness

### Timeout Management

- **Gather Timeout**: 5-minute timeout for gathering row cells
- **Queue Timeouts**: Reasonable timeouts on queue operations
- **Graceful Degradation**: Continues processing when individual cells fail

### Signal Handling

- **SIGINT/SIGTERM**: Graceful shutdown of worker processes
- **Process Cleanup**: Terminates child processes on shutdown
- **Resource Cleanup**: Closes file handles and queues properly

## Performance Characteristics

### Throughput Optimization

- **Parallel I/O**: Multiple workers reading from Zarr simultaneously
- **Concurrent Processing**: Band combination happens in parallel
- **Streaming Pipeline**: Data flows through stages without blocking

### Memory Efficiency

- **Fixed Memory Usage**: Master arrays sized once, reused throughout
- **Minimal Copying**: Data passed by reference where possible
- **Chunked Processing**: Large datasets processed in manageable pieces

## Configuration Parameters

### Key Constants

```python
CELL_OVERLAP = 480           # Natural overlap between cells
EDGE_EXCLUSION = 10          # Pixels overwritten for blending
EFFECTIVE_OVERLAP = 470      # Remaining overlap after exclusion
PAD_SIZE = 480              # Padding around arrays
GATHER_TIMEOUT_SECONDS = 300 # Row gathering timeout
```

### Processing Parameters

- **PSF Sigma**: Gaussian convolution kernel width (default: 60.0)
- **Projections Limit**: Optional limit on number of projections to process
- **Data Root**: Base directory for input and output data

## Usage Example

```python
from process_ps1 import run_modern_sliding_window_pipeline

# Process PS1 data for TESS sector 20, camera 3, CCD 3
run_modern_sliding_window_pipeline(
    sector=20,
    camera=3, 
    ccd=3,
    data_root="data",
    projections_limit=None,  # Process all projections
    psf_sigma=60.0          # Default PSF sigma
)
```

## Dependencies

### Core Libraries
- **numpy**: Array operations and mathematical functions
- **pandas**: CSV data handling and manipulation
- **zarr**: Efficient array storage and retrieval
- **astropy**: FITS file handling and WCS operations
- **scipy**: Convolution and signal processing

### Internal Modules
- **band_utils**: PS1 band combination and processing
- **csv_utils**: CSV file reading and parsing utilities
- **zarr_utils**: Zarr data loading and management
- **convolution_utils**: PSF convolution operations

## Monitoring and Logging

### Progress Tracking
- Row-by-row processing progress
- Worker queue status monitoring  
- Memory usage tracking

### Error Reporting
- Failed cell loading notifications
- Timeout and queue overflow warnings
- Processing statistics and timing

This architecture provides a robust, scalable solution for processing large PS1 datasets while maintaining the correct sliding window semantics required for proper image padding and convolution.
