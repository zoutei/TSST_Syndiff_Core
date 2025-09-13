"""
Producer-consumer pipeline for processing PS1 skycells.

- Producer (Threads): Loads raw band data from Zarr store (I/O-bound).
- Consumer (Processes): Applies flux conversion to raw data (CPU-bound).
"""

import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Manager

# Assumes zarr_utils and band_combination are in the same package
import band_utils
import zarr_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _producer_loader(tasks_queue, processed_queue, zarr_path):
    """
    Producer function: loads data using threads.
    Pulls a task (projection, skycell) from the tasks_queue, loads its data,
    and puts the raw data into the processed_queue for the consumers.
    """
    while True:
        try:
            projection, skycell = tasks_queue.get_nowait()
        except Exception:
            break  # No more tasks

        try:
            logger.info(f"[Producer] Loading data for {projection}/{skycell}...")
            # Use the parallel loader for a single skycell's bands/masks/headers
            bands, masks, headers = zarr_utils.load_skycell_bands_masks_and_headers(zarr_path, projection, skycell)
            if bands:
                # Put the loaded raw data onto the queue for consumers (include headers)
                processed_queue.put(("data", projection, skycell, bands, headers))
            if masks:
                # Masks are optional, so only add if they exist
                processed_queue.put(("mask", projection, skycell, masks))
        except KeyError:
            # This is a common, expected error if a skycell doesn't exist.
            # Log it as a warning instead of an error to reduce noise.
            logger.warning(f"[Loader] KeyError: Skycell '{skycell}' not found in projection '{projection}'. Skipping task.")
        except Exception as e:
            # Catch any other unexpected errors and log the full traceback
            logger.error(f"[Loader] Producer failed to load {projection}/{skycell} with an unexpected error: {e}", exc_info=True)

    logger.info("[Loader] Producer has finished loading all tasks.")


def _consumer_processor(processed_queue, results_dict):
    """
    Consumer function: processes data in separate processes.
    Pulls raw data from the processed_queue, applies flux conversion (CPU-heavy),
    and stores the final combined image/mask in the results_dict.
    """
    while True:
        item = processed_queue.get()
        if item is None:
            break  # Sentinel value received, exit loop

        if len(item) == 5:  # data item with headers
            item_type, projection, skycell, data, headers = item
        else:  # mask item
            item_type, projection, skycell, data = item
            headers = None

        try:
            if item_type == "data":
                logger.info(f"[Consumer] Processing bands for {projection}/{skycell}...")
                # This is the CPU-bound part - include headers for proper flux conversion
                combined_image, _ = band_utils.process_skycell_bands(data, None, headers)

                # Store the result - need to handle Manager dict properly
                key = f"{projection}/{skycell}"
                if key not in results_dict:
                    results_dict[key] = {}

                # Get the current dict, modify it, and reassign
                current_dict = dict(results_dict[key])
                current_dict["image"] = combined_image
                results_dict[key] = current_dict

            elif item_type == "mask":
                logger.info(f"[Consumer] Combining masks for {projection}/{skycell}...")
                combined_mask = band_utils.combine_masks(data)

                # Store the result - need to handle Manager dict properly
                key = f"{projection}/{skycell}"
                if key not in results_dict:
                    results_dict[key] = {}

                # Get the current dict, modify it, and reassign
                current_dict = dict(results_dict[key])
                current_dict["mask"] = combined_mask
                results_dict[key] = current_dict

        except Exception as e:
            logger.error(f"[Loader] Consumer failed to process {projection}/{skycell}: {e}")

    logger.info("[Loader] A consumer process has finished.")


def run_processing_pipeline(zarr_path: str, skycells_to_process: list[tuple[str, str]], num_producers: int = 4, num_consumers: int = 4):
    """
    Orchestrates the producer-consumer pipeline for processing skycells.

    Args:
        zarr_path: Path to the Zarr store.
        skycells_to_process: A list of (projection, skycell) tuples to process.
        num_producers: Number of threads to use for loading data.
        num_consumers: Number of processes to use for CPU-bound work.

    Returns:
        A dictionary containing the processed images and masks.
    """
    with Manager() as manager:
        # Queue for tasks to be loaded by producers
        tasks_queue = manager.Queue()
        for task in skycells_to_process:
            tasks_queue.put(task)

        # Queue for raw data to be processed by consumers
        # A larger queue size acts as a buffer between producers and consumers
        processed_queue = manager.Queue(maxsize=len(skycells_to_process) * 2)

        # A process-safe dictionary to store final results
        results_dict = manager.dict()

        # --- Start Consumers First ---
        # They will run in separate processes and wait for data
        with ProcessPoolExecutor(max_workers=num_consumers) as consumer_executor:
            for _ in range(num_consumers):
                consumer_executor.submit(_consumer_processor, processed_queue, results_dict)

            # --- Then Start Producers ---
            # They will run in threads to load data and fill the queue
            with ThreadPoolExecutor(max_workers=num_producers) as producer_executor:
                for _ in range(num_producers):
                    producer_executor.submit(_producer_loader, tasks_queue, processed_queue, zarr_path)

            # --- Signal Consumers to Stop ---
            # Once producers are done, put sentinel values in the queue
            for _ in range(num_consumers):
                processed_queue.put(None)

        logger.info("[Loader] Pipeline execution complete.")
        # Convert the Manager dict to a regular Python dict
        return {k: dict(v) for k, v in results_dict.items()}


if __name__ == "__main__":
    # --- Example Usage ---
    # This is a dummy setup. Replace with your actual paths and IDs.
    skycells_to_run = [["2556", "080"], ["2556", "081"], ["2556", "082"], ["2556", "083"], ["2556", "084"], ["2556", "085"], ["2556", "086"], ["2556", "087"], ["2556", "088"], ["2556", "089"]]
    ZARR_PATH = "data/ps1_skycells_zarr/ps1_skycells.zarr"

    start_time = time.time()
    # Run the pipeline with 2 loader threads and 4 CPU processes
    final_results = run_processing_pipeline(ZARR_PATH, skycells_to_run, num_producers=4, num_consumers=10)
    end_time = time.time()

    print("\n--- Pipeline Summary ---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")
    print(f"Processed {len(final_results)} skycells.")
    # print("Final results keys:", final_results.keys())
