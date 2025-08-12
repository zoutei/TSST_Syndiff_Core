import time

import matplotlib.pyplot as plt
import numpy as np

# Generate a large test image (6500 x 6500)
TEST_SHAPE = (6500, 65000)
TEST_SIGMA = 60.0
np.random.seed(42)
image = np.random.rand(*TEST_SHAPE).astype(np.float32)

results = {}

# # 1. SciPy
# try:
#     from scipy import ndimage

#     start = time.time()
#     scipy_result = ndimage.gaussian_filter(image, sigma=TEST_SIGMA, mode="constant", cval=np.nan)
#     elapsed = time.time() - start
#     results["scipy"] = elapsed
#     print(f"scipy.ndimage.gaussian_filter: {elapsed:.2f} s")
# except Exception as e:
#     print(f"scipy.ndimage.gaussian_filter: FAILED ({e})")

# # 2. OpenCV
# try:
#     import cv2

#     ksize = int(TEST_SIGMA * 6) | 1  # kernel size must be odd
#     start = time.time()
#     opencv_result = cv2.GaussianBlur(image, (ksize, ksize), TEST_SIGMA, borderType=cv2.BORDER_CONSTANT)
#     elapsed = time.time() - start
#     results["opencv"] = elapsed
#     print(f"cv2.GaussianBlur: {elapsed:.2f} s")
# except Exception as e:
#     print(f"cv2.GaussianBlur: FAILED ({e})")

# 3. Dask Array
try:
    import dask.array as da
    from dask.diagnostics import ProgressBar

    try:
        from dask_image.ndfilters import gaussian_filter as dask_gaussian_filter
    except ImportError:
        raise ImportError("dask_image is required for Dask gaussian_filter. Install with 'pip install dask-image'.")

    dimage = da.from_array(image, chunks=(1024, 1024))
    start = time.time()
    with ProgressBar():
        dask_result = dask_gaussian_filter(dimage, sigma=TEST_SIGMA, mode="constant", cval=np.nan).compute()
    elapsed = time.time() - start
    results["dask"] = elapsed
    print(f"dask_image.ndfilters.gaussian_filter: {elapsed:.2f} s")
except Exception as e:
    print(f"dask_image.ndfilters.gaussian_filter: FAILED ({e})")

# 4. PyTorch (CPU) -- Skipped due to excessive runtime
# try:
#     import torch
#     import torch.nn.functional as F
#
#     torch_image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
#
#     # Create Gaussian kernel
#     def get_gaussian_kernel2d(kernel_size, sigma):
#         ax = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
#         xx, yy = torch.meshgrid([ax, ax], indexing="ij")
#         kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
#         kernel = kernel / torch.sum(kernel)
#         return kernel
#
#     ksize = int(TEST_SIGMA * 6) | 1
#     kernel = get_gaussian_kernel2d(ksize, TEST_SIGMA)
#     kernel = kernel.unsqueeze(0).unsqueeze(0)
#     start = time.time()
#     torch_result = F.conv2d(torch_image, kernel, padding=ksize // 2)
#     elapsed = time.time() - start
#     results["torch"] = elapsed
#     print(f"torch.nn.functional.conv2d: {elapsed:.2f} s")
# except Exception as e:
#     print(f"torch.nn.functional.conv2d: FAILED ({e})")

# # 5. JAX (CPU)
# try:
#     import jax.numpy as jnp
#     from jax.scipy.ndimage import gaussian_filter as jax_gaussian_filter

#     jimage = jnp.array(image)
#     start = time.time()
#     jax_result = jax_gaussian_filter(jimage, sigma=TEST_SIGMA, mode="constant", cval=jnp.nan)
#     if hasattr(jax_result, "block_until_ready"):
#         jax_result.block_until_ready()
#     elapsed = time.time() - start
#     results["jax"] = elapsed
#     print(f"jax.scipy.ndimage.gaussian_filter: {elapsed:.2f} s")
# except Exception as e:
#     print(f"jax.scipy.ndimage.gaussian_filter: FAILED ({e})")

print("\nSummary (seconds):")
for k, v in results.items():
    print(f"{k}: {v:.2f}")

# plot using plt.imshow top and bottom before and after
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))
axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original Image")
axes[1].imshow(dask_result, cmap="gray")
axes[1].set_title("Dask Result")
plt.tight_layout()
plt.show()
