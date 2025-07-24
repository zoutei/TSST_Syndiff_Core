import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from joblib import Parallel, delayed
from tqdm import tqdm
from glob import glob

from astropy.io import fits
from astropy.wcs import WCS


tess_filename = "data/tess/20_3_3/tess2020019135923-s0020-3-3-0165-s_ffic.fits"
tess_hdu = fits.open(tess_filename)
tess_wcs = WCS(tess_hdu[1].header)
tess_data = tess_hdu[1].data.astype(float)

t_y, t_x = np.shape(tess_data)
ty, tx = np.mgrid[:t_y, :t_x]
ty = ty.ravel().astype(int)
tx = tx.ravel().astype(int)

skycell_path = "data/tess_comb_skycells_conv/"

mask_files = glob(skycell_path + "*stk.rizy.conv.mask.fits")
ps1_files = glob(skycell_path + "*.stk.rizy.conv.fits")
ps1_files.sort()

registrations = glob("data/skycell_pixel_mapping/sector020/*.gz")
registrations.sort()
registrations = registrations[2:]

reg_sc = np.array([s.split("skycell.")[-1].split("_")[0] for s in registrations])
ps1_sc = np.array([s.split("skycell.")[-1].split(".stk")[0] for s in ps1_files])


def parallel_downsample(registration, ps1_files, reg_sc, ps1_sc, scene, scene_num, scene_mask, padding=500):
    reg = registration
    ps1_ind = np.where(reg_sc == ps1_sc)[0]
    if len(ps1_ind) > 0:
        try:
            # print(f"Processing {reg_sc[ps1_ind[0]]} with {ps1_files[ps1_ind[0]]}")
            y_pix, x_pix = scene.shape
            ps1_assignment_x = fits.open(reg)[1].data
            ps1_assignment_x = ps1_assignment_x.astype(int)
            ps1_assignment_x[ps1_assignment_x == 65535] = -1
            ps1_assignment_y = fits.open(reg)[2].data
            ps1_assignment_y = ps1_assignment_y.astype(int)
            ps1_assignment_y[ps1_assignment_y == 65535] = 0

            ps1_assignment = ps1_assignment_x + ps1_assignment_y * x_pix

            pind = ps1_assignment.ravel()
            sort_ind = np.argsort(pind)

            ps1_filename = ps1_files[ps1_ind[0]]
            ps1_mask_filename = ps1_filename.replace(".fits", ".mask.fits")
            ps1_data = fits.open(ps1_filename)[0].data
            ps1_mask = fits.open(ps1_mask_filename)[0].data
            ps1_rav = ps1_data[padding:-padding, padding:-padding].ravel()[sort_ind]
            ps1_mask_rav = ps1_mask[500:-500, 500:-500].ravel()[sort_ind]

            tess_pixels = np.unique(pind[np.isfinite(pind)]).astype(int)
            tess_pixels = tess_pixels[tess_pixels >= 0]

            breaks = np.where(np.diff(pind[sort_ind]) > 0)[0] + 1
            breaks = np.append(breaks, len(ps1_rav))

            sums = np.zeros(len(breaks) - 1, dtype=float)
            isums = np.zeros(len(breaks) - 1, dtype=int)
            msums = np.zeros(len(breaks) - 1, dtype=int)
            for i in range(len(breaks) - 1):
                sums[i] = np.sum(ps1_rav[breaks[i]: breaks[i + 1]])
                isums[i] = breaks[i + 1] - breaks[i]
                msums[i] = np.sum(ps1_mask_rav[breaks[i] : breaks[i + 1]] != 0)

            scene[ty[tess_pixels], tx[tess_pixels]] += sums
            scene_num[ty[tess_pixels], tx[tess_pixels]] += isums
            scene_mask[ty[tess_pixels], tx[tess_pixels]] += msums

        except Exception as e:
            print(f"Error processing {reg_sc}: {e}")
    return scene, scene_num, scene_mask


def run_parallel_downsample(registrations, reg_sc_split, tess_data_shape, ps1_files, ps1_sc):
    scene = np.zeros(tess_data_shape)
    scene_num = np.zeros(tess_data_shape, dtype=int)
    scene_mask = np.zeros(tess_data_shape, dtype=int)
    for i, reg_sc in enumerate(reg_sc_split):
        scene, scene_num, scene_mask = parallel_downsample(registration=registrations[i], ps1_files=ps1_files, reg_sc=reg_sc, ps1_sc=ps1_sc, scene=scene, scene_num=scene_num, scene_mask=scene_mask)
        print(f"Processed skycell({i+1}/{len(reg_sc_split)})")
    return scene, scene_num, scene_mask


n = 60
start_time = time.time()
reg_sc_split = np.array_split(reg_sc, n)
registrations_split = np.array_split(np.array(registrations), n)
inds = np.arange(len(reg_sc_split))
tess_data_shape = tess_data.shape

scenes = Parallel(n_jobs=n)(delayed(run_parallel_downsample)(registrations=registrations_split[i], reg_sc_split=reg_sc_split[i], tess_data_shape=tess_data_shape, ps1_files=ps1_files, ps1_sc=ps1_sc) for i in inds)

scenes = np.array(scenes)
np.save("data/tess_downsampled_detailed.npy", scenes)

scene = np.sum(scenes, axis=0)
np.save("data/tess_downsampled.npy", scene)
print(f"Total time taken: {((time.time() - start_time)/60):.2f} minutes")
