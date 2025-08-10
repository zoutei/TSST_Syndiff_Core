import os
import argparse
import requests
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

def get_img_name(skycell_name, filters="rizy", home_dir="./", domask=False):
    skycell_name = skycell_name.split(".")
    output_dir = os.path.join(home_dir, f"{skycell_name[1]}/{skycell_name[2]}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filter in filters:
        file_name = f"{skycell_name[1]}/{skycell_name[2]}/rings.v3.skycell.{skycell_name[1]}.{skycell_name[2]}.stk.{filter}.unconv.fits"
        if domask:
            file_name = file_name.split(".")
            file_name[-1] = "mask.fits"
            file_name = ".".join(file_name)
        if not os.path.exists(os.path.join(home_dir, file_name)):
            url = "http://ps1images.stsci.edu/rings.v3.skycell/" + file_name
            # print(url)
            response = requests.get(url)
            if response.status_code == 200:
                outputfilename = os.path.join(output_dir, file_name.split("/")[-1] + ".fz")
                with open(outputfilename, "wb") as file:
                    file.write(response.content)
                os.system("funpack -D " + outputfilename)
                # print("Downloaded " + file_name.split("/")[-1])
                # return "Downloaded"
            # else:
            #     print("Failed to download the file " + file_name.split("/")[-1])
                # return "Not Found"


def download_images_for_row(row, save_path: str, filters: str = "rizy", download_masks: bool = True):
    """Download images for a single row"""
    if row is not None:
        try:
            get_img_name(row, filters=filters, home_dir=save_path, domask=False)
            if download_masks:
                get_img_name(row, filters=filters, home_dir=save_path, domask=True)
        except Exception as e:
            print(f"Error downloading image for row: {e}")


if __name__ == "__main__":
# Example usage (update paths to your sector/camera/ccd):
    sector = 20
    camera = 3
    ccd = 3
    skycells_df = pd.read_csv(f"data/skycell_pixel_mapping/sector_{sector:04d}/camera_{camera}/ccd_{ccd}/tess_s{sector:04d}_{camera}_{ccd}_master_skycells_list.csv")
    save_path = "data/ps1_skycells"
    unique_ps1_images = skycells_df["NAME"].unique()
    unique_ps1_images.sort()
    Parallel(n_jobs=60)(delayed(download_images_for_row)(row, save_path) for row in tqdm(unique_ps1_images, desc="Downloading images"))
