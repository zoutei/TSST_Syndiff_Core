import pandas as pd
import os
import requests
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


def download_images_for_row(row):
    """Download images for a single row"""
    save_path = "data/ps1_skycells/"
    if row is not None:
        try:
            get_img_name(row, filters="rizy", home_dir=save_path, domask=False)
            get_img_name(row, filters="rizy", home_dir=save_path, domask=True)
        except Exception as e:
            print(f"Error downloading image for row: {e}")


skycells_df = pd.read_csv("data/SkyCells/skycell_s20_c11.csv")
unique_ps1_images = skycells_df["Name"].unique()
unique_ps1_images.sort()
# unique_ps1_images = unique_ps1_images[:1]
Parallel(n_jobs=12)(delayed(download_images_for_row)(row) for row in tqdm(unique_ps1_images, desc="Downloading images"))
