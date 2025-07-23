import time
from process_ps1 import combine_ps1

datapath = "data/ps1_skycells/"
skycells = "data/SkyCells/skycell_s20_c11.csv"
savepath = "data/tess_comb_skycells"
catalog_path = "data/catalogs/Sector20_ccd11"

start = time.time()
combine_ps1(datapath=datapath, skycells=skycells, catalog_path=catalog_path, savepath=savepath, cores=30, verbose=1, overwrite=True)
end = time.time()
print(f"Process completed in {(end - start)/3600:.2f} hours.")
