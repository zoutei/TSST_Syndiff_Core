from pancakes import Pancakes

tess_file = 'data/tess/20_3_3/tess2020019172923-s0020-3-3-0165-s_ffic.fits'
savepath = 'data/skycell_pixel_mapping/sector020'

pancakes = Pancakes(tess_file, savepath=savepath, num_cores=28, sector=20, use_multiple_cores_per_task=False, overwrite=False, buffer=120)

pancakes.complete_mapping()
