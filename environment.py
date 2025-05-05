from pysim.parsing import File, Folder 
"""
These are the environment variables that tells pysim where to find/look for different directories or templates or whatnot
"""
pysimDir = Folder("/home/x-kgootkin/pysim") # Where the package lives
simulationDir = Folder("/anvil/scratch/x-kgootkin/sims/") # default location to put/look for simulations
dHybridRtemplate = Folder(pysimDir.path + "/templates/dHybridR/") # where the base dHybridR template is
figDir = Folder("/home/x-kgootkin/turbulence/figures/") # default location for figures to go
frameDir = Folder("/home/x-kgootkin/frames/") # default location for frames of videos to go
videoDir = Folder("/home/x-kgootkin/videos/") # default location for videos to go