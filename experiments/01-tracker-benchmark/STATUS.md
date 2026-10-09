# Experiment 1 live status

Completed: 12 uniform-color debugtest +12 separatelyseededcalibration real SAPIEN clips; frozen CoTracker3+GTdepth/poses, DELTA+GTdepth/poses, SpaTrackerV2 BF16 and corrected float32outer/SDPA-math GTgeometry all12each. RGB-only Spa geometry frontend cachedall12; CoTracker andSpa use identicalcache, withone initialGTdepthscale diagnostic. Raw unscaled nativegeometry saved. Alllearnedtracker scientificdata inNVMe.

Extended axis/velocity/acceleration, GTvisible/occluded/dynamic/static/foreground masks, and native camera/depth/focal diagnostics saved inextended_metrics.json beside eachbundle. GT exact-front-ray lift visibleEPE<1e-8m; rendered-nearest-depth variant explicitly measuresrasteraliasing. InvalidFP32silentattentionfallback quarantined andexcluded. Legacyone-channelzero broadcastcontrol replacedbyexplicit3Dzero_motion_xyz.

Newtextured64×64querystaticfixed/orbit GT pair isvalidated; main-gridtracker profiling waitingtemporarilyforExp02leaseofGPU0/1. Noexp01GPUprocessrunningat03:12UTC.

Data: `/mnt/nvme/scratch/phyla-ubuntu/data/exp01/`
Reports/footage: `results/ai/debug-pilot/`
Logs: `/mnt/nvme/scratch/phyla-ubuntu/runs/exp01/`

Debuggeometryuniformsurfacesandone sceneperarchetype limitgeneralizationclaims. Scriptedlinkedrigidboxesratherthannativerobotjointarticulation. OfficialTAPVid3Dmetric integration, broadscene/assetreplication notdone. Nodeexpires05:11UTC,09Oct2026; archiveby05:05.
