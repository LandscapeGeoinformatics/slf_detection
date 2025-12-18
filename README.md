# slf_segmentation
Code supplement: Mapping small landscape features in agricultural lands using CNN-based semantic segmentation

## Environment setup
- Install micromamba. See https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html
- Clone the repository
- Navigate to the repository folder, and create a new micromamba environment using `micromamba env create --file environment.yml`

## Usage
### Model training
- Specify input data and hyperparameters in `training/train.py` and run
- Use the stored weights (*.pt) in testing and inference steps
- Use the training logs in csv to plot the model behavior during training
### Model testing
- Run `inference/test_inference.py` on a single test site image (e.g. 5x5km2) to obtain probability prediction at test-site level
- Calculate AUC/ROC to define optimal probability threshold and accuracy metrics with `inference/test_accuracy.py`.
### Inference on mosaic
- Run `inference/patches_inference` to obtain probability prediction at patches level
- Batch resample each probability patch into coarser resolution virtual rasters (vrt) with `inference/resample_patches.sh`
- Group each vrt into processing tiles using `inference/group_patches_to_tile.py`.
- Run `inference/mosaic_tile.py` to create mosaic based on patches
### Post-processing
- To apply non-arable land mask, run `postprocessing/mask.sh`
- To remove sieve pixels, run `postprocessing/sieve_removal.py`
- Convert probability mosaic raster(s)/tiles into polygon (geopackage) using `postprocessing/polygonize.py` 
- Simplify and smoothen the vertices of each polygon features using `postprocessing/smooth_polygon.py`.
