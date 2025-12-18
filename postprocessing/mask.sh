#!/bin/bash

#SBATCH --job-name=mask
#SBATCH --output=mask_%A.out
#SBATCH --error=mask_%A.error
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G

module load python/3.10.10

# Activate Micromamba environment
source ~/.bashrc
micromamba activate geopy

# Input mask (vector)
MASK_GPKG="/landscape_elements/working/scripts/pria_pollumassiivid_bf2m.gpkg"

# Folder with rasters to mask
INPUT_DIR="/landscape_elements/working/postprocessing/mosaic/tile"
OUT_DIR="/landscape_elements/working/postprocessing/mosaic/tile_masked"

mkdir -p "$OUT_DIR"

echo "Starting masking for rasters in $INPUT_DIR"

for tif in "$INPUT_DIR"/*.tif; do
    fname=$(basename "$tif")
    out="$OUT_DIR/${fname%.*}_masked.tif"

    echo "Processing: $fname"

    gdalwarp \
        -cutline "$MASK_GPKG" \
        -dstnodata 0 \
        -of COG \
        -co COMPRESS=LZW \
        -co BIGTIFF=YES \
        "$tif" "$out"

    echo "Done: $out"
done

echo "All rasters processed."

sbatch sieve_removal.sh
