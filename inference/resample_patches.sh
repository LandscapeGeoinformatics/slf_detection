#!/bin/bash

#SBATCH --job-name=resample
#SBATCH --output=resample_%A.out
#SBATCH --error=resample_%A.error
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=5-00:00:00
#SBATCH --mem=6G

module load python/3.10.10

# Activate Micromamba environment
source ~/.bashrc
micromamba activate geopy

cd /landscape_elements/working/scripts/wide_pred2/rgb_summer_patches

find . -type f -name "*.tif" | while read -r i; do

    out="${i%.*}_out.vrt"

    if [ -f "$out" ]; then

        echo "Skipping $out (already exists)"

        continue

    fi

    gdalwarp -of VRT "$i" "$out" -t_srs "EPSG:3301" -tr 1 1 -r near

done
