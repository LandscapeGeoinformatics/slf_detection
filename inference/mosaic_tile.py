import argparse
import json
import os
from inference.utils.blend import blend_patches_to_raster  # updated function

parser = argparse.ArgumentParser()
parser.add_argument("--part", type=str, required=True, help="Tile key, e.g., part_0_0")
parser.add_argument("--json", type=str, default="patch_groups_4x4.json", help="JSON file with patch lists")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save tile mosaics")
parser.add_argument("--blend", type=str, default="hann", help="Blend mode: average, smooth, hann")
parser.add_argument("--dtype", type=str, default="uint16", help="Output data type: uint16 or uint8")
parser.add_argument("--mask", type=str, default=None, help="Optional mask raster to apply")
args = parser.parse_args()

# --- Load patch list for this tile ---
with open(args.json) as f:
    groups = json.load(f)

if args.part not in groups:
    raise ValueError(f"Tile key '{args.part}' not found in JSON")

patch_files = groups[args.part]

# --- Output path for this tile ---
os.makedirs(args.output_dir, exist_ok=True)
output_path = os.path.join(args.output_dir, f"mosaic_{args.part}.tif")

# --- Call the blending function directly with patch_files ---
blend_patches_to_raster(
    output_path=output_path,
    patch_files=patch_files,
    blend=args.blend,
    dtype=args.dtype,
    mask_file=args.mask  # Pass the mask argument
)

print(f"Tile {args.part} done: {output_path}")
