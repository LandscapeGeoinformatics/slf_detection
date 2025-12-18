import rasterio
import os
import numpy as np
import json

patch_folder = "rgb_summer_patches"  # folder with patch_0000.tif ... patch_680505.tif
ncols = 4
nrows = 4
output_json = "patch_groups_4x4.json"

# Get patch list
patch_files = sorted([
    os.path.join(patch_folder, f)
    for f in os.listdir(patch_folder)
    if f.lower().endswith(".vrt")
])

# Get bounds
bounds_list = []
for f in patch_files:
    with rasterio.open(f) as src:
        bounds_list.append(src.bounds)

minx = min(b.left for b in bounds_list)
maxx = max(b.right for b in bounds_list)
miny = min(b.bottom for b in bounds_list)
maxy = max(b.top for b in bounds_list)

# Compute grid edges
x_edges = np.linspace(minx, maxx, ncols + 1)
y_edges = np.linspace(miny, maxy, nrows + 1)

# Assign patches to grid cells
grid_groups = {(r, c): [] for r in range(nrows) for c in range(ncols)}

for f, b in zip(patch_files, bounds_list):
    cx = (b.left + b.right) / 2
    cy = (b.bottom + b.top) / 2
    col = np.searchsorted(x_edges, cx) - 1
    row = np.searchsorted(y_edges, cy) - 1
    if 0 <= row < nrows and 0 <= col < ncols:
        grid_groups[(row, col)].append(f)

# Save JSON
groups_json = {f"part_{r}_{c}": files for (r, c), files in grid_groups.items() if files}
with open(output_json, "w") as f:
    json.dump(groups_json, f, indent=2)

print(f"Patch groups saved to {output_json}")
