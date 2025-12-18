import rasterio
from rasterio import features
import torch
import torch.nn.functional as F
import numpy as np
import geopandas as gpd
import os
import glob
from tqdm import tqdm

# optional: helps reduce fragmentation for large CUDA allocations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# define device ('cpu' or 'cuda')
print('Using PyTorch version:', torch.__version__)
if torch.cuda.is_available():
    print('Using GPU, device name:', torch.cuda.get_device_name(0))
    device = torch.device('cuda')
else:
    print('No GPU found, using CPU instead.')
    device = torch.device('cpu')


# define function to rasterize polygon mask
def rasterize_gpkg_mask(gpkg_file, src, burn_value=1):
    """Rasterize polygons from GPKG to match the raster dataset (returns numpy mask)."""
    gdf = gpd.read_file(gpkg_file)
    mask = features.rasterize(
        [(geom, burn_value) for geom in gdf.geometry],
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype="uint8"
    )
    return mask


# define function to apply model and save prediction patches
def apply_model_and_save_prediction_patches(
    src,
    model,
    patch_size=500,
    step=250,
    batch_size=1,
    device=device,
    threshold=None,
    patch_output_folder="patch_outputs",
    base_name="image",
    mask=None,
):
    """Memory-efficient streaming inference: process one patch at a time and save as GeoTIFF."""
    
    import gc
    #import torch.nn.functional as F

    os.makedirs(patch_output_folder, exist_ok=True)

    model.eval()
    H, W = src.height, src.width
    total_patches = ((H + step - 1) // step) * ((W + step - 1) // step)

    pbar = tqdm(total=total_patches, desc=f"Processing {base_name}", unit="patch")
    patch_idx = 0

    with torch.inference_mode():
        for row_off in range(0, H, step):
            for col_off in range(0, W, step):
                # determine patch size
                width = min(patch_size, W - col_off)
                height = min(patch_size, H - row_off)
                window = rasterio.windows.Window(col_off, row_off, width, height)

                # read patch from raster
                patch = src.read(window=window)

                # convert to tensor and move to GPU
                patch_tensor = torch.from_numpy(patch).float() / 255.0
                patch_tensor = patch_tensor.unsqueeze(0).to(device)

                # pad to nearest multiple of 32 for model
                target_h = (height + 31) // 32 * 32
                target_w = (width + 31) // 32 * 32
                patch_tensor = F.pad(patch_tensor, (0, target_w - width, 0, target_h - height))

                # run inference
                output = torch.sigmoid(model(patch_tensor))

                # crop to original size
                pred = output[0, 0, :height, :width]

                # threshold or scaling
                if threshold is not None:
                    pred = (pred > threshold).float()
                else:
                    pred = pred * 1000

                # move to CPU and convert to numpy
                pred_np = np.squeeze(pred.cpu().numpy())

                # apply mask if provided
                if mask is not None:
                    pred_np *= mask[row_off:row_off + height, col_off:col_off + width]

                # convert dtype
                if np.issubdtype(pred_np.dtype, np.floating):
                    if threshold is not None:
                        pred_np = pred_np.astype(np.uint8)
                    else:
                        pred_np = np.clip(pred_np, 0, 65535).astype(np.uint16)

                # save patch
                transform = src.window_transform(window)
                meta = {
                    "driver": "GTiff",
                    "height": pred_np.shape[0],
                    "width": pred_np.shape[1],
                    "count": 1,
                    "dtype": str(pred_np.dtype),
                    "crs": src.crs,
                    "transform": transform,
                    "nodata": 0,
                    "compress": "lzw",
                    "photometric": "minisblack",
                }
                patch_filename = os.path.join(patch_output_folder, f"patch_{patch_idx:04d}.tif")
                with rasterio.open(patch_filename, "w", **meta) as dst:
                    dst.write(pred_np, 1)

                # cleanup to free memory
                del patch_tensor, output, pred, pred_np
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

                patch_idx += 1
                pbar.update(1)

    pbar.close()
    print(f"Saved {patch_idx} patch predictions in: {patch_output_folder}")


def run_inference_on_folder(
    input_folder,
    output_folder,
    model,
    patch_size=2000,
    step=1500,
    batch_size=1,
    device=device,
    threshold=None,
    exts=(".tif", ".vrt"),
    gpkg_file=None,
):
    """Run inference on all rasters in a folder and save per-patch predictions."""
    os.makedirs(output_folder, exist_ok=True)

    raster_files = []
    for ext in exts:
        raster_files.extend(glob.glob(os.path.join(input_folder, f"*{ext}")))

    for raster_file in raster_files:
        base_name = os.path.splitext(os.path.basename(raster_file))[0]
        patch_output_folder = os.path.join(output_folder, f"{base_name}_patches")
        os.makedirs(patch_output_folder, exist_ok=True)

        with rasterio.open(raster_file) as src:
            mask = rasterize_gpkg_mask(gpkg_file, src) if gpkg_file else None
            if mask is not None:
                print(f"Applying GPKG mask for {base_name}...")

            apply_model_and_save_prediction_patches(
                src,
                model,
                patch_size=patch_size,
                step=step,
                batch_size=batch_size,
                device=device,
                threshold=threshold,
                patch_output_folder=patch_output_folder,
                base_name=base_name,
                mask=mask,
            )
