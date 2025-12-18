import rasterio

import os
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# function for blending mode for overlapping patches (average, cosine, gaussian)

device = torch.device('cpu')

def get_blend_window(h, w, mode="average", device=device):
    """Generate blending weights of shape (h, w)."""
    # average
    if mode == "average":
        return torch.ones((h, w), dtype=torch.float32, device=device)
    
    # cosine
    elif mode == "cosine":
        y = torch.hann_window(h, periodic=True, dtype=torch.float32, device=device)
        x = torch.hann_window(w, periodic=True, dtype=torch.float32, device=device)
        return torch.ger(y, x)

    # gaussian
    elif mode == "gaussian":
        # σ = 1/4 of patch size
        sigma_y = h / 4.0
        sigma_x = w / 4.0
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
        )
        cy, cx = h / 2.0, w / 2.0
        gauss = torch.exp(-(((yy - cy) ** 2) / (2 * sigma_y**2) + ((xx - cx) ** 2) / (2 * sigma_x**2)))
        return gauss / gauss.max()

    else:
        raise ValueError(f"Unknown blend mode: {mode}. Use average, cosine, or gaussian.")
 
 
# function for applying model and output wide prediction
def apply_segmentation_tensor(image_tensor, model, patch_size=500, step=250, batch_size=1, device=device, threshold=None, blend="average"):
    """
    image_tensor: torch.Tensor of shape (1, C, H, W), values in 0-1
    threshold: if set (e.g. 0.5), output will be binary mask, otherwise probabilities
    Returns prediction map of shape (H, W)
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    _, C, H, W = image_tensor.shape

    # store sums and counts
    predictions = torch.zeros((1, 1, H, W), dtype=torch.float32, device='cpu')
    counts = torch.zeros((1, 1, H, W), dtype=torch.float32, device='cpu')

    with torch.inference_mode():
        for row in tqdm(range(0, H - patch_size + 1, step), desc="Processing Rows"):
            row_patches = []
            locations = []

            for col in range(0, W - patch_size + 1, step):
                patch = image_tensor[:, :, row:row + patch_size, col:col + patch_size]

                # pad each patch to multiple of 32
                _, _, ph, pw = patch.shape
                pad_h = (32 - ph % 32) % 32
                pad_w = (32 - pw % 32) % 32
                patch_padded = F.pad(patch, (0, pad_w, 0, pad_h))

                row_patches.append(patch_padded)
                locations.append((row, col, ph, pw))  # keep original size

            if not row_patches:
                continue

            batch = torch.cat(row_patches, dim=0)

            outputs = []
            for i in range(0, batch.shape[0], batch_size):
                sub_batch = batch[i:i + batch_size].to(device)
                sub_output = model(sub_batch)
                sub_output = torch.sigmoid(sub_output)
                outputs.append(sub_output.cpu())

            output_tensor = torch.cat(outputs, dim=0)

            # crop back to original patch size and blend
            for i, (r, c, ph, pw) in enumerate(locations):
                pred_patch = output_tensor[i, 0, :ph, :pw]
                weight = get_blend_window(ph, pw, mode=blend, device=device)

                predictions[:, :, r:r + ph, c:c + pw] += pred_patch * weight
                counts[:, :, r:r + ph, c:c + pw] += weight

            torch.cuda.empty_cache()

    predictions = predictions / torch.clamp(counts, min=1e-6)

    # optional thresholding
    if threshold is not None:
        predictions = (predictions > threshold).float()
    else:
        predictions = (predictions * 1000).to(torch.uint16)

    return predictions[0, 0]  # shape: (H, W)


# function for passing raster file to be predicted by the model
def run_inference_mosaic(input_folder, output_folder, model,
                            patch_size=500, step=250, batch_size=4,
                            device=device, threshold=None, exts=(".tif", ".vrt"), blend="average"):
    """
    Apply model to all raster files in input_folder. Supports multiple extensions.
    
    exts: tuple of extensions to process (e.g., ('.tif', '.vrt'))
    """
    os.makedirs(output_folder, exist_ok=True)

    # Collect all files with the specified extensions
    raster_files = []
    for ext in exts:
        raster_files.extend(glob.glob(os.path.join(input_folder, f"*{ext}")))

    for raster_file in raster_files:
        base_name = os.path.basename(raster_file)
        name_no_ext = os.path.splitext(base_name)[0]

        # read raster
        with rasterio.open(raster_file) as src:
            img = src.read()
            img = np.transpose(img, (1, 2, 0))  # (H, W, C)
            ras_meta = src.profile

        # convert to tensor (PIL automatically rescale to 0-1)
        pil_image = Image.fromarray(img.astype(np.uint8))
        image_tensor = transforms.ToTensor()(pil_image).unsqueeze(0)  # (1, C, H, W)

        # apply model
        prediction = apply_segmentation_tensor(
            image_tensor, model,
            patch_size=patch_size,
            step=step,
            batch_size=batch_size,
            device=device,
            threshold=threshold,
            blend="average" # "average", "cosine", or "gaussian"
        )
        prediction_np = prediction.squeeze().cpu().numpy()#.astype(np.float32)

        # update metadata
        ras_meta.update({
            'count': 1,
            'dtype': 'uint16' if threshold is None else 'uint8',
            'driver': 'GTiff',
            'photometric': 'minisblack',
            'compress': 'lzw',
            'nodata': 0,
        })

        # save output
        output_path = os.path.join(output_folder, f"{name_no_ext}_pred.tif")
        with rasterio.open(output_path, 'w', **ras_meta) as dst:
            dst.write(prediction_np, 1)

        print(f"Saved: {output_path}")