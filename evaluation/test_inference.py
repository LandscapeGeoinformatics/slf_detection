import numpy as np
import torch
from evaluation.utils.test import run_inference_mosaic

print('Using PyTorch version:', torch.__version__)
device = torch.device('cpu') # or 'cuda

# all the different things we have to fix to make things reproducible
seed_value = 0

torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
np.random.seed(seed_value)
# optional: for further ensuring reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import segmentation_models_pytorch as smp

# initialize model
model = smp.Unet(
    in_channels=3,
    classes=1,
)

checkpoint = torch.load('model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

epochs = checkpoint['epoch']
print(f"Load model trained for {epochs+1} epochs")

# run prediction
run_inference_mosaic(
    input_folder="/landscape_elements/working/test_sites/test",
    output_folder="output",
    model=model,
    patch_size=2000,   # patch size
    step=1500,         # stride
    batch_size=1,
    device=device,
    threshold=None,    # set threshold for direct binary output, else probability
    exts=(".vrt"),     # raster format can be can be ".tif" or ".vrt"
    blend="cosine"     # blending overlapping patch can be "average", "cosine/hann", or "gaussian"
)