import numpy as np
import torch
import segmentation_models_pytorch as smp
from utils.pred import run_inference_on_folder

# define device ('cpu' or 'cuda')
print('Using PyTorch version:', torch.__version__)
if torch.cuda.is_available():
    print('Using GPU, device name:', torch.cuda.get_device_name(0))
    device = torch.device('cuda')
else:
    print('No GPU found, using CPU instead.')
    device = torch.device('cpu')

# All the different things we have to fix to make things reproducible
seed_value = 0

torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
np.random.seed(seed_value)

# Optional: for further ensuring reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_grad_enabled(False) # no gradients needed for inference

# load model
n_channel = 3
n_class = 1 # binary=1

# initialize model
model = smp.Unet(
    in_channels=n_channel, # adjust to image channel n
    classes=n_class,
)

# load model weights
checkpoint = torch.load('/gpfs/helios/home/fauzan/landscape_elements/working/scripts/train_aug/best_model_unetresnet50_bce_1e-4.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

epochs = checkpoint['epoch']
print(f"Load model trained for {epochs+1} epochs")

# apply model
run_inference_on_folder(
    input_folder="/landscape_elements/working/orthophotos/summer/mosaic", 
    output_folder="/landscape_elements/working/pred", 
    model=model,
    patch_size=2000, 
    step=1500, 
    batch_size=1,
    device=device, 
    threshold=None, 
    exts=(".tif", ".vrt"),
    gpkg_file=None,
    
)
