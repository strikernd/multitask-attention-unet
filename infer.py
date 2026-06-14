import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF

from unet_model import UNet

def run_tta_inference():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths (Update to local directories)
    model_path = './unet_final_v1.pth'
    test_dir = './data/test_corrupt/'
    output_csv = './submission.csv'
    
    # Load Model
    print(f"Loading weights from {model_path}...")
    model = UNet().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.png')])
    submission_data = []
    batch_size = 64 

    print(f"Starting Test-Time Augmentation (TTA) Inference on {len(test_files)} images...")
    
    for i in tqdm(range(0, len(test_files), batch_size)):
        batch_filenames = test_files[i : i + batch_size]
        batch_tensors = []

        for f in batch_filenames:
            img_path = os.path.join(test_dir, f)
            img = Image.open(img_path).convert('RGB')
            batch_tensors.append(TF.to_tensor(img))

        imgs_t = torch.stack(batch_tensors).to(DEVICE)
        
        # Create augmented batch (Horizontal Flip)
        imgs_flipped = TF.hflip(imgs_t)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                # Pass 1: Normal
                recon_norm, _ = model(imgs_t)
                
                # Pass 2: Flipped
                recon_flip_raw, _ = model(imgs_flipped)
                recon_flip = TF.hflip(recon_flip_raw) # Un-flip the prediction

                # TTA Averaging to reduce model variance
                final_recon = (recon_norm + recon_flip) / 2.0
            
            # Format to 0-255 uint8 range
            final_recon = torch.clamp(final_recon, 0, 1) * 255.0
            final_recon = final_recon.round().cpu().numpy().astype(np.uint8)
            
            for j, img_np in enumerate(final_recon):
                # Flatten the image for submission format
                interleaved = img_np.transpose(1, 2, 0).flatten()
                img_id = batch_filenames[j].split('.')[0]
                submission_data.append([img_id] + interleaved.tolist())

    print(f"Generating {output_csv}...")
    columns = ['id'] + [f'pixel_{k}' for k in range(3072)]
    sub_df = pd.DataFrame(submission_data, columns=columns)
    sub_df.to_csv(output_csv, index=False)
    
    print("Success! Inference complete.")

if __name__ == "__main__":
    run_tta_inference()
