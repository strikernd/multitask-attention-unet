import os
import random
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T

# Import the custom model structure
from unet_model import UNet

# ---------------------------------------------------------
# 1. Dataset & Augmentation

class UNetDataset(Dataset):
    def __init__(self, csv_path, clean_dir, corrupt_dir, augment=True):
        df = pd.read_csv(csv_path)
        self.augment = augment
        self.label_map = {n: i for i, n in enumerate(sorted(df['class'].unique()))}
        self.data = []
        
        print("Pre-loading data to RAM...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            c = TF.to_tensor(Image.open(os.path.join(corrupt_dir, row['corrupt_filename'])).convert('RGB'))
            t = TF.to_tensor(Image.open(os.path.join(clean_dir, row['clean_filename'])).convert('RGB'))
            self.data.append((c, t, self.label_map[row['class']]))

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        c_img, t_img, label = self.data[idx]
        
        if self.augment:
            # Horizontal Flip
            if random.random() > 0.5:
                c_img, t_img = TF.hflip(c_img), TF.hflip(t_img)
            
            # Translation with reflection padding to prevent black borders
            c_img = TF.pad(c_img, 2, padding_mode='reflect')
            t_img = TF.pad(t_img, 2, padding_mode='reflect')
            i, j, h, w = T.RandomCrop.get_params(c_img, (32, 32))
            c_img, t_img = TF.crop(c_img, i, j, h, w), TF.crop(t_img, i, j, h, w)

            # Brightness jitter
            bright = random.uniform(0.9, 1.1)
            c_img = torch.clamp(c_img * bright, 0, 1)
            t_img = torch.clamp(t_img * bright, 0, 1)
            
            # Sparse Random Pixel swap (Noise injection)
            if random.random() > 0.5:
                indices = torch.randint(0, 32, (10, 2))
                for idx_pair in indices:
                    c_img[:, idx_pair[0], idx_pair[1]] = torch.rand(3)

        return c_img, t_img, label

# ---------------------------------------------------------
# 2. Training Pipeline

def train_model():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths (Update to local directories)
    train_csv = './data/train.csv'
    clean_dir = './data/train_clean/'
    corrupt_dir = './data/train_corrupt/'
    save_path = './unet_final_v1.pth'
    
    TOTAL_EPOCHS = 140
    BATCH_SIZE = 512
    MAX_LR = 4e-3
    
    model = UNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR/10, weight_decay=1e-2)
    
    ds = UNetDataset(train_csv, clean_dir, corrupt_dir, augment=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=MAX_LR, 
        steps_per_epoch=len(loader), 
        epochs=TOTAL_EPOCHS
    )
    
    scaler = torch.amp.GradScaler('cuda')
    mse_crit = nn.MSELoss()
    ce_crit = nn.CrossEntropyLoss()

    print(f"Starting Training on {DEVICE} for {TOTAL_EPOCHS} epochs...")

    for epoch in range(TOTAL_EPOCHS):
        model.train()
        epoch_labels, epoch_preds = [], []
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}", leave=False)
        
        for noisy, clean, label in pbar:
            noisy, clean, label = noisy.to(DEVICE), clean.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            
            # Mixed Precision Forward Pass
            with torch.amp.autocast(device_type='cuda'):
                out, cls_logits = model(noisy)
                # Combined Loss: Pixel MSE + Auxiliary Class Cross-Entropy
                loss = mse_crit(out, clean) + 0.1 * ce_crit(cls_logits, label)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            with torch.no_grad():
                epoch_labels.append(label.cpu())
                epoch_preds.append(cls_logits.argmax(1).cpu())
                k_mse = mse_crit(out, clean).item() * (255**2)
                pbar.set_postfix({'k_mse': f"{k_mse:.1f}"})

        y_t = torch.cat(epoch_labels).numpy()
        y_p = torch.cat(epoch_preds).numpy()
        f1 = f1_score(y_t, y_p, average='macro')
        print(f"Epoch {epoch+1} | F1: {f1:.4f} | k_mse: {k_mse:.1f}")

    # ---------------------------------------------------------
    # 3. Save Final Model & Label Map
    
    print("\nTraining complete. Saving weights...")
    save_dict = {
        'model_state': model.state_dict(),
        'label_map': ds.label_map
    }
    torch.save(save_dict, save_path)
    print(f"Success: Model and mapping saved to {save_path}")

if __name__ == "__main__":
    train_model()
