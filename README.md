# Multitask Attention UNet for Image Reconstruction Under Corruption

This repository contains a custom PyTorch UNet architecture built entirely from scratch to restore images subjected to severe corruption, noise, blurring, and masking. 

Unlike standard UNet implementations, this model incorporates several advanced computer vision techniques, including auxiliary multi-task learning, spatial attention mechanisms, and test-time augmentation, to maximize reconstruction fidelity.

## Performance
* Metric: Mean Squared Error (MSE) / F1-Score for bottleneck classification.
* Reconstructs 32x32 images corrupted by extreme brightness shifts, spatial translations, and randomized sparse pixel masking.

## Tech Stack
* Frameworks: PyTorch, Torchvision
* Optimization: Mixed Precision Training (`torch.amp`), OneCycleLR Scheduling
* Evaluation: Scikit-learn (F1 Score)

## Architecture & Technical Innovations
This architecture deviates from standard tutorial implementations by introducing several production-grade improvements:

1. Multi-Task Learning via Auxiliary Bottleneck Classification
A classification head was added at the lowest point (bottleneck) of the UNet. By calculating Cross-Entropy loss for class prediction alongside MSE pixel loss, the latent space is forced to organize semantically. This acts as a powerful regularizer (e.g., if the bottleneck mathematically identifies a "truck," the decoder is heavily biased to reconstruct sharp edges rather than organic textures).

2. Spatial Attention Gates
Instead of blindly passing noisy encoder features across skip-connections to the decoder, custom Attention Gates (utilizing Adaptive Pooling and Sigmoid activations) were implemented. These gates learn to actively suppress corruption and highlight only relevant spatial features before merging.

3. Anti-Aliasing via Bilinear Upsampling
Standard `ConvTranspose2d` layers frequently cause "checkerboard artifacts" in generative tasks. This model replaces them with Bilinear Upsampling followed by standard convolutions to mathematically interpolate pixels smoothly and avoid grid distortions.

4. Dim-Pixel Gradient Flow (PReLU)
Standard ReLU activations permanently kill negative values (dead neurons), which destroys edge detail in dark or dimly lit areas of an image. Parametric ReLU (PReLU) was used throughout the network to learn dynamic slopes for negative values, ensuring gradients continuously flow through dark pixels.

5. Test-Time Augmentation (TTA)
The inference pipeline utilizes TTA. It passes both the raw corrupted image and a horizontally flipped version through the network, un-flips the generated output, and averages the pixels. This significantly reduces the variance of the model's spatial predictions and smooths out hallucinated artifacts.

## Repository Structure
* `unet_model.py` - Contains the modular neural network classes (`DeepResBlock`, `UpsampleBlock`, `AttentionGate`, `UNet`).
* `train.py` - The training script. Handles dataset loading, heavy data augmentation (flips, translations, sparse noise injection), and Mixed Precision training.
* `infer.py` - The inference script. Loads the trained weights and runs the Test-Time Augmentation loop to generate the final reconstructed images.
* `requirements.txt` - Project dependencies.

## How to Run

### 1. Environment Setup
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Training the Model
Ensure your data is structured with `train.csv`, `train_clean/`, and `train_corrupt/` directories inside a `./data/` folder.
```bash
python train.py
```
Note: The script outputs the trained model weights and recovered label mappings to `./unet_final_v1.pth`.

### 3. Running Inference
Ensure your test images are located in `./data/test_corrupt/`.
```bash
python infer.py
```
Outputs a `submission.csv` containing the flattened pixel arrays of the reconstructed images.

## License
This code is released under the [MIT License](LICENSE).
