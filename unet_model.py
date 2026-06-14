import torch
import torch.nn as nn

# ---------------------------------------------------------
# Neural Network Building Blocks

class DeepResBlock(nn.Module):
    """
    A 3-layer convolutional residual block.
    Uses PReLU (Parametric ReLU) instead of standard ReLU to allow gradient flow 
    through negatively valued (dim/dark) pixels, preventing dead neurons during reconstruction.
    """
    def __init__(self, in_c, out_c, stride=1, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.prelu1 = nn.PReLU()
        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.prelu3 = nn.PReLU()
        
        self.dropout = nn.Dropout2d(dropout)

        # Shortcut to match dimensions for the residual connection
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = self.prelu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.dropout(out)
        
        # Final residual addition
        return self.prelu3(out + residual)

class UpsampleBlock(nn.Module):
    """
    Uses Bilinear Upsampling instead of ConvTranspose2d to scale up spatial dimensions.
    This interpolates pixels smoothly and actively prevents 'checkerboard artifacts' 
    commonly seen in generative image tasks.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.up(x)
        out = self.prelu(self.bn(self.conv(out)))
        return out

class AttentionGate(nn.Module):
    """
    Spatial Attention Gate to filter skip-connections.
    Instead of blindly passing noisy encoder features to the decoder, this gate uses 
    Adaptive Average Pooling and a Sigmoid activation to learn which spatial features 
    are relevant, suppressing noise before concatenation.
    """
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, skip):
        return skip * self.gate(skip)


# ---------------------------------------------------------
# Main UNet Architecture

class UNet(nn.Module):
    """
    Custom UNet featuring Multi-Task Learning.
    Incorporates an auxiliary classification head at the bottleneck. By calculating Cross-Entropy 
    loss alongside MSE pixel loss, the latent space is forced to organize semantically, 
    acting as a powerful regularizer for image reconstruction.
    """
    def __init__(self):
        super().__init__()
        
        # Encoder Path
        self.enc1 = DeepResBlock(3, 64)
        self.down1 = DeepResBlock(64, 128, stride=2) 
        
        self.enc2 = DeepResBlock(128, 128)
        self.down2 = DeepResBlock(128, 256, stride=2) 
        
        self.enc3 = DeepResBlock(256, 256)
        self.down3 = DeepResBlock(256, 512, stride=2)

        # Bottleneck
        self.bottleneck = DeepResBlock(512, 512, dropout=0.2)
        
        # Auxiliary Classification Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

        # Decoder Path with Attention Gates
        self.att3 = AttentionGate(256)
        self.up3 = UpsampleBlock(512, 256)
        self.dec3 = DeepResBlock(256, 256)

        self.att2 = AttentionGate(128)
        self.up2 = UpsampleBlock(256, 128)
        self.dec2 = DeepResBlock(128, 128)

        self.att1 = AttentionGate(64)
        self.up1 = UpsampleBlock(128, 64)
        self.dec1 = DeepResBlock(64, 64)

        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x_noisy):
        # Encoder
        s1 = self.enc1(x_noisy)
        s2 = self.enc2(self.down1(s1))
        s3 = self.enc3(self.down2(s2))
        
        # Bottleneck & Auxiliary Classification
        b = self.bottleneck(self.down3(s3))
        cls_logits = self.classifier(b)
        
        # Decoder
        d3 = self.dec3(self.up3(b) + self.att3(s3))
        d2 = self.dec2(self.up2(d3) + self.att2(s2))
        d1 = self.dec1(self.up1(d2) + self.att1(s1))
        
        # Predict residue to subtract (Residual Learning for image restoration)
        reconstructed = x_noisy - self.final_conv(d1)
        
        return reconstructed, cls_logits
