import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from tqdm import tqdm

class CoordResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 1) Load a ResNet18 and replace first conv to accept 5 channels (RGB + X + Y)
        backbone = resnet18(pretrained=pretrained)
        w0 = backbone.conv1.weight.data.clone()  # (64,3,7,7)
        self.conv1 = nn.Conv2d(
            in_channels=5, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        # init: copy pretrained RGB weights, zero for the two coord channels
        with torch.no_grad():
            self.conv1.weight[:, :3] = w0
            self.conv1.weight[:, 3:] = 0

        # 2) Hook the rest of ResNet (but **remove** its fc)
        self.backbone = nn.Sequential(
            self.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,   # outputs (B,512,1,1)
        )
        # 3) A small regression head on top
        self.regressor = nn.Sequential(
            nn.Flatten(),         # → (B,512)
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),    # predict normalized (x, y) in [0,1]
        )

    def forward(self, x):
        # x: (B,3,H,W) in [0,1]
        B, C, H, W = x.shape

        # build coord channels, normalized to [-1,1]
        xs = torch.linspace(-1, 1, steps=W, device=x.device)
        ys = torch.linspace(-1, 1, steps=H, device=x.device)
        grid_x = xs.view(1, 1, 1, W).expand(B, 1, H, W)
        grid_y = ys.view(1, 1, H, 1).expand(B, 1, H, W)

        x = torch.cat([x, grid_x, grid_y], dim=1)  # → (B,5,H,W)
        f = self.backbone(x)                       # → (B,512,1,1)
        out = self.regressor(f)                    # → (B,2) in roughly [-1,1]
        # map from [-1,1] to [0,1]
        out = (out + 1) * 0.5
        return out