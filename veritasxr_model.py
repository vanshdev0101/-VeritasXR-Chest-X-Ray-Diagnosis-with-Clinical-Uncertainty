import torch
import torch.nn as nn
import torch.nn.functional as F

class DualPathBlock(nn.Module):
    """A deeper block for each path with residual connection"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        # Residual shortcut
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))


class SqueezeExcitation(nn.Module):
    """Attention — lets the model focus on important channels"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class VeritasXR(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # ── Stem ──────────────────────────────────────────────
        # Large kernel — X-rays have diffuse patterns not sharp edges
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # ── Local Path (small kernels — finds patches) ─────────
        self.local_1 = DualPathBlock(64, 128, kernel_size=3)
        self.local_2 = DualPathBlock(128, 256, kernel_size=3)
        self.local_se = SqueezeExcitation(256)
        self.local_pool = nn.AdaptiveAvgPool2d((4, 4))

        # ── Global Path (large kernels — sees whole lung) ──────
        self.global_1 = DualPathBlock(64, 128, kernel_size=7)
        self.global_2 = DualPathBlock(128, 256, kernel_size=7)
        self.global_se = SqueezeExcitation(256)
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        # ── Learned merge weights ──────────────────────────────
        # This is your core novelty — model learns how much
        # to trust local vs global features
        self.path_weights = nn.Parameter(torch.ones(2) / 2)

        # ── Classifier Head ────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # ── Uncertainty Head ───────────────────────────────────
        # Outputs 0 = certain, 1 = uncertain
        # This is what makes VeritasXR clinically useful
        self.uncertainty_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # ── Weight initialization ──────────────────────────────
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        # Stem — shared feature extraction
        x = self.stem(x)

        # Local path — detail focused
        local = self.local_1(x)
        local = self.local_2(local)
        local = self.local_se(local)
        local = self.local_pool(local)

        # Global path — structure focused
        glob = self.global_1(x)
        glob = self.global_2(glob)
        glob = self.global_se(glob)
        glob = self.global_pool(glob)

        # Learned weighted merge
        w = F.softmax(self.path_weights, dim=0)
        merged = w[0] * local + w[1] * glob

        # Outputs
        verdict = self.classifier(merged)
        uncertainty = self.uncertainty_head(merged)

        return verdict, uncertainty


def get_model():
    return VeritasXR(num_classes=2)