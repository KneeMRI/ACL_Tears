import torch
import torch.nn as nn
import torchvision.models as models


class MRNetACLModel(nn.Module):
    """
    Baseline MRNet ACL model.

    Supports:
        - Axial
        - Coronal
        - Sagittal
        - Any combination of the three

    Architecture:
        MRI slices
            ↓
        Pretrained ResNet-18
            ↓
        Mean pooling across slices
            ↓
        Concatenate selected plane features
            ↓
        Binary classifier
    """

    def __init__(self, planes):
        super().__init__()

        if not planes:
            raise ValueError("At least one MRI plane must be selected.")

        valid_planes = {"axial", "coronal", "sagittal"}

        if not set(planes).issubset(valid_planes):
            raise ValueError(
                f"Invalid plane. Choose from {valid_planes}"
            )

        self.planes = planes

        # Pretrained ResNet-18
        self.backbone = models.resnet18(weights="DEFAULT")

        # ResNet-18 produces 512-dimensional features
        self.backbone.fc = nn.Identity()

        # One 512-dimensional feature vector per plane
        self.classifier = nn.Linear(
            512 * len(self.planes),
            1
        )

    def extract_plane_features(self, volume):
        """
        Input:
            volume: [B, S, H, W]

        Output:
            [B, 512]
        """

        B, S, H, W = volume.shape

        # Grayscale → 3 channels
        volume = volume.unsqueeze(2)
        volume = volume.repeat(1, 1, 3, 1, 1)

        # [B,S,3,H,W] → [B*S,3,H,W]
        volume = volume.reshape(
            B * S,
            3,
            H,
            W
        )

        # ResNet input size
        volume = nn.functional.interpolate(
            volume,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        # Extract slice-level features
        features = self.backbone(volume)

        # [B*S,512] → [B,S,512]
        features = features.reshape(
            B,
            S,
            512
        )

        # Mean pooling across slices
        features = features.mean(dim=1)

        return features

    def forward(self, volumes):

        plane_features = []

        for plane in self.planes:

            if plane not in volumes:
                raise KeyError(
                    f"Plane '{plane}' was not provided."
                )

            features = self.extract_plane_features(
                volumes[plane]
            )

            plane_features.append(features)

        # Combine features from selected planes
        combined_features = torch.cat(
            plane_features,
            dim=1
        )

        # Binary classification logit
        output = self.classifier(
            combined_features
        )

        return output.squeeze(1)