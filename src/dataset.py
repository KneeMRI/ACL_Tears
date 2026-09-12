import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MRNetACLDataset(Dataset):
    """
    MRNet ACL dataset.

    Loads the labeled MRI studies and returns:
        - sagittal volume
        - coronal volume
        - ACL label

    Each volume has shape:
        [number_of_slices, 256, 256]
    """

    def __init__(self, base_path, split="train", planes=("sagittal", "coronal")):
        self.base_path = base_path
        self.split = split
        self.planes = planes

        # Select the appropriate ACL label file
        if split == "train":
            label_file = os.path.join(base_path, "train_acl.csv")
        elif split == "valid":
            label_file = os.path.join(base_path, "valid_acl.csv")
        else:
            raise ValueError("split must be 'train' or 'valid'")

        # Load labels
        labels = pd.read_csv(label_file)

        # The CSV contains two columns: study ID and ACL label
        labels.columns = ["id", "acl"]

        labels["id"] = pd.to_numeric(
            labels["id"], errors="coerce"
        )
        labels["acl"] = pd.to_numeric(
            labels["acl"], errors="coerce"
        )

        # Remove invalid rows
        labels = labels.dropna(subset=["id", "acl"])

        labels["id"] = labels["id"].astype(int)
        labels["acl"] = labels["acl"].astype(int)

        # Remove the unlabeled study
        if split == "train":
            labels = labels[labels["id"] != 0]
        elif split == "valid":
            labels = labels[labels["id"] != 1130]

        self.labels = labels.reset_index(drop=True)

        # Store study IDs
        self.exam_ids = self.labels["id"].tolist()

        print(
            f"{split.upper()} dataset: "
            f"{len(self.exam_ids)} labeled studies"
        )

    def __len__(self):
        return len(self.exam_ids)

    def __getitem__(self, index):

        # Study ID
        exam_id = self.exam_ids[index]

        # ACL label
        label = int(self.labels.iloc[index]["acl"])

        volumes = {}

        # Load requested MRI planes
        for plane in self.planes:

            file_path = os.path.join(
                self.base_path,
                self.split,
                plane,
                f"{exam_id:04d}.npy"
            )

            # Check that the file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"MRI file not found:\n{file_path}"
                )

            # Load MRI volume
            volume = np.load(file_path).astype(np.float32)

            # Normalize pixel values from [0, 255] to [0, 1]
            volume = volume / 255.0

            # Convert NumPy → PyTorch tensor
            volumes[plane] = torch.from_numpy(volume)

        return volumes, torch.tensor(
            label,
            dtype=torch.float32
        )


def collate_fn(batch):
    """
    Custom collate function.

    MRI studies have different numbers of slices.
    We pad volumes within a batch to the largest slice count.
    """

    batch_volumes, batch_labels = zip(*batch)

    planes = batch_volumes[0].keys()

    output = {}

    for plane in planes:

        volumes = [
            item[plane]
            for item in batch_volumes
        ]

        # Find largest number of slices in this batch
        max_slices = max(
            volume.shape[0]
            for volume in volumes
        )

        padded_volumes = []

        for volume in volumes:

            num_slices = volume.shape[0]

            if num_slices < max_slices:

                padding = torch.zeros(
                    max_slices - num_slices,
                    volume.shape[1],
                    volume.shape[2],
                    dtype=volume.dtype
                )

                volume = torch.cat(
                    [volume, padding],
                    dim=0
                )

            padded_volumes.append(volume)

        output[plane] = torch.stack(
            padded_volumes
        )

    labels = torch.stack(batch_labels)

    return output, labels