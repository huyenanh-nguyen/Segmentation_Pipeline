#!/usr/bin/env python3
"""
U-Net instance segmentation for microscopy images.

The existing project-specific preprocessing is retained:
- access.txt
- ND2 loading
- Background().darkfield_assignment(...)[0]["raw_minus_dark"]
- TIFF instance annotations
- Trainingpreperation().split_frames_by_key(...)
- Trainingpreperation().spatial_augmentation(...)
- Trainingpreperation().pixelaugmentation(...)

The U-Net predicts:
    channel 0: cell foreground logits
    channel 1: boundary logits

Watershed converts these predictions into individual instance IDs.
Ground-truth instance IDs are preserved and used for instance-level evaluation.

Required packages:
    pip install torch numpy nd2 tifffile scipy scikit-image matplotlib

Run this script from the project directory containing access.txt and DataProcessing/.
"""

import nd2
import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

from skimage.feature import peak_local_max
from skimage.segmentation import watershed

import csv
import random
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from DataProcessing.trainingsdataprocessing import Trainingpreperation
from DataProcessing.background import Background
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

# =============================================================================
# Configuration
# =============================================================================

NUM_AUGMENTATIONS_PER_FRAME = 2

EPOCHS = 300
BATCH_SIZE = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BASE_CHANNELS = 32
SEED = 42

OUTPUT = Path.cwd() / "cellpose_training"
MODEL_DIR = Path.cwd() / "unet_models"
PREDICTION_DIR = MODEL_DIR / "instance_predictions"

# Watershed parameters. These may need tuning for your cell size/density.
CELL_THRESHOLD = 0.50
MIN_DISTANCE = 12
BOUNDARY_WEIGHT = 2.0

# Weight of boundary loss relative to cell loss.
BOUNDARY_LOSS_WEIGHT = 1.0

# Save predicted and ground-truth instance masks for test frames.
SAVE_TEST_PREDICTIONS = True

# Number of workers: 0 is safest for cluster/MPS and custom project objects.
NUM_WORKERS = 0


# =============================================================================
# Reproducibility
# =============================================================================

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Existing data preparation: intentionally kept project-specific
# =============================================================================

with open("access.txt", "r", encoding="utf-8") as file:
    content = [line.strip() for line in file]

if len(content) < 3:
    raise ValueError(
        "access.txt must contain at least three lines: "
        "the expected data path in content[1] and annotation path in content[2]."
    )

data_directory = Path.cwd() / content[1].lstrip("/\\")
datapath = list(data_directory.glob("*.nd2"))

if not datapath:
    raise RuntimeError(f"No ND2 files found in {data_directory}")

data = {}

for nd2_file in datapath:
    # Kept from the existing project pipeline.
    raw = nd2.imread(nd2_file).astype(np.float32)

    # Use the darkfield-corrected image as in the original Cellpose workflow.
    new_img = Background().darkfield_assignment(nd2_file)[0]["raw_minus_dark"]
    corrected = np.asarray(new_img, dtype=np.float32)

    if corrected.ndim != 3:
        raise ValueError(
            f"{nd2_file.name}: expected corrected data with shape "
            f"(frames, height, width), got {corrected.shape}. "
            "If your ND2 has multiple channels, select the intended channel "
            "before adding it to data."
        )

    data[nd2_file.stem] = corrected

# Original ImageJ annotation folders.
annotation_root = Path.cwd() / content[2].lstrip("/\\")
if not annotation_root.exists():
    raise FileNotFoundError(f"Annotation directory does not exist: {annotation_root}")

maskpath = [folder for folder in annotation_root.iterdir() if folder.is_dir()]
mask_files = {folder.name: list(folder.glob("*")) for folder in maskpath}

# Optional conversion from ImageJ ROIs to TIFFs, retained but not run automatically.
# OUTPUT.mkdir(exist_ok=True)
# for name, files in mask_files.items():
#     for roi_path in files:
#         Trainingpreperation().process_rois_to_tiff(
#             str(roi_path), str(OUTPUT), 600, 600, name
#         )

# Load TIFF instance masks. Do NOT binarize them: IDs 1, 2, 3, ... are instances.
annotation = {}
frameindex = {}

if not OUTPUT.exists():
    raise FileNotFoundError(
        f"Annotation TIFF directory does not exist: {OUTPUT}. "
        "Export your instance-labelled TIFF masks there first."
    )

for folder in OUTPUT.iterdir():
    if not folder.is_dir():
        continue

    listholder = {}
    placeholder = []

    for mask_file in folder.glob("*.tif*"):
        match = re.search(
            r"(\d+)\.tif{1,2}$",
            mask_file.name,
            flags=re.IGNORECASE
        )

        if match is None:
            print(f"Skipping non-frame TIFF: {mask_file.name}")
            continue

        frame = int(match.group(1))
        mask = np.squeeze(tiff.imread(mask_file))

        if mask.ndim != 2:
            raise ValueError(
                f"{mask_file} has shape {mask.shape}; expected a 2D mask."
            )

        if not np.issubdtype(mask.dtype, np.integer):
            # Only allow float masks if every value is exactly integer-valued.
            if not np.all(np.isfinite(mask)) or not np.allclose(mask, np.round(mask)):
                raise ValueError(
                    f"{mask_file} contains non-integer label values. "
                    "Instance masks must contain integer IDs."
                )

        mask = mask.astype(np.int32)

        if np.any(mask < 0):
            raise ValueError(f"{mask_file} contains negative instance IDs.")

        listholder[frame] = mask
        placeholder.append(frame)

    if listholder:
        annotation[folder.name] = listholder
        frameindex[folder.name] = sorted(placeholder)

if not annotation:
    raise RuntimeError(f"No TIFF instance masks found below {OUTPUT}.")

# Check image/mask pairing.
for key, frames in frameindex.items():
    if key not in data:
        raise KeyError(
            f"Annotation folder '{key}' has no corresponding ND2 file basename."
        )

    for frame_idx in frames:
        if frame_idx >= len(data[key]):
            raise IndexError(
                f"{key}: mask frame {frame_idx} exceeds ND2 frame range "
                f"(0..{len(data[key]) - 1})."
            )

        if data[key][frame_idx].shape != annotation[key][frame_idx].shape:
            raise ValueError(
                f"{key}, frame {frame_idx}: ND2 image shape "
                f"{data[key][frame_idx].shape} does not match mask shape "
                f"{annotation[key][frame_idx].shape}."
            )

        ids = np.unique(annotation[key][frame_idx])
        print(
            f"Annotation check | {key} | frame {frame_idx} | "
            f"{len(ids) - int(0 in ids)} instances | IDs: {ids[:12]}"
        )


# =============================================================================
# Existing split and augmentation logic
# =============================================================================

train_frames, val_frames, test_frames = (
    Trainingpreperation().split_frames_by_key(frameindex)
)

train_images, train_masks = [], []
val_images, val_masks = [], []
test_images, test_masks = [], []

# Keep sample identity for output filenames and traceability.
val_sample_info = []
test_sample_info = []

prep = Trainingpreperation()


def validate_augmented_instance_mask(original_mask, augmented_mask, context="augmentation"):
    """
    Detect label corruption caused by non-nearest-neighbour interpolation.

    Spatial augmentation must transform instance masks with nearest-neighbour
    interpolation. The transformed mask may lose IDs due to cropping, but it
    must not invent fractional or new IDs.
    """
    aug = np.asarray(augmented_mask)

    if aug.ndim != 2:
        raise ValueError(
            f"{context}: augmented mask must be 2D, got {aug.shape}."
        )

    if not np.all(np.isfinite(aug)):
        raise ValueError(f"{context}: augmented mask contains NaN/Inf.")

    if not np.allclose(aug, np.round(aug)):
        raise ValueError(
            f"{context}: augmentation created non-integer mask values. "
            "Use nearest-neighbour interpolation for masks."
        )

    original_ids = set(np.unique(original_mask).tolist())
    augmented_ids = set(np.unique(np.round(aug).astype(np.int32)).tolist())

    unexpected = augmented_ids - original_ids
    if unexpected:
        raise ValueError(
            f"{context}: augmentation created unexpected instance IDs "
            f"{sorted(unexpected)[:20]}. Check spatial_augmentation(): "
            "mask interpolation must be nearest-neighbour."
        )

    return np.round(aug).astype(np.int32)


for key, frames in train_frames.items():
    for frame_idx in frames:
        original_image = data[key][frame_idx]
        original_mask = annotation[key][frame_idx]

        train_images.append(original_image)
        train_masks.append(original_mask)

        for aug_idx in range(NUM_AUGMENTATIONS_PER_FRAME):
            aug_img, aug_mask = prep.spatial_augmentation(
                original_image,
                original_mask
            )

            aug_mask = validate_augmented_instance_mask(
                original_mask,
                aug_mask,
                context=f"{key}, frame {frame_idx}, augmentation {aug_idx}"
            )

            aug_img = prep.pixelaugmentation(aug_img)

            train_images.append(np.asarray(aug_img, dtype=np.float32))
            train_masks.append(aug_mask)


for key, frames in val_frames.items():
    for frame_idx in frames:
        val_images.append(data[key][frame_idx])
        val_masks.append(annotation[key][frame_idx])
        val_sample_info.append((key, frame_idx))


for key, frames in test_frames.items():
    for frame_idx in frames:
        test_images.append(data[key][frame_idx])
        test_masks.append(annotation[key][frame_idx])
        test_sample_info.append((key, frame_idx))


print(f"Total training samples:   {len(train_images)}")
print(f"Total validation samples: {len(val_images)}")
print(f"Total test samples:       {len(test_images)}")

if not train_images or not val_images:
    raise RuntimeError(
        "Training or validation set is empty. Check split_frames_by_key "
        "and the number of annotations."
    )


# =============================================================================
# Device selection
# =============================================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"NVIDIA GPU (CUDA) is active: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Apple Silicon GPU (MPS) is active.")
else:
    device = torch.device("cpu")
    print("GPU not available; defaulting to CPU.")


# =============================================================================
# Targets and Dataset
# =============================================================================

def make_boundary_target(instance_mask):
    """
    Mark pixels on both sides of a boundary between two different,
    non-background instance IDs.

    Boundaries between a cell and background are not included in this
    boundary target; the cell foreground head learns the outer cell region.
    """
    mask = np.asarray(instance_mask, dtype=np.int32)
    boundary = np.zeros(mask.shape, dtype=bool)

    # Horizontal neighbours.
    diff_x = (
        (mask[:, 1:] != mask[:, :-1])
        & (mask[:, 1:] > 0)
        & (mask[:, :-1] > 0)
    )
    boundary[:, 1:] |= diff_x
    boundary[:, :-1] |= diff_x

    # Vertical neighbours.
    diff_y = (
        (mask[1:, :] != mask[:-1, :])
        & (mask[1:, :] > 0)
        & (mask[:-1, :] > 0)
    )
    boundary[1:, :] |= diff_y
    boundary[:-1, :] |= diff_y

    return boundary.astype(np.float32)


class SegmentationDataset(Dataset):
    def __init__(self, images, instance_masks):
        if len(images) != len(instance_masks):
            raise ValueError("Images and masks must have the same length.")

        self.images = images
        self.instance_masks = instance_masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.asarray(self.images[index], dtype=np.float32)
        instance_mask = np.asarray(self.instance_masks[index])

        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got {image.shape}.")

        if instance_mask.ndim != 2:
            raise ValueError(f"Expected 2D instance mask, got {instance_mask.shape}.")

        if image.shape != instance_mask.shape:
            raise ValueError(f"Image/mask shape mismatch: {image.shape} vs {instance_mask.shape}") # type: ignore[arg-type]

        if not np.issubdtype(instance_mask.dtype, np.integer):
            if not np.all(np.isfinite(instance_mask)) or not np.allclose(
                instance_mask, np.round(instance_mask)
            ):
                raise ValueError("Instance mask contains non-integer values.")

        instance_mask = instance_mask.astype(np.int32)

        if np.any(instance_mask < 0):
            raise ValueError("Instance IDs must be non-negative.")

        # Per-frame robust intensity scaling, retained from the previous code.
        low, high = np.percentile(image, (1, 99))

        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            image = np.zeros_like(image, dtype=np.float32)
        else:
            image = np.clip(
                (image - low) / (high - low),
                0.0,
                1.0
            ).astype(np.float32)

        # Target 1: foreground containing all cells.
        cell_target = (instance_mask > 0).astype(np.float32)

        # Target 2: boundary between touching/different instance IDs.
        boundary_target = make_boundary_target(instance_mask)

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image[None])
        ).float()

        cell_tensor = torch.from_numpy(
            np.ascontiguousarray(cell_target[None])
        ).float()

        boundary_tensor = torch.from_numpy(
            np.ascontiguousarray(boundary_target[None])
        ).float()

        # Keep the original IDs for instance-level validation and comparison.
        instance_tensor = torch.from_numpy(
            np.ascontiguousarray(instance_mask)
        ).long()

        return image_tensor, cell_tensor, boundary_tensor, instance_tensor


# =============================================================================
# U-Net
# =============================================================================

class DoubleConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__() # type: ignore[arg-type]
        self.layers = nn.Sequential( # type: ignore[arg-type]
            nn.Conv2d(channels_in, channels_out, kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out, kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    """
    Two-head U-Net:
        output[:, 0:1] = cell foreground logits
        output[:, 1:2] = inter-instance boundary logits
    """

    def __init__(self, base=BASE_CHANNELS):
        super().__init__()

        self.e1 = DoubleConv(1, base)
        self.e2 = DoubleConv(base, base * 2)
        self.e3 = DoubleConv(base * 2, base * 4)
        self.e4 = DoubleConv(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)

        self.b = DoubleConv(base * 8, base * 16)

        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.d4 = DoubleConv(base * 16, base * 8)

        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.d3 = DoubleConv(base * 8, base * 4)

        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = DoubleConv(base * 4, base * 2)

        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = DoubleConv(base * 2, base)

        # Two output channels: cell and boundary.
        self.out = nn.Conv2d(base, 2, kernel_size=1)

    @staticmethod
    def resize(x, reference):
        if x.shape[-2:] != reference.shape[-2:]:
            x = F.interpolate(
                x,
                size=reference.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
        return x

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))

        b = self.b(self.pool(e4))

        d4 = self.d4(torch.cat(
            [e4, self.resize(self.u4(b), e4)],
            dim=1
        ))

        d3 = self.d3(torch.cat(
            [e3, self.resize(self.u3(d4), e3)],
            dim=1
        ))

        d2 = self.d2(torch.cat(
            [e2, self.resize(self.u2(d3), e2)],
            dim=1
        ))

        d1 = self.d1(torch.cat(
            [e1, self.resize(self.u1(d2), e1)],
            dim=1
        ))

        return self.out(d1)


# =============================================================================
# Loss functions
# =============================================================================

def dice_loss(logits, target, eps=1e-6):
    probabilities = torch.sigmoid(logits)

    intersection = (probabilities * target).sum((1, 2, 3))
    total = probabilities.sum((1, 2, 3)) + target.sum((1, 2, 3))

    dice = (2.0 * intersection + eps) / (total + eps)
    return (1.0 - dice).mean()


def loss_function(
    predictions,
    cell_target,
    boundary_target,
    boundary_loss_weight=BOUNDARY_LOSS_WEIGHT
):
    cell_logits = predictions[:, 0:1]
    boundary_logits = predictions[:, 1:2]

    cell_bce = F.binary_cross_entropy_with_logits(
        cell_logits,
        cell_target
    )
    cell_dice = dice_loss(cell_logits, cell_target)
    cell_loss = cell_bce + cell_dice

    # Compensate moderately for the fact that boundary pixels are sparse.
    positive = boundary_target.sum()
    negative = boundary_target.numel() - positive
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0)

    boundary_bce = F.binary_cross_entropy_with_logits(
        boundary_logits,
        boundary_target,
        pos_weight=pos_weight
    )

    return cell_loss + boundary_loss_weight * boundary_bce


# =============================================================================
# Watershed reconstruction
# =============================================================================


def watershed_instances(
    cell_probability,
    boundary_probability,
    cell_threshold=CELL_THRESHOLD,
    min_distance=MIN_DISTANCE,
    boundary_weight=BOUNDARY_WEIGHT
):
    """
    Convert the two U-Net probability maps into an integer instance-label map.

    Returns:
        int32 array, background=0, instances=1,2,3,...
    """

    cell_probability = np.asarray(
        cell_probability, dtype=np.float32
    )
    boundary_probability = np.asarray(
        boundary_probability, dtype=np.float32
    )

    if cell_probability.shape != boundary_probability.shape:
        raise ValueError(
            "Cell and boundary probability maps must have same shape."
        )

    # 1. Threshold cell probability
    foreground = cell_probability >= cell_threshold
    foreground = np.asarray(foreground, dtype=bool)

    if not np.any(foreground):
        return np.zeros(foreground.shape, dtype=np.int32)

    # 2. Compute distance transform
    distance = np.asarray(
        ndi.distance_transform_edt(foreground),
        dtype=np.float32
    )

    # 3. Construct watershed elevation
    elevation = (
        -distance
        + boundary_weight * boundary_probability
    )

    # 4. Find local maxima as candidate cell centres
    coordinates = peak_local_max(
        distance,
        labels=foreground.astype(np.uint8),
        min_distance=max(1, int(min_distance)),
        exclude_border=False
    )

    # 5. Create marker image
    markers_image = np.zeros(
        foreground.shape,
        dtype=np.int32
    )

    for marker_id, (row, col) in enumerate(coordinates, start=1):
        markers_image[row, col] = marker_id

    # 6. Fallback if no local maxima were found
    if markers_image.max() == 0:
        markers_image, num_features = ndi.label(foreground)

        if num_features == 0:
            return np.zeros(foreground.shape, dtype=np.int32)

    # 7. Watershed segmentation
    labels = watershed(
        elevation,
        markers=markers_image,
        mask=foreground,
        watershed_line=True
    )

    return labels.astype(np.int32)



# =============================================================================
# Instance metrics: IoU matching and Panoptic Quality
# =============================================================================

def instance_metrics(true_mask, pred_mask, iou_threshold=0.5):
    """
    Match predicted and ground-truth instances using Hungarian assignment.

    A match is a true positive when IoU >= iou_threshold.
    Returns TP, FP, FN, precision, recall, and PQ.
    """
    true_mask = np.asarray(true_mask, dtype=np.int32)
    pred_mask = np.asarray(pred_mask, dtype=np.int32)

    true_ids = np.unique(true_mask)
    pred_ids = np.unique(pred_mask)

    true_ids = true_ids[true_ids != 0]
    pred_ids = pred_ids[pred_ids != 0]

    n_true = len(true_ids)
    n_pred = len(pred_ids)

    if n_true == 0 and n_pred == 0:
        return {
            "tp": 0, "fp": 0, "fn": 0,
            "precision": 1.0, "recall": 1.0, "pq": 1.0
        }

    if n_true == 0:
        return {
            "tp": 0, "fp": n_pred, "fn": 0,
            "precision": 0.0, "recall": 1.0, "pq": 0.0
        }

    if n_pred == 0:
        return {
            "tp": 0, "fp": 0, "fn": n_true,
            "precision": 1.0, "recall": 0.0, "pq": 0.0
        }

    # Map arbitrary IDs to compact indices.
    true_index = np.zeros(true_mask.shape, dtype=np.int32)
    pred_index = np.zeros(pred_mask.shape, dtype=np.int32)

    for idx, label_id in enumerate(true_ids, start=1):
        true_index[true_mask == label_id] = idx

    for idx, label_id in enumerate(pred_ids, start=1):
        pred_index[pred_mask == label_id] = idx

    # Pairwise intersections from a contingency table.
    contingency = np.zeros((n_true + 1, n_pred + 1), dtype=np.int64)
    np.add.at(contingency, (true_index.ravel(), pred_index.ravel()), 1)

    intersections = contingency[1:, 1:]
    true_areas = contingency[1:, :].sum(axis=1)
    pred_areas = contingency[:, 1:].sum(axis=0)

    unions = (
        true_areas[:, None]
        + pred_areas[None, :]
        - intersections
    )

    iou = intersections / np.maximum(unions, 1)

    # Hungarian assignment maximizes total IoU.
    rows, cols = linear_sum_assignment(-iou)

    matched_ious = []
    for row, col in zip(rows, cols):
        if iou[row, col] >= iou_threshold:
            matched_ious.append(float(iou[row, col]))

    tp = len(matched_ious)
    fp = n_pred - tp
    fn = n_true - tp

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0

    denominator = tp + 0.5 * fp + 0.5 * fn
    pq = sum(matched_ious) / denominator if denominator else 1.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "pq": pq
    }


def semantic_dice(true_mask, pred_mask, eps=1e-6):
    true_fg = np.asarray(true_mask) > 0
    pred_fg = np.asarray(pred_mask) > 0

    intersection = np.logical_and(true_fg, pred_fg).sum()
    denominator = true_fg.sum() + pred_fg.sum()

    return float((2.0 * intersection + eps) / (denominator + eps))


# =============================================================================
# Prediction helper
# =============================================================================

@torch.no_grad()
def predict_instance_mask(model, image_tensor):
    model.eval()

    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)

    cell_probability = torch.sigmoid(logits[0, 0]).cpu().numpy()
    boundary_probability = torch.sigmoid(logits[0, 1]).cpu().numpy()

    instance_prediction = watershed_instances(
        cell_probability,
        boundary_probability
    )

    return cell_probability, boundary_probability, instance_prediction


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    total_loss = 0.0
    total_semantic_dice = 0.0
    total_pq = 0.0
    total_precision = 0.0
    total_recall = 0.0
    count = 0

    for images, cell_target, boundary_target, true_instances in loader:
        images = images.to(device)
        cell_target = cell_target.to(device)
        boundary_target = boundary_target.to(device)

        predictions = model(images)

        loss = loss_function(
            predictions,
            cell_target,
            boundary_target
        )

        batch_size = images.size(0)

        for batch_idx in range(batch_size):
            cell_probability = torch.sigmoid(
                predictions[batch_idx, 0]
            ).cpu().numpy()

            boundary_probability = torch.sigmoid(
                predictions[batch_idx, 1]
            ).cpu().numpy()

            predicted_instances = watershed_instances(
                cell_probability,
                boundary_probability
            )

            true_instance_mask = true_instances[batch_idx].cpu().numpy()

            total_semantic_dice += semantic_dice(
                true_instance_mask,
                predicted_instances
            )

            metrics = instance_metrics(
                true_instance_mask,
                predicted_instances,
                iou_threshold=0.5
            )

            total_pq += metrics["pq"]
            total_precision += metrics["precision"]
            total_recall += metrics["recall"]

        total_loss += loss.item() * batch_size
        count += batch_size

    if count == 0:
        raise RuntimeError("Evaluation loader is empty.")

    return {
        "loss": total_loss / count,
        "semantic_dice": total_semantic_dice / count,
        "pq": total_pq / count,
        "precision": total_precision / count,
        "recall": total_recall / count
    }


# =============================================================================
# Save visual comparisons and predicted instance masks
# =============================================================================

def save_instance_comparison(
    image,
    true_mask,
    predicted_mask,
    output_path,
    title=""
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input image")

    axes[1].imshow(true_mask, cmap="nipy_spectral", interpolation="nearest")
    axes[1].set_title(
        f"Ground truth instances ({len(np.unique(true_mask)) - int(0 in true_mask)})"
    )

    axes[2].imshow(
        predicted_mask,
        cmap="nipy_spectral",
        interpolation="nearest"
    )
    axes[2].set_title(
        f"Predicted instances "
        f"({len(np.unique(predicted_mask)) - int(0 in predicted_mask)})"
    )

    for ax in axes:
        ax.axis("off")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_predictions(model, images, masks, sample_info, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    model.eval()

    for index, (image, true_mask, info) in enumerate(
        zip(images, masks, sample_info)
    ):
        key, frame_idx = info

        dataset_item = SegmentationDataset([image], [true_mask])[0]
        image_tensor = dataset_item[0].unsqueeze(0)

        cell_prob, boundary_prob, pred_mask = predict_instance_mask(
            model,
            image_tensor
        )

        # Keep the original annotation IDs for direct comparison.
        true_mask = np.asarray(true_mask, dtype=np.int32)

        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
        stem = f"{index:05d}_{safe_key}_frame_{frame_idx:04d}"

        tiff.imwrite(output_dir / f"{stem}_predicted_instances.tif", pred_mask)
        tiff.imwrite(output_dir / f"{stem}_ground_truth_instances.tif", true_mask)
        tiff.imwrite(
            output_dir / f"{stem}_cell_probability.tif",
            cell_prob.astype(np.float32)
        )
        tiff.imwrite(
            output_dir / f"{stem}_boundary_probability.tif",
            boundary_prob.astype(np.float32)
        )

        # Save a normalized input only for convenient visual inspection.
        low, high = np.percentile(np.asarray(image, dtype=np.float32), (1, 99))
        if high > low:
            display_image = np.clip(
                (np.asarray(image, dtype=np.float32) - low) / (high - low),
                0, 1
            )
        else:
            display_image = np.zeros_like(image, dtype=np.float32)

        save_instance_comparison(
            display_image,
            true_mask,
            pred_mask,
            output_dir / f"{stem}_comparison.png",
            title=f"{key} | frame {frame_idx}"
        )

        metrics = instance_metrics(true_mask, pred_mask)
        metrics["sample"] = stem
        metrics["ground_truth_instances"] = len(np.unique(true_mask)) - int(
            0 in true_mask
        )
        metrics["predicted_instances"] = len(np.unique(pred_mask)) - int(
            0 in pred_mask
        )
        metrics["semantic_dice"] = semantic_dice(true_mask, pred_mask)
        rows.append(metrics)

        print(
            f"Saved {stem} | GT instances={metrics['ground_truth_instances']} | "
            f"Predicted instances={metrics['predicted_instances']} | "
            f"PQ={metrics['pq']:.4f}"
        )

    if rows:
        csv_path = output_dir / "instance_metrics.csv"
        fieldnames = [
            "sample", "ground_truth_instances", "predicted_instances",
            "tp", "fp", "fn", "precision", "recall", "pq", "semantic_dice"
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Instance metrics saved to: {csv_path}")


# =============================================================================
# Training
# =============================================================================

def main():
    seed_everything()

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        SegmentationDataset(train_images, train_masks),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        SegmentationDataset(val_images, val_masks),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=NUM_WORKERS
    )

    test_loader = None
    if test_images:
        test_loader = DataLoader(
            SegmentationDataset(test_images, test_masks),
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=NUM_WORKERS
        )

    model = UNet().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Select the best checkpoint by instance-level validation PQ.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=12
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_model_path = MODEL_DIR / "unet_instance_best.pth"
    last_model_path = MODEL_DIR / "unet_instance_last.pth"

    best_val_pq = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        items = 0

        for images, cell_target, boundary_target, _true_instances in train_loader:
            images = images.to(device)
            cell_target = cell_target.to(device)
            boundary_target = boundary_target.to(device)

            optimizer.zero_grad(set_to_none=True)

            predictions = model(images)

            loss = loss_function(
                predictions,
                cell_target,
                boundary_target
            )

            # Backpropagation: calculate gradients.
            loss.backward()

            # Clip gradients to reduce unstable updates.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            # Optimizer: update trainable weights.
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            items += images.size(0)

        train_loss = running_loss / max(items, 1)

        val_metrics = evaluate(model, val_loader)
        scheduler.step(val_metrics["pq"])

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_metrics["loss"],
            "val_semantic_dice": val_metrics["semantic_dice"],
            "val_pq": val_metrics["pq"],
            "base_channels": BASE_CHANNELS,
            "cell_threshold": CELL_THRESHOLD,
            "min_distance": MIN_DISTANCE,
            "boundary_weight": BOUNDARY_WEIGHT
        }

        torch.save(checkpoint, last_model_path)

        if val_metrics["pq"] > best_val_pq:
            best_val_pq = val_metrics["pq"]
            torch.save(checkpoint, best_model_path)
            best_marker = " [BEST]"
        else:
            best_marker = ""

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train loss={train_loss:.4f} | "
            f"val loss={val_metrics['loss']:.4f} | "
            f"semantic Dice={val_metrics['semantic_dice']:.4f} | "
            f"PQ={val_metrics['pq']:.4f} | "
            f"precision={val_metrics['precision']:.4f} | "
            f"recall={val_metrics['recall']:.4f} | "
            f"lr={current_lr:.2e}"
            f"{best_marker}"
        )

    # Load best validation checkpoint before final test evaluation.
    if not best_model_path.exists():
        raise RuntimeError("No best model checkpoint was saved.")

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    print(
        f"\nBest checkpoint: epoch {checkpoint['epoch']} | "
        f"validation PQ={checkpoint['val_pq']:.4f}"
    )

    if test_loader is not None:
        test_metrics = evaluate(model, test_loader)

        print(
            "\nFinal test results | "
            f"loss={test_metrics['loss']:.4f} | "
            f"semantic Dice={test_metrics['semantic_dice']:.4f} | "
            f"PQ={test_metrics['pq']:.4f} | "
            f"precision={test_metrics['precision']:.4f} | "
            f"recall={test_metrics['recall']:.4f}"
        )

        if SAVE_TEST_PREDICTIONS:
            save_predictions(
                model,
                test_images,
                test_masks,
                test_sample_info,
                PREDICTION_DIR
            )
    else:
        print("No test samples were produced by split_frames_by_key().")

    print(f"\nBest model saved to: {best_model_path}")
    print(f"Last model saved to: {last_model_path}")

    if SAVE_TEST_PREDICTIONS and test_images:
        print(f"Test predictions saved to: {PREDICTION_DIR}")


if __name__ == "__main__":
    main()