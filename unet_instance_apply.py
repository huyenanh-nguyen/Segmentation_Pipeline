#!/usr/bin/env python3
"""Apply trained two-output U-Net + watershed to every ND2 frame.
Run from the project root containing access.txt and DataProcessing/.
"""
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import re
from pathlib import Path
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import csv



import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nd2
import numpy as np
import tifffile as tiff
import torch
from torch import nn
from torch.nn import functional as F
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment


from DataProcessing.background import Background

CHECKPOINT_PATH = Path.cwd() / "unet_models" / "unet_instance_best.pth"
PREDICTION_DIR = Path.cwd() / "unet_predictions_watershed"
ANNOTATION_DIR = Path.cwd() / "cellpose_training"
BATCH_SIZE = 2
SAVE_PROBABILITY_MAPS = True
SAVE_BINARY_MASKS = True
SAVE_INSTANCE_MASKS = True
SAVE_COMPARISON_FIGURES = True
MAX_COMPARISON_FIGURES = 12
# None means use parameters stored in the trained checkpoint.
CELL_THRESHOLD_OVERRIDE = None
MIN_DISTANCE_OVERRIDE = None
BOUNDARY_WEIGHT_OVERRIDE = None
BOUNDARY_THRESHOLD_OVERRIDE = None
BOUNDARY_SMOOTHING_OVERRIDE = None

BOUNDARY_THRESHOLD = 0.55
BOUNDARY_SMOOTHING = 0.8
BASE_CHANNELS = 32
CELL_THRESHOLD = 0.5
MIN_DISTANCE = 12
BOUNDARY_WEIGHT = 2.0

class DoubleConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                channels_in, channels_out, kernel_size=3,
                padding=1, bias=False
            ),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels_out, channels_out, kernel_size=3,
                padding=1, bias=False
            ),
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

def watershed_instances(
    cell_probability,
    boundary_probability,
    cell_threshold=CELL_THRESHOLD,
    min_distance=MIN_DISTANCE,
    boundary_weight=BOUNDARY_WEIGHT,
    boundary_threshold=BOUNDARY_THRESHOLD,
    boundary_smoothing=BOUNDARY_SMOOTHING,
):
    """
    Convert U-Net cell and boundary probability maps into instance labels.

    Weak predicted boundaries are merged after watershed segmentation.
    Background = 0; instances = 1, 2, 3, ...
    """

    cell_probability = np.asarray(cell_probability, dtype=np.float32)
    boundary_probability = np.asarray(
        boundary_probability, dtype=np.float32
    )

    if cell_probability.shape != boundary_probability.shape:
        raise ValueError(
            "Cell and boundary probability maps must have the same shape."
        )

    if cell_probability.ndim != 2:
        raise ValueError("Probability maps must be 2D.")

    if boundary_smoothing < 0:
        raise ValueError("boundary_smoothing must be non-negative.")

    foreground = cell_probability >= cell_threshold

    if not np.any(foreground):
        return np.zeros(foreground.shape, dtype=np.int32)

    # Smooth the predicted boundary map.
    smooth_boundary = ndi.gaussian_filter(
        boundary_probability,
        sigma=boundary_smoothing,
    )

    # Distance transform inside the foreground.
    distance = ndi.distance_transform_edt(foreground)

    # Find candidate cell centres.
    coordinates = peak_local_max(
        distance,
        labels=foreground.astype(np.uint8),
        min_distance=max(1, int(min_distance)),
        exclude_border=False,
    )

    markers_image = np.zeros(foreground.shape, dtype=np.int32)

    for marker_id, (row, col) in enumerate(coordinates, start=1):
        markers_image[row, col] = marker_id

    # Fallback if no local maxima were found.
    if markers_image.max() == 0:
        markers_image, _ = ndi.label(foreground)

    # Combine distance and predicted boundary information.
    elevation = -distance + boundary_weight * smooth_boundary

    # Initial watershed segmentation.
    labels = watershed(
        elevation,
        markers=markers_image,
        mask=foreground,
        watershed_line=False,
    ).astype(np.int32)

    # ---------------------------------------------------------
    # Merge neighbouring instances separated by weak boundaries
    # ---------------------------------------------------------

    parent = np.arange(labels.max() + 1, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:
            parent[root_b] = root_a

    # Collect boundary scores for neighbouring instance pairs.
    pair_scores = {}

    def collect_pairs(label_a, label_b, boundary_a, boundary_b):
        valid = (
            (label_a > 0)
            & (label_b > 0)
            & (label_a != label_b)
        )

        if not np.any(valid):
            return

        ids_a = label_a[valid]
        ids_b = label_b[valid]

        scores = np.maximum(
            boundary_a[valid],
            boundary_b[valid],
        )

        for a, b, score in zip(ids_a, ids_b, scores):
            pair = tuple(sorted((int(a), int(b))))
            pair_scores.setdefault(pair, []).append(float(score))

    # Horizontal neighbours.
    collect_pairs(
        labels[:, :-1],
        labels[:, 1:],
        smooth_boundary[:, :-1],
        smooth_boundary[:, 1:],
    )

    # Vertical neighbours.
    collect_pairs(
        labels[:-1, :],
        labels[1:, :],
        smooth_boundary[:-1, :],
        smooth_boundary[1:, :],
    )

    # Merge regions whose shared boundary is weak.
    for (label_a, label_b), scores in pair_scores.items():
        mean_boundary_score = np.mean(scores)

        if mean_boundary_score < boundary_threshold:
            union(label_a, label_b)

    # ---------------------------------------------------------
    # Relabel instances consecutively: 1, 2, 3, ...
    # ---------------------------------------------------------

    merged = np.zeros_like(labels, dtype=np.int32)
    root_to_new_id = {}

    for old_id in np.unique(labels):
        if old_id == 0:
            continue

        root = find(int(old_id))

        if root not in root_to_new_id:
            root_to_new_id[root] = len(root_to_new_id) + 1

        merged[labels == old_id] = root_to_new_id[root]

    return merged

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


def device_for_inference():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU (CUDA/ROCm): {torch.cuda.get_device_name(0)}")
        return device
    if torch.backends.mps.is_available():
        print("GPU (Apple MPS)")
        return torch.device("mps")
    print("CPU (no GPU detected)")
    return torch.device("cpu")


def normalize(image):
    """Exactly the 1st–99th percentile per-frame normalization from training."""
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, (1, 99))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0, 1).astype(np.float32)


def load_tiff_annotations():
    annotations = {}
    if not ANNOTATION_DIR.exists():
        return annotations
    for folder in ANNOTATION_DIR.iterdir():
        if not folder.is_dir():
            continue
        masks = {}
        for mask_file in folder.glob("*.tif*"):
            match = re.search(r"(\d+)\.tif{1,2}$", mask_file.name, flags=re.IGNORECASE)
            if match:
                mask = np.squeeze(tiff.imread(mask_file))
                if mask.ndim != 2 or not np.all(np.isfinite(mask)):
                    raise ValueError(f"Invalid instance mask: {mask_file}")
                if not np.allclose(mask, np.round(mask)):
                    raise ValueError(f"Non-integer instance IDs in {mask_file}")
                masks[int(match.group(1))] = mask.astype(np.int32)
        if masks:
            annotations[folder.name] = masks
    return annotations


def save_comparison(image, ground_truth, prediction, path, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Darkfield-corrected input (normalized)")
    axes[1].imshow(ground_truth, cmap="nipy_spectral", interpolation="nearest")
    axes[1].set_title(f"Ground truth ({np.unique(ground_truth[ground_truth > 0]).size} cells)")
    axes[2].imshow(prediction, cmap="nipy_spectral", interpolation="nearest")
    axes[2].set_title(f"U-Net + watershed ({np.unique(prediction[prediction > 0]).size} cells)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    device = device_for_inference()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = UNet(base=int(checkpoint.get("base_channels", BASE_CHANNELS))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    cell_threshold = (CELL_THRESHOLD_OVERRIDE if CELL_THRESHOLD_OVERRIDE is not None
                      else checkpoint.get("cell_threshold", CELL_THRESHOLD))
    min_distance = (MIN_DISTANCE_OVERRIDE if MIN_DISTANCE_OVERRIDE is not None
                    else checkpoint.get("min_distance", MIN_DISTANCE))
    boundary_weight = (BOUNDARY_WEIGHT_OVERRIDE if BOUNDARY_WEIGHT_OVERRIDE is not None
                       else checkpoint.get("boundary_weight", BOUNDARY_WEIGHT))
    boundary_threshold = (
                        BOUNDARY_THRESHOLD_OVERRIDE
                        if BOUNDARY_THRESHOLD_OVERRIDE is not None
                        else checkpoint.get("boundary_threshold", BOUNDARY_THRESHOLD)
                            )

    boundary_smoothing = (
        BOUNDARY_SMOOTHING_OVERRIDE
        if BOUNDARY_SMOOTHING_OVERRIDE is not None
        else checkpoint.get("boundary_smoothing", BOUNDARY_SMOOTHING)
    )
    print(
    f"Loaded epoch {checkpoint.get('epoch', '?')} | "
    f"cell_threshold={cell_threshold} | "
    f"min_distance={min_distance} | "
    f"boundary_weight={boundary_weight} | "
    f"boundary_threshold={boundary_threshold} | "
    f"boundary_smoothing={boundary_smoothing}"
            )

    with open("access.txt", encoding="utf-8") as file:
        content = [line.strip() for line in file]
    if len(content) < 2:
        raise ValueError("access.txt must specify ND2 folder in content[1]")
    data_directory = Path.cwd() / content[1].lstrip("/\\")
    nd2_files = sorted(data_directory.glob("*.nd2"))
    if not nd2_files:
        raise FileNotFoundError(f"No ND2 files found in {data_directory}")
    annotations = load_tiff_annotations()
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    comparisons_saved = 0

    with torch.inference_mode():
        for nd2_file in nd2_files:
            recording = nd2_file.stem
            print(f"\nProcessing {nd2_file.name}", flush=True)
            # Use exactly the darkfield correction used for training.
            corrected = Background().darkfield_assignment(nd2_file)[0]["raw_minus_dark"]
            frames = np.asarray(corrected, dtype=np.float32)
            if frames.ndim == 2:
                frames = frames[None]
            if frames.ndim == 4 and frames.shape[1] == 1:
                frames = frames[:, 0]
            if frames.ndim != 3:
                raise ValueError(f"{nd2_file.name}: expected (T,H,W), got {frames.shape}. "
                                 "Select the same single channel as during training.")
            recording_output = PREDICTION_DIR / recording
            recording_output.mkdir(parents=True, exist_ok=True)
            for start in range(0, len(frames), BATCH_SIZE):
                stop = min(start + BATCH_SIZE, len(frames))
                images = np.stack([normalize(frame) for frame in frames[start:stop]])
                tensor = torch.from_numpy(images[:, None]).to(device)
                logits = model(tensor)
                if logits.shape[1] != 2:
                    raise ValueError(f"Expected two output channels, got {logits.shape}")
                probabilities = torch.sigmoid(logits).cpu().numpy()
                for local_idx, probability_pair in enumerate(probabilities):
                    frame_idx = start + local_idx
                    cell_probability, boundary_probability = probability_pair
                    binary_mask = cell_probability >= cell_threshold
                    instance_mask = watershed_instances(
                                    cell_probability,
                                    boundary_probability,
                                    cell_threshold=cell_threshold,
                                    min_distance=min_distance,
                                    boundary_weight=boundary_weight,
                                    boundary_threshold=boundary_threshold,
                                    boundary_smoothing=boundary_smoothing,
                                ) 
                    stem = f"frame_{frame_idx:04d}"
                    if SAVE_PROBABILITY_MAPS:
                        tiff.imwrite(recording_output / f"{stem}_cell_probability.tif",
                                     cell_probability.astype(np.float32))
                        tiff.imwrite(recording_output / f"{stem}_boundary_probability.tif",
                                     boundary_probability.astype(np.float32))
                    if SAVE_BINARY_MASKS:
                        tiff.imwrite(recording_output / f"{stem}_binary.tif", binary_mask.astype(np.uint8))
                    if SAVE_INSTANCE_MASKS:
                        tiff.imwrite(recording_output / f"{stem}_instances.tif", instance_mask.astype(np.int32))
                    ground_truth = annotations.get(recording, {}).get(frame_idx)
                    if ground_truth is not None:
                        if ground_truth.shape != instance_mask.shape:
                            raise ValueError(f"{recording} frame {frame_idx}: GT {ground_truth.shape} "
                                             f"!= prediction {instance_mask.shape}")
                        metrics = instance_metrics(ground_truth, instance_mask)
                        metrics.update(recording=recording, frame=frame_idx,
                                       semantic_dice=semantic_dice(ground_truth, instance_mask),
                                       gt_instances=np.unique(ground_truth[ground_truth > 0]).size,
                                       pred_instances=np.unique(instance_mask[instance_mask > 0]).size)
                        rows.append(metrics)
                        if SAVE_COMPARISON_FIGURES and comparisons_saved < MAX_COMPARISON_FIGURES:
                            save_comparison(images[local_idx], ground_truth, instance_mask,
                                            recording_output / f"{stem}_comparison.png",
                                            f"{recording} | frame {frame_idx} | PQ {metrics['pq']:.3f}")
                            comparisons_saved += 1
                print(f"  Frames {start}–{stop-1} / {len(frames)-1}", flush=True)

    if rows:
        output = PREDICTION_DIR / "annotated_instance_metrics.csv"
        fields = ["recording", "frame", "gt_instances", "pred_instances", "tp", "fp",
                  "fn", "precision", "recall", "pq", "semantic_dice"]
        with open(output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Evaluated {len(rows)} annotated frames | mean PQ={np.mean([r['pq'] for r in rows]):.4f} "
              f"| mean Dice={np.mean([r['semantic_dice'] for r in rows]):.4f}")
        print(f"Metrics saved: {output}")
    else:
        print("No matching manual annotations found; exported predictions only.")
    print(f"Done. All predictions: {PREDICTION_DIR}")


if __name__ == "__main__":
    main()
