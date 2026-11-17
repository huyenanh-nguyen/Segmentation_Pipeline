#!/usr/bin/env python3
"""
Temporal U-Net + optical flow + marker-controlled watershed + track IDs.

Run from the project root:
    python run_tracking.py

Requires the existing DataProcessing/background.py and U-Net checkpoint.
"""

from pathlib import Path
import csv
import re
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, find_boundaries
from scipy.optimize import linear_sum_assignment
import cv2
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

from DataProcessing.background import Background


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

CHECKPOINT_PATH = Path.cwd() / "unet_models" / "unet_instance_best.pth"
OUTPUT_DIR = Path.cwd() / "unet_predictions_tracked"

# Project folders. No access.txt is used:
#   Path.cwd()/Data  -> ND2 recordings (plus unrelated files such as darkfield TIFFs)
#   Path.cwd()/masks -> original ImageJ ROI/ZIP annotation files
DATA_DIR = Path.cwd() / "Data"
ROI_DIR = Path.cwd() / "masks"

# Optional: exported TIFF masks from the original training preparation.
# Tracking inference does not require these TIFFs. If this folder exists and
# contains recording subfolders with frame TIFFs, they are loaded for validation.
ANNOTATION_TIFF_DIR = Path.cwd() / "cellpose_training"

BASE_CHANNELS = 32
CELL_THRESHOLD = 0.50
MIN_DISTANCE = 12
BOUNDARY_WEIGHT = 2.0
BOUNDARY_SMOOTHING = 0.8

CHECKPOINT_EVERY = 50
PAUSE_AT_CHECKPOINTS = False  # True only for interactive terminal runs

FLOW_PYR_SCALE = 0.5
FLOW_LEVELS = 3
FLOW_WINSIZE = 21
FLOW_ITERATIONS = 3
FLOW_POLY_N = 5
FLOW_POLY_SIGMA = 1.2

MIN_WARPED_SEED_PIXELS = 2

# Re-entry matching is heuristic. It only considers old and new detections
# near an image border, and compares area and mean intensity.
REIDENTIFICATION_ENABLED = True
REID_BORDER_MARGIN = 35
REID_MAX_COST = 0.45
REID_MAX_AREA_RATIO = 3.0
REID_MAX_INTENSITY_RATIO = 3.0

SAVE_PROBABILITY_MAPS = False
SAVE_BINARY_MASKS = True


# ---------------------------------------------------------------------
# SAME TWO-OUTPUT U-NET ARCHITECTURE AS THE INFERENCE SCRIPT
# ---------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
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
        self.out = nn.Conv2d(base, 2, 1)

    @staticmethod
    def resize(x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:],
                              mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.b(self.pool(e4))
        d4 = self.d4(torch.cat([e4, self.resize(self.u4(b), e4)], 1))
        d3 = self.d3(torch.cat([e3, self.resize(self.u3(d4), e3)], 1))
        d2 = self.d2(torch.cat([e2, self.resize(self.u2(d3), e2)], 1))
        d1 = self.d1(torch.cat([e1, self.resize(self.u1(d2), e1)], 1))
        return self.out(d1)


def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA/ROCm:", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = UNet(base=int(ckpt.get("base_channels", BASE_CHANNELS))).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, device


# ---------------------------------------------------------------------
# INPUT AND PREPROCESSING
# ---------------------------------------------------------------------

def normalize(image):
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image, (1, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0, 1).astype(np.float32)


def find_nd2_files():
    """
    Find ND2 recordings only in the project Data folder.

    Other files may be present in Data (e.g. darkfield TIFFs); they are
    deliberately ignored. No access.txt is read.
    """
    data_directory = DATA_DIR

    if not data_directory.is_dir():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_directory}. "
            "Run this script from the project root containing the Data folder."
        )

    files = sorted(
        path for path in data_directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".nd2"
    )

    if not files:
        raise FileNotFoundError(
            f"No .nd2 recordings found directly inside: {data_directory}"
        )

    print(f"Found {len(files)} ND2 recording(s) in: {data_directory}")
    for path in files:
        print(f"  - {path.name}")

    return files


def load_recording_data(nd2_files):
    """
    Return data[recording_stem] = darkfield-corrected (T,H,W) float32.

    Only the ND2 paths returned by find_nd2_files() are passed to nd2.imread
    and Background().darkfield_assignment(). Other files in Data are ignored.
    """
    data = {}
    background = Background()

    for file in nd2_files:
        print(f"Loading ND2 recording: {file.name}")

        # Preserve the original data-loading step.
        raw = nd2.imread(file).astype(np.float32)

        # Use the corrected image sequence, as in the original Cellpose code.
        corrected = background.darkfield_assignment(file)[0]["raw_minus_dark"]
        frames = np.asarray(corrected, dtype=np.float32)

        if frames.ndim == 2:
            frames = frames[None, :, :]
        elif frames.ndim == 4 and frames.shape[1] == 1:
            frames = frames[:, 0, :, :]

        if frames.ndim != 3:
            raise ValueError(
                f"{file.name}: corrected data has shape {frames.shape}; "
                "expected (T,H,W). Select the same channel used for U-Net training."
            )

        data[file.stem] = frames
        del raw

        print(f"  {file.stem}: {frames.shape}, dtype={frames.dtype}")

    return data


def collect_roi_source_files():
    """
    Collect original ImageJ annotation files from Path.cwd()/masks.

    Expected structure:
        masks/
            recording_A/
                *.roi or *.zip
            recording_B/
                *.roi or *.zip

    These source files are not converted during tracking. They are retained
    as annotation references; the temporal tracking algorithm creates IDs
    from model predictions.
    """
    roi_directory = ROI_DIR

    if not roi_directory.is_dir():
        raise FileNotFoundError(
            f"ROI directory does not exist: {roi_directory}. "
            "Expected a masks folder next to run_tracking.py."
        )

    maskpath = [
        folder for folder in roi_directory.iterdir()
        if folder.is_dir()
    ]
    mask_files = {
        folder.name: sorted(folder.glob("*"))
        for folder in maskpath
    }

    print(f"Found {len(mask_files)} ROI annotation folder(s) in: {roi_directory}")
    for name, files in mask_files.items():
        print(f"  - {name}: {len(files)} file(s)")

    return roi_directory, mask_files


def load_optional_tiff_annotations(data):
    """
    Optionally load already-exported TIFF masks from ANNOTATION_TIFF_DIR.

    This is only for validation against the recordings. The tracking inference
    does not depend on these TIFFs. Original ROI/ZIP files remain in masks/.
    """
    if not ANNOTATION_TIFF_DIR.is_dir():
        print(
            f"No exported TIFF annotation directory found at "
            f"{ANNOTATION_TIFF_DIR}; skipping TIFF annotation validation."
        )
        return {}, {}

    annotation = {}
    frameindex = {}

    for folder in ANNOTATION_TIFF_DIR.iterdir():
        if not folder.is_dir():
            continue

        listholder = {}
        placeholder = []

        for mask_file in folder.glob("*.tif*"):
            match = re.search(
                r"(\\d+)\\.tif{1,2}$",
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

            listholder[frame] = (mask > 0).astype(np.float32)
            placeholder.append(frame)

        if listholder:
            annotation[folder.name] = listholder
            frameindex[folder.name] = sorted(placeholder)

    # Validate only if TIFF annotations were found.
    for key, frames in frameindex.items():
        if key not in data:
            raise KeyError(
                f"TIFF annotation folder '{key}' has no matching ND2 file "
                f"basename. Expected folder name to match file.stem."
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
                    f"{data[key][frame_idx].shape} does not match TIFF mask "
                    f"shape {annotation[key][frame_idx].shape}."
                )

    if annotation:
        print(
            f"Loaded optional TIFF annotations for {len(annotation)} recording(s) "
            f"from {ANNOTATION_TIFF_DIR}"
        )
    else:
        print(
            f"No frame TIFF annotations found in {ANNOTATION_TIFF_DIR}; "
            "continuing without annotation validation."
        )

    return annotation, frameindex


# ---------------------------------------------------------------------
# OPTICAL FLOW: WARP PREVIOUS TRACK LABELS INTO THE CURRENT FRAME
# ---------------------------------------------------------------------

def to_uint8(image):
    lo, hi = np.percentile(image, (1, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    return (np.clip((image - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def warp_previous_labels(previous_image, current_image, previous_labels):
    prev8 = to_uint8(previous_image)
    curr8 = to_uint8(current_image)

    # Contiguous single-channel uint8 inputs are expected by OpenCV.
    curr8 = np.ascontiguousarray(curr8, dtype=np.uint8)
    prev8 = np.ascontiguousarray(prev8, dtype=np.uint8)

    # Use OpenCV's standard None initial-flow argument. getattr avoids a
    # conflicting overload in some generated Pylance OpenCV stubs.
    farneback = getattr(cv2, "calcOpticalFlowFarneback")
    flow = farneback(
        curr8, prev8, None,
        float(FLOW_PYR_SCALE),
        int(FLOW_LEVELS),
        int(FLOW_WINSIZE),
        int(FLOW_ITERATIONS),
        int(FLOW_POLY_N),
        float(FLOW_POLY_SIGMA),
        0,
    )
    h, w = current_image.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    map_x = xx + flow[..., 0]
    map_y = yy + flow[..., 1]
    warped = cv2.remap(
        previous_labels.astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return np.rint(warped).astype(np.int32)


# ---------------------------------------------------------------------
# TEMPORAL MARKER-CONTROLLED WATERSHED
# ---------------------------------------------------------------------

def temporal_watershed(cell_prob, boundary_prob, warped_labels, next_id):
    foreground = cell_prob >= CELL_THRESHOLD
    if not np.any(foreground):
        return np.zeros(foreground.shape, np.int32), [], next_id

    smooth_boundary = ndi.gaussian_filter(
        boundary_prob.astype(np.float32), sigma=BOUNDARY_SMOOTHING
    )
    distance = ndi.distance_transform_edt(foreground)
    markers = np.zeros(foreground.shape, dtype=np.int32)

    # Preserve previous track IDs wherever warped labels overlap current
    # U-Net foreground.
    if warped_labels is not None:
        for old_id in np.unique(warped_labels):
            if old_id <= 0:
                continue
            region = (warped_labels == old_id) & foreground
            if int(region.sum()) >= MIN_WARPED_SEED_PIXELS:
                markers[region] = int(old_id)

    new_ids = []
    coords = peak_local_max(
        distance,
        labels=foreground.astype(np.uint8),
        min_distance=max(1, int(MIN_DISTANCE)),
        exclude_border=False,
    )

    # Add seeds for foreground regions not already represented by a warped
    # track marker. Avoid placing a new seed immediately beside an old seed.
    for row, col in coords:
        r0, r1 = max(0, row - 2), min(foreground.shape[0], row + 3)
        c0, c1 = max(0, col - 2), min(foreground.shape[1], col + 3)
        if np.any(markers[r0:r1, c0:c1] > 0):
            continue
        markers[row, col] = next_id
        new_ids.append(next_id)
        next_id += 1

    if not np.any(markers):
        components, count = ndi.label(foreground)
        for component in range(1, count + 1):
            markers[components == component] = next_id
            new_ids.append(next_id)
            next_id += 1

    elevation = -distance + BOUNDARY_WEIGHT * smooth_boundary
    labels = watershed(
        elevation, markers=markers, mask=foreground, watershed_line=False
    ).astype(np.int32)

    new_ids = [tid for tid in new_ids if np.any(labels == tid)]
    return labels, new_ids, next_id


# ---------------------------------------------------------------------
# TRACK MANAGER AND RE-IDENTIFICATION
# ---------------------------------------------------------------------

def get_measurements(image, mask):
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return None
    return {
        "x": float(xx.mean()),
        "y": float(yy.mean()),
        "area": int(len(xx)),
        "mean_intensity": float(image[mask].mean()),
    }


def near_border(x, y, shape):
    h, w = shape
    return (x < REID_BORDER_MARGIN or x >= w - REID_BORDER_MARGIN
            or y < REID_BORDER_MARGIN or y >= h - REID_BORDER_MARGIN)


class TrackManager:
    def __init__(self):
        self.next_id = 1
        self.tracks = {}
        self.observations = []

    def reidentify(self, labels, image, frame_index, new_ids):
        if not REIDENTIFICATION_ENABLED or not new_ids:
            return labels

        h, w = labels.shape
        dormant = [
            tid for tid, t in self.tracks.items()
            if t["state"] == "lost"
            and t["last_seen"] < frame_index - 1
            and near_border(t["x"], t["y"], labels.shape)
        ]
        candidates = []
        for new_id in new_ids:
            m = labels == new_id
            d = get_measurements(image, m)
            if d and near_border(d["x"], d["y"], labels.shape):
                candidates.append((new_id, d))

        if not dormant or not candidates:
            return labels

        costs = np.full((len(dormant), len(candidates)), 1e6, np.float32)
        for i, old_id in enumerate(dormant):
            old = self.tracks[old_id]
            for j, (_, new) in enumerate(candidates):
                area_ratio = max(
                    new["area"] / max(1, old["area"]),
                    old["area"] / max(1, new["area"])
                )
                intensity_ratio = max(
                    (abs(new["mean_intensity"]) + 1e-5) /
                    (abs(old["mean_intensity"]) + 1e-5),
                    (abs(old["mean_intensity"]) + 1e-5) /
                    (abs(new["mean_intensity"]) + 1e-5)
                )
                if area_ratio > REID_MAX_AREA_RATIO:
                    continue
                if intensity_ratio > REID_MAX_INTENSITY_RATIO:
                    continue
                costs[i, j] = (
                    0.65 * min(1.0, abs(np.log(area_ratio))) +
                    0.35 * min(1.0, abs(np.log(intensity_ratio)))
                )

        rows, cols = linear_sum_assignment(costs)
        remap = {}
        for r, c in zip(rows, cols):
            if costs[r, c] <= REID_MAX_COST:
                remap[candidates[c][0]] = dormant[r]

        if not remap:
            return labels

        result = labels.copy()
        original = labels.copy()
        for new_id, old_id in remap.items():
            result[original == new_id] = old_id
        return result

    def update(self, labels, image, frame_index, new_ids):
        labels = self.reidentify(labels, image, frame_index, new_ids)
        present = [int(x) for x in np.unique(labels) if x > 0]

        for tid, info in self.tracks.items():
            if tid not in present:
                info["state"] = "lost"

        for tid in present:
            d = get_measurements(image, labels == tid)
            if tid not in self.tracks:
                self.tracks[tid] = {
                    "first_seen": frame_index, "observations": 0
                }
            self.tracks[tid].update(d)
            self.tracks[tid]["last_seen"] = frame_index
            self.tracks[tid]["state"] = "active"
            self.tracks[tid]["observations"] += 1
            self.observations.append({
                "frame": frame_index, "track_id": tid,
                "centroid_x": d["x"], "centroid_y": d["y"],
                "area_pixels": d["area"],
                "mean_intensity": d["mean_intensity"],
                "status": "active",
            })
        return labels

    def write_csvs(self, folder):
        with (folder / "track_observations.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            fields = [
                "frame", "track_id", "centroid_x", "centroid_y",
                "area_pixels", "mean_intensity", "status"
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.observations)

        rows = []
        for tid, t in sorted(self.tracks.items()):
            rows.append({
                "track_id": tid,
                "first_seen": t.get("first_seen"),
                "last_seen": t.get("last_seen"),
                "observations": t.get("observations", 0),
                "final_state": t.get("state"),
                "last_area_pixels": t.get("area"),
                "last_mean_intensity": t.get("mean_intensity"),
            })
        with (folder / "track_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            fields = [
                "track_id", "first_seen", "last_seen", "observations",
                "final_state", "last_area_pixels", "last_mean_intensity"
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------
# REVIEW FIGURES AND RUNNER
# ---------------------------------------------------------------------

def save_checkpoint_figure(image, cell_prob, boundary_prob, labels,
                           output_path, recording, frame_index):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    axes[1].imshow(cell_prob, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("Cell probability")
    axes[2].imshow(boundary_prob, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("Boundary probability")
    axes[3].imshow(image, cmap="gray")
    edge = find_boundaries(labels, mode="outer")
    axes[3].imshow(np.ma.masked_where(~edge, edge), cmap="autumn", alpha=0.95)
    axes[3].set_title(f"Tracked instances: {len(np.unique(labels[labels > 0]))}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{recording} | frame {frame_index}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_recording(nd2_path, frames, model, device):
    recording = nd2_path.stem
    print(f"\nProcessing {recording}", flush=True)

    out = OUTPUT_DIR / recording
    mask_dir = out / "instance_masks"
    review_dir = out / "review_checkpoints"
    mask_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    tracker = TrackManager()
    previous_image = None
    previous_labels = None

    with torch.inference_mode():
        for frame_index, raw in enumerate(frames):
            image = normalize(raw)
            tensor = torch.from_numpy(image[None, None]).to(device)
            logits = model(tensor)
            if logits.shape[1] != 2:
                raise ValueError(f"Expected two U-Net outputs, got {logits.shape}")
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            cell_prob, boundary_prob = probs[0], probs[1]

            warped = None
            if previous_image is not None:
                warped = warp_previous_labels(
                    previous_image, image, previous_labels
                )

            labels, new_ids, tracker.next_id = temporal_watershed(
                cell_prob, boundary_prob, warped, tracker.next_id
            )
            labels = tracker.update(labels, image, frame_index, new_ids)
            tracker.next_id = max(
                tracker.next_id,
                int(labels.max()) + 1 if labels.size else 1
            )

            tiff.imwrite(
                mask_dir / f"frame_{frame_index:04d}_instances.tif",
                labels.astype(np.uint32)
            )
            if SAVE_BINARY_MASKS:
                tiff.imwrite(
                    out / f"frame_{frame_index:04d}_binary.tif",
                    (cell_prob >= CELL_THRESHOLD).astype(np.uint8)
                )
            if SAVE_PROBABILITY_MAPS:
                tiff.imwrite(
                    out / f"frame_{frame_index:04d}_cell_probability.tif",
                    cell_prob.astype(np.float32)
                )
                tiff.imwrite(
                    out / f"frame_{frame_index:04d}_boundary_probability.tif",
                    boundary_prob.astype(np.float32)
                )

            previous_image = image.copy()
            previous_labels = labels.copy()

            is_checkpoint = (
                (frame_index + 1) % CHECKPOINT_EVERY == 0
                or frame_index == len(frames) - 1
            )
            if is_checkpoint:
                review_path = review_dir / f"checkpoint_frame_{frame_index:04d}.png"
                save_checkpoint_figure(
                    image, cell_prob, boundary_prob, labels,
                    review_path, recording, frame_index
                )
                print(f"Review checkpoint saved: {review_path}", flush=True)
                if PAUSE_AT_CHECKPOINTS:
                    answer = input(
                        "Inspect checkpoint. Enter = continue, q = stop: "
                    ).strip().lower()
                    if answer == "q":
                        print("Stopped by user.")
                        break

            if (frame_index + 1) % 25 == 0 or frame_index == len(frames) - 1:
                print(
                    f"  {frame_index + 1}/{len(frames)} frames | "
                    f"{len(tracker.tracks)} tracks seen",
                    flush=True
                )

    tracker.write_csvs(out)
    print(f"Saved recording results to: {out}", flush=True)


def main():
    # Project layout:
    #   ./Data  = ND2 recordings (other files are ignored)
    #   ./masks = original ROI/ZIP annotation folders
    # No access.txt is used.
    nd2_files = find_nd2_files()
    roi_directory, mask_files = collect_roi_source_files()

    # Darkfield-corrected recordings, using the existing preprocessing method.
    data = load_recording_data(nd2_files)

    # Optional TIFF annotations are only used for validation, never as track IDs.
    annotation, frameindex = load_optional_tiff_annotations(data)

    print(f"Original ROI/ZIP source directory: {roi_directory}")
    print(
        "Original ROI/ZIP files were located. They are not converted during "
        "tracking and are not used as temporal track IDs."
    )
    if annotation:
        print(
            "Exported TIFF annotations were also loaded and checked against "
            "the matching ND2 recordings."
        )

    model, device = load_model()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for nd2_path in nd2_files:
        recording = nd2_path.stem
        run_recording(nd2_path, data[recording], model, device)

    print(f"\\nFinished. Results are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
