import os
import sys
import argparse
import json
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

# Ensure project root is on sys.path to import utils
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import bubbledetection
from utils import answerkey


def collect_image_paths(root: str) -> List[str]:
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)


def extract_patches_for_image(img_path: str, labels_from_detection: bool = True) -> List[Tuple[np.ndarray, int]]:
    """
    Extract bubble ROIs and simple labels using threshold heuristic as proxy.
    Label: 1 = filled, 0 = empty. If labels_from_detection is False, labels are -1.
    """
    img = Image.open(img_path)
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    samples: List[Tuple[np.ndarray, int]] = []
    for subject, q, opt, roi in bubbledetection.iter_bubble_rois(gray):
        if roi.size == 0 or roi.shape[0] < 8 or roi.shape[1] < 8:
            continue
        # Resize to 28x28
        roi28 = cv2.resize(roi, (28, 28))
        label = -1
        if labels_from_detection:
            # Use same rule as fallback threshold to assign pseudo-labels
            _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            filled_ratio = np.sum(thresh == 255) / thresh.size
            label = 1 if filled_ratio > 0.15 else 0
        samples.append((roi28, label))
    return samples


def main():
    parser = argparse.ArgumentParser(description="Extract bubble patches dataset from OMR images")
    parser.add_argument('--images-root', required=True, help='Folder with OMR images (will search recursively)')
    parser.add_argument('--out-npy', default='dataset_bubbles.npy', help='Output .npy file with (X, y) arrays')
    parser.add_argument('--weak-labels', action='store_true', help='Use threshold heuristic to label patches (1=filled,0=empty)')
    parser.add_argument('--max-images', type=int, default=0, help='Optional cap on number of images to process (0 = no cap)')
    parser.add_argument('--balance', action='store_true', help='Downsample majority class to balance 0/1 labels')
    args = parser.parse_args()

    image_paths = collect_image_paths(args.images_root)
    if not image_paths:
        print(f"No images found in {args.images_root}")
        return

    X_list = []
    y_list = []
    total = 0
    processed = 0
    for p in image_paths:
        samples = extract_patches_for_image(p, labels_from_detection=args.weak_labels)
        for roi28, lab in samples:
            X_list.append(roi28)
            y_list.append(lab)
        total += len(samples)
        print(f"Processed {p} -> {len(samples)} patches")
        processed += 1
        if args.max_images and processed >= args.max_images:
            break

    X = np.stack(X_list, axis=0).astype('float32') / 255.0
    X = X.reshape((-1, 28, 28, 1))
    y = np.array(y_list, dtype=np.int64)

    # Optionally balance classes
    if args.balance:
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]
        if len(idx0) and len(idx1):
            m = min(len(idx0), len(idx1))
            sel0 = np.random.choice(idx0, m, replace=False)
            sel1 = np.random.choice(idx1, m, replace=False)
            sel = np.concatenate([sel0, sel1])
            np.random.shuffle(sel)
            X = X[sel]
            y = y[sel]
            print(f"Balanced dataset to {len(X)} samples (each class ~{m})")

    np.save(args.out_npy, {'X': X, 'y': y})
    # Report class distribution
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    print(f"Saved dataset: {args.out_npy} with {X.shape[0]} samples; class dist: {dist}")


if __name__ == '__main__':
    main()
