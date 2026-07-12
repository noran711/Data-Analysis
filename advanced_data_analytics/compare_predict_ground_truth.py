import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import cv2

img_paths = [
    "datasets/PenRaw_test/images/img_009_0000.png",
    "datasets/PenRaw_test/masks/img_009_mask.png",
    "predictions/PenSeg_168images_100epochs_0fold/img_009.png"
]

output_path = "predictions/PenSeg_168images_100epochs_0fold/compared_result_ground_truth/result_img_009.png"

images = [mpimg.imread(p) for p in img_paths]

img = images[0]
gt = images[1]
pred = images[2]

# --- grayscale fallback ---
if gt.ndim == 3:
    gt = gt[..., 0]
if pred.ndim == 3:
    pred = pred[..., 0]

# --- target shape = prediction (nnU-Net output space) ---
target_shape = pred.shape[:2]

# --- resize GT to prediction space ---
gt_resized = cv2.resize(
    gt,
    (target_shape[1], target_shape[0]),
    interpolation=cv2.INTER_NEAREST
)

# --- binarize ---
gt_bin = (gt_resized > 0).astype(np.uint8)
pred_bin = (pred > 0).astype(np.uint8)

# --- metrics ---
intersection = np.logical_and(gt_bin, pred_bin).sum()
union = np.logical_or(gt_bin, pred_bin).sum()

dice = (2 * intersection) / (gt_bin.sum() + pred_bin.sum() + 1e-8)
iou = intersection / (union + 1e-8)

# --- plot ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

titles = [
    os.path.basename(img_paths[0]),
    "Ground Truth Mask",
    "Prediction Mask"
]

axes[0].imshow(img, cmap="gray")
axes[1].imshow(gt_resized, cmap="gray")
axes[2].imshow(pred, cmap="gray")

for ax, title in zip(axes, titles):
    ax.axis("off")
    ax.set_title(title)

# --- centered metrics below all images ---
fig.text(
    0.5, -0.05,
    f"Dice Score: {dice:.4f}    |    IoU: {iou:.4f}",
    ha="center",
    fontsize=12
)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Saved: {output_path}")
print(f"Dice: {dice:.4f}")
print(f"IoU: {iou:.4f}")

plt.show()