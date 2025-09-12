import json
import numpy as np
import pycocotools.mask as mask_util
import cv2

def get_point_by_mask_region(mask, region="top"):
    """
    Get a representative point on the mask based on a region keyword.
    Args:
        mask: (H, W) binary mask
        region: str, one of ['top', 'bottom', 'left', 'right', 'center',
                             'top-left', 'top-right', 'bottom-left', 'bottom-right']
    Returns:
        (x, y) tuple of pixel coordinates
    """
    ys, xs = np.where(mask == 1)
    if len(xs) == 0 or len(ys) == 0:
        return None  # Empty mask

    # Bounding box
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)

    # Candidate set of all mask points
    points = np.stack([xs, ys], axis=1)

    if "top" in region:
        points = points[points[:, 1] == y_min]
    elif "bottom" in region:
        points = points[points[:, 1] == y_max]

    if "left" in region:
        points = points[points[:, 0] == np.min(points[:, 0])]
    elif "right" in region:
        points = points[points[:, 0] == np.max(points[:, 0])]

    if region == "center":
        x_center = int((x_min + x_max) / 2)
        y_center = int((y_min + y_max) / 2)
        return (x_center, y_center)

    if len(points) > 0:
        median_idx = len(points) // 2
        return tuple(points[median_idx])
    else:
        # fallback: return center if region not matched
        return (int((x_min + x_max) / 2), int((y_min + y_max) / 2))


# === Load the JSON output ===
with open('outputs/test_sam2.1/grounded_sam2_hf_model_demo_results.json', 'r') as f:
    data = json.load(f)

# === Load original image ===
img_path = data["image_path"]
img = cv2.imread(img_path)

# === Get mask RLE and decode ===
annotation = data["annotations"][1]
rle = annotation["segmentation"]
mask = mask_util.decode(rle)  # (H, W), dtype=uint8

# === Specify the region of interest ===
region = "center"  # can be "top", "bottom", "center", etc.

# === Get point and draw ===
point = get_point_by_mask_region(mask, region)

if point:
    annotated_img = img.copy()
    cv2.circle(annotated_img, point, radius=5, color=(0, 0, 255), thickness=-1)
    cv2.putText(annotated_img, 'affordance point', (point[0] + 5, point[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    save_path = f"outputs/test_sam2.1/image_with_feature_point.jpg"
    cv2.imwrite(save_path, annotated_img)
    print(f"{region} point drawn at: {point}")
    print(f"Saved to: {save_path}")
else:
    print("No valid point found in the mask.")