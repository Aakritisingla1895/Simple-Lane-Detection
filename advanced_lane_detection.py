from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Normalized points: top-left, top-right, bottom-right, bottom-left.
# Adjust these points for your camera angle.
ROAD_ROI = np.float32([
    [0.43, 0.62], [0.57, 0.62],
    [0.93, 0.96], [0.07, 0.96],
])

BIRD_ROI = np.float32([
    [0.23, 0.00], [0.77, 0.00],
    [0.77, 1.00], [0.23, 1.00],
])


def lane_pixel_mask(rgb):
    """Combine lane-marking color and locally bright vertical edges."""
    hue, light, saturation = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS))

    white = (light >= 195) & (saturation <= 105)
    yellow = (
        (hue >= 12) & (hue <= 40)
        & (saturation >= 90) & (light >= 75)
    )

    smooth = cv2.GaussianBlur(light, (5, 5), 0)
    local_mean = cv2.GaussianBlur(light, (0, 0), sigmaX=7)
    contrast = light.astype(np.int16) - local_mean.astype(np.int16)

    dx = np.abs(cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3))
    dy = np.abs(cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3))
    bright_edge = (dx >= 35) & (dx >= 0.8 * dy) & (contrast >= 15)

    return (white | yellow | bright_edge).astype(np.uint8) * 255


def sliding_window_pixels(x, y, start_x, height, width):
    """Search from the bottom of the image and recenter after each window."""
    center = start_x
    margin = max(8, int(0.08 * width))
    selected = np.zeros(x.size, dtype=bool)

    for rows in reversed(np.array_split(np.arange(height), 9)):
        inside = (
            (y >= rows[0]) & (y <= rows[-1])
            & (np.abs(x - center) <= margin)
        )
        selected |= inside

        if inside.sum() >= 15:
            center = np.median(x[inside])

    return selected


def fit_curve(x, y, height, width):
    """Fit x = ay² + by + c and iteratively remove outliers."""
    if x.size < 30 or np.ptp(y) < 0.30 * height:
        return None

    x_norm = x / (width - 1)
    y_norm = y / (height - 1)
    curve = np.polyfit(y_norm, x_norm, 2)

    for _ in range(3):
        residual = np.abs(x_norm - np.polyval(curve, y_norm))
        tolerance = np.clip(2.8 * np.median(residual), 0.012, 0.045)
        keep = residual <= tolerance

        if keep.sum() < 30 or np.ptp(y_norm[keep]) < 0.30:
            return None

        curve = np.polyfit(y_norm[keep], x_norm[keep], 2)

    return curve


def detect_lane(rgb):
    height, width = rgb.shape[:2]
    scale = np.float32([width - 1, height - 1])
    source = ROAD_ROI * scale
    destination = BIRD_ROI * scale

    transform = cv2.getPerspectiveTransform(source, destination)
    inverse = cv2.getPerspectiveTransform(destination, source)

    mask = lane_pixel_mask(rgb)
    roi_mask = np.zeros_like(mask)
    cv2.fillConvexPoly(roi_mask, np.rint(source).astype(np.int32), 255)
    mask = cv2.bitwise_and(mask, roi_mask)

    bird = cv2.warpPerspective(
        mask, transform, (width, height), flags=cv2.INTER_NEAREST
    )

    # Seed each search where the bottom half has the most lane pixels.
    histogram = np.count_nonzero(bird[height // 2:], axis=0).astype(np.float32)
    histogram = cv2.blur(histogram[None, :], (max(3, width // 50), 1))[0]
    middle = width // 2
    y, x = np.nonzero(bird)

    if histogram[:middle].max() == 0 or histogram[middle:].max() == 0:
        return mask, bird, rgb.copy(), "Both lane bases required"

    starts = [np.argmax(histogram[:middle]), middle + np.argmax(histogram[middle:])]
    curves = []

    for start in starts:
        keep = sliding_window_pixels(x, y, start, height, width)
        curves.append(fit_curve(x[keep], y[keep], height, width))

    if any(curve is None for curve in curves):
        return mask, bird, rgb.copy(), "Insufficient lane pixels"

    # Width is a fraction of the warped image, measured at 80 rows.
    rows = np.linspace(0.10, 1.0, 80)
    left_x, right_x = [np.polyval(curve, rows) for curve in curves]
    lane_width = right_x - left_x

    if lane_width.min() < 0.23 or lane_width.max() > 0.84 or np.ptp(lane_width) > 0.23:
        return mask, bird, rgb.copy(), "Lane geometry rejected"

    left = np.rint(np.column_stack((left_x, rows)) * scale).astype(np.int32)
    right = np.rint(np.column_stack((right_x, rows)) * scale).astype(np.int32)

    overlay = np.zeros_like(rgb)
    cv2.fillPoly(overlay, [np.vstack((left, right[::-1]))], (25, 210, 75))
    cv2.polylines(overlay, [left, right], False, (255, 215, 0), 4)

    overlay = cv2.warpPerspective(overlay, inverse, (width, height))
    result = cv2.addWeighted(rgb, 1.0, overlay, 0.45, 0)

    return mask, bird, result, "Curved lane detected"


paths = sorted(
    path for path in Path("test_images").glob("*")
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not paths:
    raise FileNotFoundError("Add JPG or PNG road images to test_images/")

for path in paths:
    bgr = cv2.imread(str(path))
    if bgr is None:
        print(f"Could not read {path.name}")
        continue

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask, bird, result, status = detect_lane(rgb)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(f"{path.name} — {status}")

    for ax, title, image in zip(
        axes,
        ["Original", "Color + gradient mask", "Bird’s-eye view", "Lane overlay"],
        [rgb, mask, bird, result],
    ):
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()