import numpy as np
import matplotlib.pyplot as plt
from utils.ass_002 import gog_gradient_and_magnitude
from PIL import Image
from utils.imadjust_utils import imadjust
from utils.hough_utils import houghpeaks, houghlines  
import os

output_dir = "output_images"

# a. Read input image in RGB and grayscale
img_rgb = np.array(Image.open('input_ex3.jpg').convert('RGB'))
img = np.array(Image.open('input_ex3.jpg').convert('L')).astype(float) / 255.0  # normalize to [0, 1]

# b. Apply GoG filter
Ix, Iy, G = gog_gradient_and_magnitude(img, sigma=1.0, kernel_size=5, show=True)

# c. Threshold gradient magnitude to get edge mask
threshold_ratio = 0.23
threshold_value = threshold_ratio * np.max(G)
edge_mask = G > threshold_value

# Show binary edge mask
plt.figure(figsize=(6, 6))
plt.imshow(edge_mask, cmap='gray')
plt.title(f"Binary Edge Mask")
plt.axis('off')
plt.savefig(os.path.join(output_dir, "edge_mask.png"))
plt.close()

# d. Hough Transform with gradient direction
def hough_transform(edge_mask, grad_x, grad_y):
    # Get image dimensions
    height, width = edge_mask.shape

    # Create theta vector
    thetas = np.deg2rad(np.arange(-90, 90))
    num_thetas = len(thetas)

    # Compute max rho
    diag_len = int(np.ceil(np.sqrt(height**2 + width**2)))
    rhos = np.linspace(-diag_len, diag_len, 2 * diag_len + 1)
    num_rhos = len(rhos)

    # Initialize Hough 
    H = np.zeros((num_rhos, num_thetas), dtype=np.uint64)

    # Get coordinates of edge points
    y_idxs, x_idxs = np.nonzero(edge_mask)

    # Vote based on gradient direction
    for x, y in zip(x_idxs, y_idxs):
        gx = grad_x[y, x]
        gy = grad_y[y, x]

        theta = np.arctan2(gy, gx)
        theta_deg = np.rad2deg(theta)

        # Convert to index in theta array
        theta_idx = int(np.round(theta_deg)) + 90
        theta_idx = np.clip(theta_idx, 0, num_thetas - 1)

        # Compute rho
        rho = x * np.cos(thetas[theta_idx]) + y * np.sin(thetas[theta_idx])
        rho_idx = int(np.round(rho + diag_len))

        # Increment vote
        H[rho_idx, theta_idx] += 1

    return H, np.rad2deg(thetas), rhos

# Run Hough transform
H, theta_vals, rho_vals = hough_transform(edge_mask, Ix, Iy)

# e. Enhance Hough space visibility using imadjust
H_adjusted = imadjust(H.astype(np.float32))
plt.figure(figsize=(10, 6))
plt.imshow(H_adjusted, cmap='gray')
plt.title("Hough Voting")
plt.xlabel("Theta")
plt.ylabel("Rho")
plt.savefig(os.path.join(output_dir, "hough_adjusted.png"))
plt.close()

# f. Find peaks in Hough space
num_peaks = 80
threshold = 0.5 * np.max(H)
peaks = houghpeaks(H, num_peaks=num_peaks, threshold=threshold)

# g. Plot peaks on top 
plt.figure(figsize=(10, 6))
plt.imshow(H_adjusted, cmap='gray', extent=[theta_vals[0], theta_vals[-1], rho_vals[-1], rho_vals[0]])
plt.title("g. Hough Peaks")
plt.xlabel("Theta")
plt.ylabel("Rho")
# Overlay red circles
for rho_idx, theta_idx in peaks:
    plt.plot(theta_vals[theta_idx], rho_vals[rho_idx], 'ro', markersize=4)
plt.savefig(os.path.join(output_dir, "hough_peaks.png"))
plt.close()

# h. Extract line segments from peaks using houghlines()
min_length = 10  # accept shorter line segments
fill_gap = 5     # allow smaller gaps between points

# Derive line segments
lines = houghlines(edge_mask, theta_vals, rho_vals, peaks, fill_gap, min_length)

# i. Plot final lines on original RGB image
plt.figure(figsize=(8, 8))
plt.imshow(img_rgb)
plt.title("i. Final Line Detection Visualization")

for line in lines:
    (x1, y1), (x2, y2) = line['point1'], line['point2']

    plt.plot([x1, x2], [y1, y2], color='lime', linewidth=2)
    plt.plot(x1, y1, 'ro', markersize=5)
    plt.plot(x2, y2, 'yo', markersize=5)
plt.axis('off')
plt.savefig(os.path.join(output_dir, "final_result.png"))
plt.close()