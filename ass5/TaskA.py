import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import median_filter
import os

# Load image with PIL and convert to RGB
img_path = 'ass5/input_images/inputEx5_1.jpg'
img = np.array(Image.open(img_path).convert('RGB'))
h, w, _ = img.shape

# Plotting helper 
def create_figure(img, cmap, title, hasAxes):
    plt.figure(); plt.imshow(img, cmap); plt.title(title); plt.axis("on" if hasAxes else "off")

# Manual K-means function 
def kmeans(X, k, max_iters=100):
    centers = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(max_iters):
        dists = np.linalg.norm(X[:, None] - centers[None], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
            for i in range(k)
        ])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return labels, centers

# Match clusters to fixed RGB colors
def match_clusters_to_fixed_colors(centers_rgb, fixed_colors):
    label_to_color = {}
    for i, c in enumerate(centers_rgb):
        dists = np.linalg.norm(fixed_colors - c, axis=1)
        label_to_color[i] = np.argmin(dists)
    return label_to_color

# Prepare features (RGB + spatial XY) 
pixels_rgb = img.reshape(-1, 3).astype(np.float32)
xx, yy = np.meshgrid(np.arange(w), np.arange(h))
xy = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float32)

spatial_weight = 0.1
features = np.concatenate([pixels_rgb, xy * spatial_weight], axis=1)

# Run K-means 
k = 3
labels, centers = kmeans(features, k)
labels_2d = labels.reshape(h, w)

#  Smooth labels using median filter 
filtered_labels = median_filter(labels_2d, size=1)

# Define target colors 
fixed_colors = np.array([
    [144, 238, 144],  # light green
    [255, 0, 0],      # red
    [0, 0, 139]       # dark blue
], dtype=np.float32)

# Map clusters to fixed colors
label_to_color_idx = match_clusters_to_fixed_colors(centers[:, :3], fixed_colors)
segmented_img = np.zeros_like(pixels_rgb)
for i, l in enumerate(filtered_labels.ravel()):
    segmented_img[i] = fixed_colors[label_to_color_idx[l]]
segmented_img = segmented_img.reshape(h, w, 3).astype(np.uint8)

# Show original and segmented images
create_figure(img, None, "Original Image", 0)
create_figure(segmented_img, None, f"K-Means Segmented (k={k})", 0)
plt.show()

output_dir = 'ass5/output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Convert segmented_img (numpy array) to PIL Image and save
img_to_save = Image.fromarray(segmented_img)
output_path = os.path.join(output_dir, "segmented_output.jpg")
img_to_save.save(output_path)

print(f"Segmented image saved to {output_path}")
