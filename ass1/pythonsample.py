import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # We'll use PIL (Pillow) to load images

# Let's create a simple image (a gradient)
height, width = 100, 200
gradient = np.zeros((height, width), dtype=np.uint8)

for i in range(width):
    gradient[:, i] = 255 * i / width

plt.figure(figsize=(10, 4))
plt.imshow(gradient, cmap='gray')
plt.title('Grayscale Gradient')
plt.axis('off')
plt.show()

# This is how you would typically load an image in the assignments
# Replace 'your_image.jpg' with an actual file
# img = np.array(Image.open('input_sat_image.jpg'))

# For this tutorial, let's create a synthetic image
checkerboard = np.zeros((8, 8), dtype=np.uint8)
checkerboard[::2, ::2] = 255  # Set every other pixel to white
checkerboard[1::2, 1::2] = 255

plt.imshow(checkerboard, cmap='gray')
plt.title('Checkerboard Pattern')
plt.axis('off')
plt.show()

# Converting to RGB
checkerboard_rgb = np.zeros((8, 8, 3), dtype=np.uint8)
checkerboard_rgb[::2, ::2] = [255, 0, 0]  # Red
checkerboard_rgb[1::2, 1::2] = [0, 0, 255]  # Blue

plt.figure(figsize=(5, 5))
plt.imshow(checkerboard_rgb)
plt.title('RGB Checkerboard')
plt.axis('off')
plt.show()

# Create a simple image
img = np.linspace(0, 255, 25).reshape(5, 5).astype(np.uint8)
print("Original image (uint8):\n", img)

# Convert to float in range [0, 1]
img_float = img.astype(np.float64) / 255.0
print("\nNormalized image (float64 in [0, 1]):\n", img_float)

# This normalization is common in the assignments
# When you load an image, you'll often do:
# img = np.array(Image.open('image.jpg')).astype(np.float64) / 255.0

# Display both images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.imshow(img, cmap='gray')
ax1.set_title('uint8 [0, 255]')
ax1.axis('off')

ax2.imshow(img_float, cmap='gray')
ax2.set_title('float64 [0, 1]')
ax2.axis('off')

plt.show();
