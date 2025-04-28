import utils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # We'll use PIL (Pillow) to load images

img = np.array(Image.open('ass1\different_image.jpg'))

# FUNCTIONS =======================================

def stretch_contrast(image):
    # Find min and max pixel values
    min_val = np.min(image)
    max_val = np.max(image)

    # Avoid division by zero
    if max_val - min_val == 0:
        return np.zeros_like(image, dtype=np.uint8)

    # Stretch the contrast to [0, 255]
    stretched = (image - min_val) * 255.0 / (max_val - min_val)
    return stretched.astype(np.uint8)

def threshold_image(image, threshold=int(255/2)):
    # Ensure the image is 2D
    if image.ndim != 2:
        raise ValueError("Input image must be a grayscale image (2D array).")

    # Apply thresholding
    binary_mask = 1.0 * (image > threshold)
    binary_mask = np.array(binary_mask, dtype=bool)
    return binary_mask

def morphological_filtering(binary_image, structuring_element_open, structuring_element_close):
    opened_image = utils.imopen(binary_image, structuring_element_open)
    closed_image = utils.imclose(opened_image, structuring_element_close)
    return closed_image

def overlay_mask(image, binary_mask):

    binary_mask = binary_mask.astype(bool) # Ensure mask is boolean

    overlay_img = image.copy()

    if image.ndim == 2:  # Grayscale
        overlay_img[binary_mask] = 255

    elif image.ndim == 3 and image.shape[2] == 3:  # RGB
        overlay_img[binary_mask] = [255, 255, 255]

    else:
        raise ValueError("Input image must be either grayscale or RGB with 3 channels.")

    return overlay_img

def show_image(image, title="", showHistogram=0):
    if showHistogram:
        plt.subplot(1,2,1)

    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')

    if showHistogram:
        plt.subplot(1,2,2)
        plt.hist(img.ravel(), bins=256, range=(0, 256), color='gray')
        plt.title('Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

    plt.show()

def combine_all(image, threshold, structuring_element):
    img_gray = img.mean(axis=2).astype(np.uint8) # Compute the mean across the RGB channels

    img_enhanced = stretch_contrast(img_gray) # Enhance contrast

    binary_mask = np.logical_not(threshold_image(img_enhanced, threshold))

    filtered_mask = morphological_filtering(binary_mask, structuring_element, structuring_element)
    
    img_overlay = overlay_mask(img_enhanced, filtered_mask)

    return img_overlay


# =================================================

'''
img_gray = img.mean(axis=2).astype(np.uint8) # Compute the mean across the RGB channels
show_image(img_gray, "Initial grayscale image", 1)

img_stretched = stretch_contrast(img_gray)
show_image(img_stretched, "Contrast enhanced image", 1)

binary_mask = np.logical_not(threshold_image(img_stretched, 100))
show_image(binary_mask)

structuring_element = np.ones((9, 9), bool)
filtered_mask = morphological_filtering(binary_mask, structuring_element, structuring_element)
img_overlay = overlay_mask(img_stretched, filtered_mask)
show_image(img_overlay)
'''

structuring_element = np.ones((7, 7), bool)

final_overlay = combine_all(img, 100, structuring_element)

# show_image(final_overlay, "Final Overlay")

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Different Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(final_overlay, cmap='gray')
plt.title('Output')
plt.axis('off')

plt.show()