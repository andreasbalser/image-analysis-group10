import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from provided_utility_code.imadjust_utils import imadjust
from provided_utility_code.hough_utils import houghpeaks, houghlines  
from PIL import Image  # We'll use PIL (Pillow) to load images
from scipy.signal import convolve2d

img_rgb = np.array(Image.open('ass3/input_ex3.jpg').convert('RGB'))
img = np.array(Image.open('ass3/input_ex3.jpg').convert('L')).astype(float) / 255.0

sigma = 0.5

# TASK A

# a
plt.figure(); plt.imshow(img, 'gray'); plt.title("Grayscale Image"); plt.axis('off')

# b

def gradient(I, sigma):
    r = int(round(3 * sigma))
    i = np.arange(-r, r + 1)

    # 1D Gaussian
    g = np.exp(-i**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

    # Derivative of Gaussian
    d = -i * g / sigma**2

    # Apply separable convolution: GoG
    Ix = convolve2d(convolve2d(I, g[:, np.newaxis], mode='same', boundary="symm"),
                    d[np.newaxis, :], mode='same', boundary="symm")
    Iy = convolve2d(convolve2d(I, g[np.newaxis, :], mode='same', boundary="symm"),
                    d[:, np.newaxis], mode='same', boundary="symm")

    return Ix, Iy

Ix, Iy = gradient(img, sigma)  # Compute gradient in x and y directions
magnitude = np.sqrt(Ix**2 + Iy**2)  # Calculate gradient magnitude

# c

threshold = 0.07
edge_mask = (magnitude > threshold) # Apply threshold to the gradient magnitude

plt.figure(); plt.imshow(edge_mask, 'gray'); plt.title("Binary Edge Mask"); plt.axis('off')


# d

def hough(edge_mask, Ix, Iy):
    height, width = edge_mask.shape
    y_indices, x_indices = np.nonzero(edge_mask)  # Get coordinates of edge pixels to avoid loops over the entire image
    
    max_distance = int(np.sqrt(height**2 + width**2)) # this is 'd' from the lecture slides
    theta = np.arange(-90, 90)  # Angles from -90 to 90 degrees
    rho = np.arange(-max_distance, max_distance + 1)  # Rho values
    
    # Initialize the Hough accumulator
    accumulator = np.zeros((len(rho), len(theta)), dtype=np.int8) # This is 'H' in the lectureslides
    
    for x, y in zip(x_indices, y_indices):
        # Gradient components
        df_dx = Ix[y, x] # Gradient in x direction
        df_dy = Iy[y, x] # y direction
            
        t = np.arctan(df_dy / df_dx) # Gradient direction (angle in radians ranging from -pi/2 to pi/2)
        r = int(x * np.cos(t) + y * np.sin(t)) # Distance from origin to the line (ranging from -d to d)

        # Discretize theta and rho into indices 
        theta_idx = int(np.rad2deg(t) + 90) # Convert radians to degrees and shift to positive range
        rho_idx = int(r + max_distance) # Shift rho to positive range
        
        
        accumulator[rho_idx, theta_idx] += 1
        accumulator[rho_idx - 1 : rho_idx + 2, theta_idx - 1 : theta_idx + 2] += 1

        
    return accumulator, theta, rho

accumulator, theta, rho = hough(edge_mask, Ix, Iy)

plt.figure(); plt.imshow(imadjust(accumulator.T), 'gray'); plt.title("Voting Table")

hough_peaks = houghpeaks(accumulator, 40, .4, [12, 12])
hough_lines = houghlines(edge_mask, theta, rho, hough_peaks, fill_gap=20, min_length=50)

def plot_peaks_on_accumulator(accumulator, peaks, rect_size=[12, 12]):
    plt.figure()
    plt.imshow(imadjust(accumulator.T), cmap='gray')
    ax = plt.gca()
    for peak in peaks:
        # Note: accumulator.T is plotted, so swap x/y for rectangle
        theta_idx, rho_idx = peak[1], peak[0]
        rect = patches.Rectangle(
            (rho_idx - rect_size[0] // 2, theta_idx - rect_size[1] // 2),
            rect_size[0], rect_size[1],
            linewidth=.5, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
    plt.title("Accumulator with Peak Rectangles")
    plt.axis('on')
    
def plot_hough_lines_on_image(img_rgb, hough_lines, color=(255, 0, 0), linewidth=2):
    # i. Plot final lines on original RGB image
    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.title("Final Line Detection Visualization")

    for line in hough_lines:
        (x1, y1), (x2, y2) = line['point1'], line['point2']

        plt.plot([x1, x2], [y1, y2], color='lime', linewidth=2)
        plt.plot(x1, y1, 'ro', markersize=5)
        plt.plot(x2, y2, 'yo', markersize=5)
    plt.axis('off')

plot_peaks_on_accumulator(accumulator, hough_peaks)
plot_hough_lines_on_image(img_rgb, hough_lines)

plt.show()