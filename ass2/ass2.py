import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # We'll use PIL (Pillow) to load images
from scipy.signal import convolve2d

img = np.array(Image.open('ass2/ampelmaennchen.png').convert('L')).astype(np.float32) / 255.0

def gradient_of_gaussian_kernel(size=5, sigma=1.0, direction='x'):
    assert size % 2 == 1, "Size must be odd"
    assert direction in ('x', 'y'), "Direction must be 'x' or 'y'"
    
    k = size // 2
    y, x = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1), indexing='ij')
    
    # 2D Gaussian derivative (gradient of Gaussian)
    factor = -1 / (2 * np.pi * sigma**4) # separate the constant factor to speed up the calculation

    if direction == 'x': gaussian_derivative = np.round(factor * x * np.exp(-(x**2 + y**2) / (2 * sigma**2)), 4)
    else: gaussian_derivative = np.round(factor * y * np.exp(-(x**2 + y**2) / (2 * sigma**2)), 4)
    
    return gaussian_derivative

sigma = .5
kernel_size = 5
g_x = gradient_of_gaussian_kernel(kernel_size, sigma, 'x')
g_y = gradient_of_gaussian_kernel(kernel_size, sigma, 'y')

convolve_x = convolve2d(img, g_x)
convolve_y = convolve2d(img, g_y)

g = np.sqrt(convolve_x*convolve_x + convolve_y*convolve_y)

# plt.figure(); plt.imshow(convolve_x, 'gray'); plt.title("Convolve X"); plt.axis('off')
# plt.figure(); plt.imshow(convolve_y, 'gray'); plt.title("Convolve Y"); plt.axis('off')
plt.figure(); plt.imshow(g, 'gray'); plt.title("Gradient Magnitude"); plt.axis('off')
plt.show()