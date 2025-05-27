import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # We'll use PIL (Pillow) to load images
from scipy.signal import convolve2d

img_rgb = np.array(Image.open('ass2/ampelmaennchen.png').convert('RGB'))
img = np.array(Image.open('ass2/ampelmaennchen.png').convert('L')).astype(float) / 255.0

sigma = 0.5
t_w = 0.004
t_q = 0.5

def overlay_mask(image, binary_mask):

    binary_mask = binary_mask.astype(bool) # Ensure mask is boolean

    overlay_img = image.copy()

    if image.ndim == 2:  # Grayscale
        overlay_img[binary_mask] = 255

    elif image.ndim == 3 and image.shape[2] == 3:  # RGB
        overlay_img[binary_mask] = [255, 0, 0]

    else:
        raise ValueError("Input image must be either grayscale or RGB with 3 channels.")

    return overlay_img

# TASK A =============================================================

def gradient_of_gaussian_kernel(size=5, sigma=1.0, direction='x'):
    assert size % 2 == 1, "Size must be odd"
    assert direction in ('x', 'y'), "Direction must be 'x' or 'y'"
    
    k = size // 2
    y, x = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1), indexing='ij')
    
    # 2D Gaussian derivative (gradient of Gaussian)
    factor = -1 / (2 * np.pi * sigma**4) # separate the constant factor to speed up the calculation

    if direction == 'x':    gaussian_derivative = np.round(factor * x * np.exp(-(x**2 + y**2) / (2 * sigma**2)), 4)
    else:                   gaussian_derivative = np.round(factor * y * np.exp(-(x**2 + y**2) / (2 * sigma**2)), 4)
    
    return gaussian_derivative

kernel_size = 5

G_x = gradient_of_gaussian_kernel(kernel_size, sigma, 'x') # Kernel X
G_y = gradient_of_gaussian_kernel(kernel_size, sigma, 'y') # Kernel Y

I_x = convolve2d(img, G_x, mode='same', boundary='symm') # horizontal gradient
I_y = convolve2d(img, G_y, mode='same', boundary='symm') # vertical gradient

I_x_square = I_x ** 2
I_y_square = I_y ** 2

G = np.sqrt(I_x_square + I_y_square) # Gradient magnitude

plt.figure(); plt.imshow(G, 'gray'); plt.title("Gradient Magnitude"); plt.axis('off')

# TASK B =============================================================

# a

Ix_Iy = I_x * I_y

# Smooth the products
g = np.ones((1,5))

I_x_square = convolve2d(convolve2d(I_x_square, g, mode="same"), g.T, mode="same")
I_y_square = convolve2d(convolve2d(I_y_square, g, mode="same"), g.T, mode="same")
Ix_Iy = convolve2d(convolve2d(Ix_Iy, g, mode="same"), g.T, mode="same")

# Ix2_conv = convolve2d(I_x_square, w_n, mode='same', boundary='symm')
# Iy2_conv = convolve2d(I_y_square, w_n, mode='same', boundary='symm')
# IxIy_conv = convolve2d(Ix_Iy, w_n, mode='same', boundary='symm')


# b

# Step 1: Compute trace and determinant
trace = I_x_square + I_y_square
det = I_x_square * I_y_square - Ix_Iy**2

# Step 2: Compute eigenvalues of the structure tensor
half_trace = trace / 2
under_sqrt = np.clip(half_trace**2 - det, 0, None)  # Ensure non-negative for sqrt
lambda1 = half_trace + np.sqrt(under_sqrt)
lambda2 = half_trace - np.sqrt(under_sqrt)

W = np.minimum(lambda1, lambda2)# Shi-Tomasi cornerness measure: minimum eigenvalue

# Step 3: Compute roundness q
epsilon = 1e-12  # Avoid divide-by-zero
Q = (4 * det) / (trace**2 + epsilon)


# Clamp any tiny negatives
W = np.clip(W, 0, None)
Q = np.clip(Q, 0, 1)

plt.figure(); plt.imshow(W, 'jet'); plt.title("Cornerness"); plt.axis('off')
plt.figure(); plt.imshow(Q, 'jet'); plt.title("Roundness"); plt.axis('off')

# c

mask = (W > t_w) & (Q > t_q)

# d

final_overlay = overlay_mask(img_rgb, mask)


plt.figure(); plt.imshow(final_overlay, ); plt.title("Final Overlay"); plt.axis('off')
plt.show()