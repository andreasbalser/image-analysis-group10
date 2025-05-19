from PIL import Image
from utils import rgb2grayfloat
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# Load and prepare image
img = Image.open("ampelmaennchen.png").convert("RGB")
img_np = np.asarray(img) / 255.0
gray_img = rgb2grayfloat(img_np)

## TASK A: Compute GoG kernels, apply them, and return gradient images Ix, Iy and magnitude G.

def gog_gradient_and_magnitude(gray_img, sigma=0.5, kernel_size=5, show=True):

    # Compute GoG-filter kernels for x- and y-direction
    def gog_kernels(sigma, size):
        radius = size // 2

        #Create a meshgrid of x and y coordinates centered around 0
        x, y = np.meshgrid(
            np.linspace(-radius, radius, size),
            np.linspace(-radius, radius, size)
        )

        #Compute the 2D Gaussian kernel
        gaussian = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

        #Compute the gradient of the Gaussian in x-direction
        gx = -x * gaussian / (sigma**2)

        #compute the gradient of the Gaussian in y-direction
        gy = -y * gaussian / (sigma**2)

        return gx, gy

    #Apply the two filters Gx and Gy on the input image using convolution
    def apply_gog_filters(gray, Gx, Gy):

        # Convolve the image with the Gx kernel to get the x-gradient image
        Ix = convolve2d(gray, Gx, mode='valid')

        # Convolve the image with the Gy kernel to get the y-gradient image
        Iy = convolve2d(gray, Gy, mode='valid')

        return Ix, Iy

    # Get the GoG filter kernels for x and y
    Gx, Gy = gog_kernels(sigma, kernel_size)

    # Apply filters to compute the gradient images Ix and Iy
    Ix, Iy = apply_gog_filters(gray_img, Gx, Gy)

    # Compute and visualize the gradient magnitude
    # Compute gradient magnitude
    G = np.sqrt(Ix**2 + Iy**2)

    if show:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.title("Gradient x (Ix)")
        plt.imshow(Ix, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Gradient y (Iy)")
        plt.imshow(Iy, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Gradient Magnitude (G)")
        plt.imshow(G, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return Ix, Iy, G

## TASK b: detect Förstner interest points based on gradient images Ix and Iy.

def foerstner_interest_points(Ix, Iy, gray_img, window_size=5, tw=0.004, tq=0.5, show=True):
    # Compute gradient products
    Ix2 = Ix**2
    Iy2 = Iy**2
    Ixy = Ix * Iy

    # Autocorrelation matrix via convolution
    window = np.ones((window_size, window_size))
    S_Ix2 = convolve2d(Ix2, window, mode='valid')
    S_Iy2 = convolve2d(Iy2, window, mode='valid')
    S_Ixy = convolve2d(Ixy, window, mode='valid')

    # Compute cornerness and roundness
    det_M = S_Ix2 * S_Iy2 - S_Ixy**2
    trace_M = S_Ix2 + S_Iy2

    epsilon = 1e-12
    W = det_M
    Q = 4 * det_M / (trace_M**2 + epsilon)

    # Thresholding
    interest_mask = (W > tw) & (Q > tq)
    y_coords, x_coords = np.where(interest_mask)

    # Crop image to match W/Q size
    crop_y, crop_x = W.shape
    gray_cropped = gray_img[:crop_y, :crop_x]

    if show:
        plt.figure(figsize=(16, 4))

        plt.subplot(1, 3, 1)
        plt.title("Cornerness W")
        plt.imshow(W, cmap='jet')
        plt.colorbar()
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Roundness Q")
        plt.imshow(Q, cmap='jet')
        plt.colorbar()
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Förstner Interest Points")
        plt.imshow(gray_cropped, cmap='gray')
        plt.plot(x_coords, y_coords, 'rx', markersize=1)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return interest_mask, W, Q

# Task A
Ix, Iy, G = gog_gradient_and_magnitude(gray_img, sigma=0.5, kernel_size=5)

# Task B
interest_mask, W, Q = foerstner_interest_points(Ix, Iy, gray_img, window_size=5, tw=0.004, tq=0.5)
