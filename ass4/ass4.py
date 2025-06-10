import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # We'll use PIL (Pillow) to load images

img_rgb = np.array(Image.open('ass4/input_images/taskA.jpg').convert('RGB'))
img_grayscale = np.array(Image.open('ass4/input_images/taskA.jpg').convert('L')).astype(float) / 255.0 # Image as double from 0 to 1

def create_figure (img, cmap, title, hasAxes):
    plt.figure(); plt.imshow(img, cmap); plt.title(title); plt.axis("on" if hasAxes else "off")
   
def create_gaussian_kernel(size=5, sigma=1.0):
    assert size % 2 == 1, "Size must be odd"
    
    k = size // 2
    y, x = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1), indexing='ij')
    
    # 2D Gaussian derivative (gradient of Gaussian)
    factor = 1 / (2 * np.pi * sigma**2) # separate the constant factor to speed up the calculation

    gaussian_derivative = np.round(factor * np.exp(-(x**2 + y**2) / (2 * sigma**2)), 4)
    
    return gaussian_derivative

def pad_array(img, newShape):
    old_height, old_width = img.shape
    
    pad_height = newShape[0] - old_height
    pad_width = newShape[1] - old_width
    
    padded_img = np.pad(img, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
    
    return padded_img

 
# TASK A

# Gaussian noise parameters
noise_mean = 0
noise_variance = 0.01

# Add Gaussian noise to the grayscale image
noise = np.random.normal(noise_mean, np.sqrt(noise_variance), img_grayscale.shape)
img_noisy = np.clip(img_grayscale + noise, 0, 1)  # Ensure the noisy image is still in the range [0, 1]

# Kernel parameters
kernel_size = 25
gaussian_sigma = 2.5

# Create the kernel
kernel = create_gaussian_kernel(kernel_size, gaussian_sigma)

# Pad and shift the kernel for the fft
kernel_padded = pad_array(kernel, img_noisy.shape)
kernel_padded = np.roll(kernel_padded, np.round(-kernel_size/2), (0, 1)) # roll left and up by half kernel size

# FFT on both image and kernel
image_spectrum = np.fft.fft2(img_noisy)
kernel_spectrum = np.fft.fft2(kernel_padded)

# Multiply in frequency domain
filtered_image_spectrum = image_spectrum * kernel_spectrum

# Inverse FFT to get the result
filtered_image = np.fft.ifft2(filtered_image_spectrum).real


# Plot all necessary steps
create_figure(img_noisy, 'gray', "Noisy Image", 0)
create_figure(np.log(np.abs(np.fft.fftshift(image_spectrum))), None, "image spectrum (centered & scaled)", 0)
create_figure(np.log(np.abs(np.fft.fftshift(kernel_spectrum))), None, "filter spectrum (centered & scaled)", 0)
create_figure(np.log(np.abs(np.fft.fftshift(filtered_image_spectrum))), None, "filtered image spectrum (centered & scaled)", 0)
create_figure(filtered_image, "gray", "filtered image", 0)


plt.show()