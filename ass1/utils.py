import numpy as np

def rgb2gray(rgb_img):
    if rgb_img.ndim != 3 or rgb_img.shape[2] != 3:
        raise ValueError("Input image must be an RGB image with 3 channels")
    # Luminance weights for R, G, B channels
    # weights = np.array([0.2989, 0.5870, 0.1140])
    weights = np.array([0.5, 0.5, 0.5])
    gray_img = np.dot(rgb_img[...,:3], weights)
    return gray_img.astype(np.uint8)

def imerosion(binary_img, structuring_element):
    m, n = binary_img.shape
    k_m, k_n = structuring_element.shape
    
    pad_m = k_m // 2
    pad_n = k_n // 2
    padded_img = np.pad(binary_img, ((pad_m, pad_m), (pad_n, pad_n)), mode='constant', constant_values=0)
    
    output = np.zeros_like(binary_img)
    
    for i in range(m):
        for j in range(n):
            region = padded_img[i:i+k_m, j:j+k_n]
            if np.all(np.logical_or(region == structuring_element, structuring_element == 0)):
                output[i, j] = 1
    
    return output

def imdilation(binary_img, structuring_element):
    m, n = binary_img.shape
    k_m, k_n = structuring_element.shape
    
    pad_m = k_m // 2
    pad_n = k_n // 2
    padded_img = np.pad(binary_img, ((pad_m, pad_m), (pad_n, pad_n)), mode='constant', constant_values=0)
    
    output = np.zeros_like(binary_img)
    
    for i in range(m):
        for j in range(n):
            region = padded_img[i:i+k_m, j:j+k_n]
            if np.any(np.logical_and(region == 1, structuring_element == 1)):
                output[i, j] = 1
    
    return output

def imopen(binary_img, structuring_element):
    eroded = imerosion(binary_img, structuring_element)
    opened = imdilation(eroded, structuring_element)
    return opened

def imclose(binary_img, structuring_element):
    dilated = imdilation(binary_img, structuring_element)
    closed = imerosion(dilated, structuring_element)
    return closed