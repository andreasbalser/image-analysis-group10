import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from PIL import Image

def create_figure (img, cmap, title, hasAxes):
    plt.figure(); plt.imshow(img, cmap); plt.title(title); plt.axis("on" if hasAxes else "off")

def find_boundaries(img, threshold, plotMask):
    
    # Convert to grayscale
    img_grayscale = np.array(img.convert('L')).astype(float) / 255.0 # Image as double from 0 to 1

    # Create binary mask
    binary_mask = img_grayscale > threshold
    
    if plotMask: create_figure(binary_mask, "gray", "binary mask", 0)
    
    return measure.find_contours(binary_mask, .5)
    

def build_fourier_descriptors_from_boundaries(boundaries, n):
    
    fourier_descriptors = []
    
    # For every found contour, add fourier descriptor to the list
    for boundary in boundaries:
        fd = boundary[:, 1] + 1j * boundary[:, 0]
        fd = np.fft.fft(fd)
        
        if len(fd) >= n:
            fd = fd[:n] # take only the first n elements or...
        else:
            fd = np.pad(fd, (0, n - len(fd)), mode='constant') # ...fill up with 0 until n elements
            
        # Translation invariance
        fd[0] = 0
        
        # Scale invariance
        fd = fd / np.abs(fd[1]) # Normalization
        
        # Rotation invariance
        fd = np.abs(fd)
        
        fourier_descriptors.append(fd)
    
    return fourier_descriptors

def test_descriptor(fd_test, fd_train, test_strength):    
    distance = np.linalg.norm(fd_test - fd_train)
    
    return distance < test_strength
 
# Parameters for descriptor creation  
threshold = .15
n = 28 # How strict the descriptor matches the shape
test_strength = .08

# Create the training descriptor
train_image = Image.open('ass4/input_images/trainB.png')
train_image_boundaries = find_boundaries(train_image, threshold, 1)
FD_trained = build_fourier_descriptors_from_boundaries(train_image_boundaries, n)[0] # In this case we only expect one descriptor

# Load all tets images
test_images = []
test_images.append(Image.open('ass4/input_images/test1B.jpg'))
test_images.append(Image.open('ass4/input_images/test2B.jpg'))
test_images.append(Image.open('ass4/input_images/test3B.jpg'))

# Make detection on all images
for image_idx, image in enumerate(test_images):
    print("processing image " + str(image_idx+1))
    
    # Build descriptors on test image
    image_boundaries = find_boundaries(image, threshold, 0)
    FDs_identified = build_fourier_descriptors_from_boundaries(image_boundaries, n)

    # Prepare plot
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
            
    for fd_idx, (fd, boundary) in enumerate(zip(FDs_identified, image_boundaries)):
        if test_descriptor(fd, FD_trained, test_strength):
            distance = np.linalg.norm(fd - FD_trained)
            print(f"Image {image_idx + 1}: boundary {fd_idx} matched (distance = {distance:.4f})")

            # Plot matching boundary in blue
            ax.plot(boundary[:, 1], boundary[:, 0], linewidth=2, color='blue')
            
        else:
            # Plot non-matching boundary in red
            ax.plot(boundary[:, 1], boundary[:, 0], linewidth=1, color='red')

    ax.set_title(f"Image {image_idx + 1} - Matching Boundaries in blue")
    plt.axis('off')

plt.show()