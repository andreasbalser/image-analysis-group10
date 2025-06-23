import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # We'll use PIL (Pillow) to load images

img_rgb = np.array(Image.open('ass5/input_images/inputEx5_1.jpg').convert('RGB'))

def create_figure (img, cmap, title, hasAxes):
    plt.figure(); plt.imshow(img, cmap); plt.title(title); plt.axis("on" if hasAxes else "off")

number_clusters = 8

def k_means(img_rgb, k):
    H, W, _ = img_rgb.shape

    # Flatten RGB values: shape (H*W, 3)
    rgb = img_rgb.reshape(-1, 3)

    # Create X and Y coordinate grids
    x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))

    # Flatten: shape (H*W,)
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()

    # Stack all features: shape (H*W, 5)
    feature_space = np.column_stack((rgb, x_flat, y_flat))
    
    random_colors = np.random.randint(0, 256, (k, 3)).astype(np.float64) # Start off with k random cluster centers
    random_x = np.random.randint(0, W, (k, 1)).astype(np.float64)
    random_y = np.random.randint(0, H, (k, 1)).astype(np.float64)
    cluster_centers = np.column_stack((random_colors, random_x, random_y))
    
    cluster_attribution = np.full(len(feature_space), -1) # Stores, which pixel belongs to which cluster
    
    attrib_wasChanged = 1
    loop_count = 0
    
    
    while(attrib_wasChanged == 1): # Loop stops when cluster centers don't change anymore
        attrib_wasChanged = 0
        loop_count += 1
        
        # Attribute every pixel to the nearest cluster center
        # for pixel_idx, pixel in enumerate(feature_space):
        #     min_distance = None
        #     closest_cluster_idx = -1
        #     
        #     # Check distance to every cluster center
        #     for cluster_idx, cluster_center in enumerate(cluster_centers):
        #         distance = np.linalg.norm(pixel - cluster_center)
        #         if(min_distance is None or distance < min_distance):
        #             min_distance = distance
        #             closest_cluster_idx = cluster_idx
        #         
        #     
        #     cluster_attribution[pixel_idx] = closest_cluster_idx
        
        # Compute all distances in one go
        # Resulting shape: (num_pixels, k)
        distances = np.linalg.norm(feature_space[:, None, :] - cluster_centers[None, :, :], axis=2)

        # Get the index (cluster) with the minimum distance for each pixel
        cluster_attribution = np.argmin(distances, axis=1)
        
        # Calculate the new cluster centers for this attribution
        for cluster_idx, cluster_center in enumerate(cluster_centers):
            cluster_array = feature_space[cluster_attribution == cluster_idx, :]
            new_cluster_center = cluster_array.mean(axis=0)
            
            if(np.linalg.norm(cluster_center - new_cluster_center) > 0.0):
                #print("Iteration " + str(loop_count) + ": " + str(np.linalg.norm(cluster_center - new_cluster_center)))
                cluster_centers[cluster_idx] = new_cluster_center
                attrib_wasChanged = 1
                
        print("Clustering loop ", loop_count)
                
    # Get only the RGB part of the cluster centers
    cluster_centers_rgb = cluster_centers[:, :3]  # shape (k, 3)

    # Map each pixel to the RGB of its assigned cluster
    result_pixels = cluster_centers_rgb[cluster_attribution]  # shape (H*W, 3)

    # Reshape to original image shape
    result_img = result_pixels.reshape(H, W, 3).astype(np.uint8)
        
    return result_img

clustered_img = k_means(img_rgb, number_clusters)

create_figure(clustered_img, None, "Segmented with k = 8", 0)
plt.show()