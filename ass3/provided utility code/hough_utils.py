import numpy as np


def houghpeaks(H, num_peaks, threshold=None, nhood_size=None):
    # Set default values
    if threshold is None:
        threshold = 0.5 * np.max(H)
    
    if nhood_size is None:
        nhood_size = np.array(H.shape) // 50
        nhood_size = 2 * (nhood_size // 2) + 1  # Ensure odd size
        nhood_size = np.maximum(nhood_size, 1)
    
    # Make a copy of the Hough matrix
    Hnew = H.copy()
    peaks = []
    
    # Find peaks iteratively
    for _ in range(num_peaks):
        # Check if any values are above threshold
        if np.max(Hnew) < threshold:
            break
        
        # Find location of maximum
        max_idx = np.argmax(Hnew)
        p, q = np.unravel_index(max_idx, Hnew.shape)
        
        # Add to peaks
        peaks.append([p, q])
        
        # Suppress neighborhood
        p1 = max(0, p - nhood_size[0] // 2)
        p2 = min(Hnew.shape[0], p + nhood_size[0] // 2 + 1)
        q1 = max(0, q - nhood_size[1] // 2)
        q2 = min(Hnew.shape[1], q + nhood_size[1] // 2 + 1)
        
        # Zero out the neighborhood
        Hnew[p1:p2, q1:q2] = 0
    
    # Return empty array with right shape if no peaks
    if not peaks:
        return np.zeros((0, 2), dtype=int)
    return np.array(peaks)

def houghpixels(binary_img, theta, rho, peak):
    rows, cols = binary_img.shape
    y, x = np.nonzero(binary_img)
    
    theta_rad = theta[peak[1]] * np.pi / 180
    rho_val = rho[peak[0]]
    
    # Calculate distance from origin for each pixel
    rho_pixel = x * np.cos(theta_rad) + y * np.sin(theta_rad)
    
    # Find pixels that match the peak rho value (with some tolerance)
    # Calculate bin size
    rho_bin_size = (rho[-1] - rho[0]) / (len(rho) - 1)
    # Find pixels within half bin of the peak rho
    idx = np.where(np.abs(rho_pixel - rho_val) <= 0.5 * rho_bin_size)[0]
    
    return y[idx], x[idx]

def houghlines(binary_img, theta, rho, peaks, fill_gap=20, min_length=40):
    lines = []
    fill_gap_sq = fill_gap**2
    min_length_sq = min_length**2
    
    for peak in peaks:
        # Get pixels that contributed to this peak
        r, c = houghpixels(binary_img, theta, rho, peak)
        
        if len(r) == 0:
            continue
            
        # Create coordinate pairs and sort them
        xy = np.column_stack((c, r))
        # Sort based on either x or y depending on line orientation
        r_range = np.max(r) - np.min(r)
        c_range = np.max(c) - np.min(c)
        
        if r_range > c_range:
            # Sort first by y, then by x
            idx = np.lexsort((c, r))
        else:
            # Sort first by x, then by y
            idx = np.lexsort((r, c))
            
        xy = xy[idx]
        
        # Compute squared distances between consecutive points
        diffs = np.diff(xy, axis=0)
        dists_sq = np.sum(diffs**2, axis=1)
        
        # Find gaps larger than the threshold
        gap_idx = np.where(dists_sq > fill_gap_sq)[0]
        idx_ranges = np.concatenate(([0], gap_idx + 1, [len(xy)]))
        
        # Process each segment
        for i in range(len(idx_ranges) - 1):
            start_idx = idx_ranges[i]
            end_idx = idx_ranges[i + 1] - 1
            
            if end_idx < start_idx:  # Skip if empty segment
                continue
                
            p1 = xy[start_idx]
            p2 = xy[end_idx]
            
            # Check if line segment is long enough
            line_length_sq = np.sum((p2 - p1)**2)
            
            if line_length_sq >= min_length_sq:
                lines.append({
                    'point1': p1,
                    'point2': p2,
                    'theta': theta[peak[1]],
                    'rho': rho[peak[0]]
                })
    
    return lines