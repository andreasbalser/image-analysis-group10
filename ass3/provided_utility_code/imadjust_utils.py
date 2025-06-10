import numpy as np

def stretchlim(img, tol=0.01):
    """
    Calculate the contrast limits for contrast stretching
    
    Parameters:
    -----------
    img : ndarray
        Input image
    tol : float, optional
        Tolerance value in the range [0, 1]. Default is 0.01 (1%)
        
    Returns:
    --------
    limits : ndarray
        2xN array of low and high values for each image channel
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("Image must be a numpy array")
    
    # Convert to float if not already
    if img.dtype != np.float64 and img.dtype != np.float32:
        img_f = img.astype(np.float64)
        # Normalize based on data type
        if img.dtype == np.uint8:
            img_f = img_f / 255.0
        elif img.dtype == np.uint16:
            img_f = img_f / 65535.0
        elif img.dtype == np.int16:
            img_f = (img_f - (-32768)) / (32767 - (-32768))
    else:
        img_f = img
    
    # Reshape to handle multi-dimensional images
    orig_shape = img_f.shape
    if len(orig_shape) > 2:
        # Reshape to have channels as last dimension
        n_channels = np.prod(orig_shape[2:])
        img_f = img_f.reshape(-1, n_channels)
    else:
        img_f = img_f.reshape(-1, 1)
        n_channels = 1
    
    # Calculate limits for each channel
    limits = np.zeros((2, n_channels))
    for i in range(n_channels):
        channel = img_f[:, i]
        channel = channel[~np.isnan(channel)]  # Remove NaN values
        
        if len(channel) > 0:
            # Find limits based on histogram
            sorted_vals = np.sort(channel)
            low_idx = int(max(0, np.floor(tol * len(sorted_vals))))
            high_idx = int(min(len(sorted_vals)-1, np.ceil((1-tol) * len(sorted_vals))))
            
            limits[0, i] = sorted_vals[low_idx]
            limits[1, i] = sorted_vals[high_idx]
            
            if limits[1, i] == limits[0, i]:
                # If both limits are zero, look for the smallest non-zero value
                if limits[0, i] == 0:
                    non_zeros = sorted_vals[sorted_vals > 0]
                    if len(non_zeros) > 0:
                        # Use the smallest non-zero value as high_limit
                        limits[1, i] = non_zeros[0]
                    else:
                        # If no non-zero values, add a small epsilon
                        limits[1, i] = 1e-6
                else:
                    # If non-zero but equal limits, add a small percentage
                    limits[1, i] = limits[0, i] * 1.01  # Add 1%
        else:
            # Default if channel is empty
            limits[:, i] = [0, 1]
    
    # Reshape limits if needed for multidimensional images
    if len(orig_shape) > 2:
        limits = limits.reshape(2, *orig_shape[2:])
    
    return limits

def parse_limits(limits, shape):
    """
    Parse input/output limits for imadjust
    
    Parameters:
    -----------
    limits : ndarray or None
        Input limits array or None
    shape : tuple
        Shape of the image
        
    Returns:
    --------
    parsed_limits : ndarray
        Properly formatted limits array
    """
    if limits is None or (isinstance(limits, np.ndarray) and limits.size == 0):
        # Default to [0, 1] for all channels
        if len(shape) > 2:
            n_planes = np.prod(shape[2:])
            return np.tile(np.array([[0], [1]]), (1, n_planes))
        else:
            return np.array([[0], [1]])
    
    if not isinstance(limits, np.ndarray) or limits.dtype not in [np.float32, np.float64]:
        raise ValueError("Limits must be numeric floating-point arrays")
    
    if np.min(limits) < 0 or np.max(limits) > 1:
        raise ValueError("Limits must be in the range [0, 1]")
    
    # Handle single pair of limits for all channels
    if limits.size == 2:
        if len(shape) > 2:
            n_planes = np.prod(shape[2:])
            return np.tile(limits.reshape(2, 1), (1, n_planes))
        else:
            return limits.reshape(2, 1)
    
    # Check correct format for multi-channel
    if limits.shape[0] != 2:
        raise ValueError("Limits must be a 2-row column per plane")
    
    # For multi-channel, ensure correct shape
    if len(shape) > 2:
        expected_shape = (2,) + shape[2:]
        if limits.shape != expected_shape:
            raise ValueError("Limits must be a 2-row column per plane")
    
    return limits

def imadjust_direct(img, in_limits, out_limits, gamma):
    """
    Apply direct image adjustment without input validation
    
    Parameters:
    -----------
    img : ndarray
        Input image
    in_limits : ndarray
        Input limits [low_in, high_in]
    out_limits : ndarray
        Output limits [low_out, high_out]
    gamma : ndarray
        Gamma correction value(s)
        
    Returns:
    --------
    adj : ndarray
        Adjusted image
    """
    # Check if we're doing max scale or complement
    max_scale = np.all(out_limits == np.array([[0], [1]]))
    max_scale_complement = np.all(out_limits == np.array([[1], [0]]))
    
    # Extract low and high values
    low_in = in_limits[0]
    high_in = in_limits[1]
    
    if max_scale:
        # Most common case - stretch to [0, 1]
        adj = ((img - low_in) / (high_in - low_in)) ** gamma
        adj = np.clip(adj, 0, 1)
    elif max_scale_complement:
        # Image negative/complement
        adj = ((img - low_in) / (high_in - low_in)) ** gamma
        adj = 1 - adj
        adj = np.clip(adj, 0, 1)
    else:
        # General case
        low_out = out_limits[0]
        high_out = out_limits[1]
        
        # Handle values outside input range
        adj = np.zeros_like(img, dtype=np.float64)
        
        # Values below low_in
        below_mask = img < low_in
        adj[below_mask] = low_out[below_mask] if isinstance(low_out, np.ndarray) else low_out
        
        # Values above high_in
        above_mask = img >= high_in
        adj[above_mask] = high_out[above_mask] if isinstance(high_out, np.ndarray) else high_out
        
        # Values in range
        in_range_mask = (img >= low_in) & (img < high_in)
        if np.any(in_range_mask):
            normalized = ((img[in_range_mask] - low_in[in_range_mask]) / 
                         (high_in[in_range_mask] - low_in[in_range_mask])) ** gamma[in_range_mask]
            adj[in_range_mask] = low_out[in_range_mask] + (high_out[in_range_mask] - low_out[in_range_mask]) * normalized
    
    return adj

def imadjust(img, in_limits=None, out_limits=None, gamma=1):
    """
    Adjust image intensity values.
    
    Parameters:
    -----------
    img : ndarray
        Input image or colormap.
    in_limits : ndarray, optional
        2-row array specifying the lower and upper input limits.
        Default is stretchlim(img, 0.01).
    out_limits : ndarray, optional
        2-row array specifying the lower and upper output limits.
        Default is [0, 1].
    gamma : float or ndarray, optional
        Non-negative scalar specifying the shape of the mapping curve.
        Default is 1 (linear mapping).
        
    Returns:
    --------
    adj : ndarray
        Adjusted image.
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("Image must be a numpy array")
    
    # Handle colormap case
    is_colormap = len(img.shape) == 2 and img.shape[1] >= 3 and img.shape[1] <= 4
    if is_colormap:
        original_shape = img.shape
        img = img.reshape(img.shape[0], 1, img.shape[1])
    
    # Convert to float for processing
    img_type = img.dtype
    if img.dtype != np.float64 and img.dtype != np.float32:
        img_f = img.astype(np.float64)
        # Normalize based on data type
        if img.dtype == np.uint8:
            img_f = img_f / 255.0
        elif img.dtype == np.uint16:
            img_f = img_f / 65535.0
        elif img.dtype == np.int16:
            img_f = (img_f - (-32768)) / (32767 - (-32768))
    else:
        img_f = img.copy()
    
    shape = img_f.shape
    n_planes = np.prod(shape[2:]) if len(shape) > 2 else 1
    
    # Default in_limits is stretchlim with 1% saturation
    if in_limits is None:
        in_limits = stretchlim(img_f, 0.01)
    else:
        in_limits = parse_limits(in_limits, shape)
    
    # Default out_limits is [0, 1]
    if out_limits is None:
        out_limits = np.array([[0], [1]])
    else:
        out_limits = parse_limits(out_limits, shape)
    
    # Handle gamma
    if not isinstance(gamma, (int, float, np.ndarray)) or np.any(gamma < 0):
        raise ValueError("Gamma must be non-negative")
    
    if isinstance(gamma, (int, float)):
        gamma = np.full((1, n_planes), gamma)
    elif isinstance(gamma, np.ndarray):
        if gamma.size == 1:
            gamma = np.full((1, n_planes), gamma.item())
        elif gamma.shape != (1,) + shape[2:]:
            raise ValueError("Gamma must be a scalar or 1 row per plane")
    
    # Reshape for processing
    if len(shape) > 2:
        in_limits = in_limits.reshape(2, 1, -1)
        out_limits = out_limits.reshape(2, 1, -1)
        gamma = gamma.reshape(1, 1, -1)
        img_flat = img_f.reshape(shape[0], shape[1], -1)
        
        # Process each plane
        adj_flat = np.zeros_like(img_flat)
        for i in range(n_planes):
            adj_flat[:, :, i] = imadjust_direct(
                img_flat[:, :, i], 
                in_limits[:, :, i],
                out_limits[:, :, i],
                gamma[:, :, i]
            )
        
        # Reshape back
        adj = adj_flat.reshape(shape)
    else:
        # Single plane
        adj = imadjust_direct(img_f, in_limits, out_limits, gamma)
    
    # Convert back to original type
    if img_type != np.float64 and img_type != np.float32:
        if img_type == np.uint8:
            adj = np.clip(adj * 255.0, 0, 255).astype(np.uint8)
        elif img_type == np.uint16:
            adj = np.clip(adj * 65535.0, 0, 65535).astype(np.uint16)
        elif img_type == np.int16:
            adj = np.clip(adj * (32767 - (-32768)) + (-32768), -32768, 32767).astype(np.int16)
    
    # Restore colormap shape if needed
    if is_colormap:
        adj = adj.reshape(original_shape)
    
    return adj