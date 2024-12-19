import cv2
import numpy as np
from itertools import permutations
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import os
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

def preprocess_image(img, max_dim=500):
    scale = max_dim / max(img.shape[:2])
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img

def compute_similarity(img1, img2):
    # Create ORB detector with more features
    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
    
    # Preprocess images
    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)
    
    # Compute ORB features
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return np.inf
        
    # Use FLANN matcher for faster matching
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                       table_number=6,
                       key_size=12,
                       multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Get top 2 matches for ratio test
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = []
    
    # Lowe's ratio test to filter good matches
    for match_group in matches:
        if len(match_group) >= 2:
            m, n = match_group
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    if not good_matches:
        return np.inf
        
    # Compute normalized similarity score
    min_matches = 10
    max_matches = 100
    num_matches = len(good_matches)
    
    # Average distance of good matches
    avg_distance = np.mean([m.distance for m in good_matches])
    
    # Normalize number of matches
    match_score = np.clip((num_matches - min_matches) / (max_matches - min_matches), 0, 1)
    
    # Normalize distance (lower is better)
    distance_score = 1 - np.clip(avg_distance / 100, 0, 1)
    
    # Combine scores (weight can be adjusted)
    similarity = 0.7 * match_score + 0.3 * distance_score
    
    return 1 - similarity  # Convert to distance metric where lower is more similar

def build_similarity_graph(images):
    n = len(images)
    similarity_matrix = np.zeros((n, n))
    
    # Compute pairwise similarities
    for i in range(n):
        for j in range(i + 1, n):
            similarity = compute_similarity(images[i], images[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
            
    # Ensure matrix is symmetric and has zeros on diagonal
    np.fill_diagonal(similarity_matrix, 0)
    
    return similarity_matrix

def solve_tsp_MST(similarity_matrix):
    n = len(similarity_matrix)
    
    # Create MST to get initial structure
    mst = minimum_spanning_tree(similarity_matrix).toarray()
    
    # Initialize path with degree-1 vertex (leaf node) from MST
    degrees = np.sum(mst > 0, axis=0)
    start = np.where(degrees == 1)[0][0]
    
    path = [start]
    visited = {start}
    
    while len(path) < n:
        current = path[-1]
        min_cost = float('inf')
        next_node = None
        
        # Look at both MST edges and direct connections
        for neighbor in range(n):
            if neighbor not in visited:
                # Combine MST information with direct similarity
                mst_connection = mst[current, neighbor] > 0
                direct_cost = similarity_matrix[current, neighbor]
                
                # Prefer MST edges but also consider direct connections
                adjusted_cost = direct_cost * (0.8 if mst_connection else 1.2)
                
                if adjusted_cost < min_cost:
                    min_cost = adjusted_cost
                    next_node = neighbor
                    
        if next_node is None:
            break
            
        path.append(next_node)
        visited.add(next_node)
    
    return path

def order_images(images):
    similarity_matrix = build_similarity_graph(images)
    order = solve_tsp_MST(similarity_matrix)
    return [images[i] for i in order]

def load_images_from_directory(directory):
    images = []
    filenames = sorted(os.listdir(directory))
    for filename in filenames:
        img_path = os.path.join(directory, filename)
        if os.path.isfile(img_path):
            # Load the image in BGR format
            img = cv2.imread(img_path)
            if img is not None:
                # Normalize if needed
                images.append((img, filename))
            else:
                print(f"Warning: Failed to load {img_path}")
    print("Loaded Images")
    return images


def create_video_from_images(images, output_filename, framerate):
    if not images:
        raise ValueError("Image list is empty.")

    # Get dimensions from the first image
    first_img = images[0]  # Access the image part of the tuple
    height, width = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, framerate, (width, height))

    for img in images:  # Iterate over image-filename pairs
       # Ensure the image is in uint8 format
        if img.dtype not in [np.uint8, np.uint16]:
            img = (img * 255).astype('uint8') if img.dtype in [np.float32, np.float64] else img

        # Convert grayscale to BGR if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Resize to match video size
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))  # Resize to match width and height of first image

        out.write(img)

    out.release()
    print(f"Created output video: {output_filename}")

def run(input_directory, output_filename, framerate):
    images = load_images_from_directory(input_directory)
    image_data = [img[0] for img in images]
    ordered_images = order_images(image_data)
    create_video_from_images(ordered_images, output_filename, framerate)