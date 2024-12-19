import cv2
import numpy as np
import os
from tqdm import tqdm
import time

def load_images_from_directory(directory):
    images = []
    filenames = sorted(os.listdir(directory))
    for filename in filenames:
        img_path = os.path.join(directory, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append((img, filename))

    print("Loaded Images")
    return images

def detect_and_compute_sift(image):
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_keypoints(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # Lowe's ratio test
            good_matches.append(m)
    return good_matches

def get_homography_matrix(keypoints1, keypoints2, good_matches):
    if len(good_matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
    return None

def precompute_keypoints_and_descriptors(images):
    sift = cv2.SIFT_create()
    data = []
    for img, filename in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        data.append((keypoints, descriptors, img, filename))
    return data

def calculate_image_order(images):
    num_images = len(images)
    precomputed_data = precompute_keypoints_and_descriptors(images)
    connectivity_scores = []

    for i in tqdm(range(num_images), desc="Evaluating Starting Images"):
        total_matches = 0
        _, des1, _, _ = precomputed_data[i]

        for j in range(num_images):
            if i == j:
                continue

            _, des2, _, _ = precomputed_data[j]
            good_matches = match_keypoints(des1, des2)
            total_matches += len(good_matches)

        connectivity_scores.append((total_matches, i))

    connectivity_scores.sort(reverse=True, key=lambda x: x[0])
    best_start_index = connectivity_scores[0][1]

    print(f"Best starting image: {images[best_start_index][1]} with {connectivity_scores[0][0]} matches")

    ordered_images = [images[best_start_index]]
    used_indices = set()
    used_indices.add(best_start_index)
    previous_index = best_start_index

    for _ in tqdm(range(1, num_images), desc="Ordering Images"):
        best_match_count = -1
        best_image = None
        best_index = None

        last_kp, last_des, _, _ = precomputed_data[previous_index]

        for i in range(num_images):
            if i in used_indices:
                continue

            kp, des, img, filename = precomputed_data[i]

            good_matches = match_keypoints(last_des, des)
            H = get_homography_matrix(last_kp, kp, good_matches)

            if H is not None and len(good_matches) > best_match_count:
                best_match_count = len(good_matches)
                best_image = (img, filename)
                best_index = i

        if best_image is not None:
            ordered_images.append(best_image)
            used_indices.add(best_index)
            previous_index = best_index

    print("Image ordering completed.")
    return ordered_images

def create_video_from_images(images, output_filename, framerate):
    height, width, _ = images[0][0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, framerate, (width, height))

    for img, _ in images:
        out.write(img)

    out.release()
    print("Created output video")

def run(input_directory, output_filename, framerate):
    images = load_images_from_directory(input_directory)
    ordered_images = calculate_image_order(images)
    create_video_from_images(ordered_images, output_filename, framerate)

