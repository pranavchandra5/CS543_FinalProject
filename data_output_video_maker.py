import cv2
import numpy as np
import os

videoName = "Bluemlisalphutte Flyover"
algorithms = ["SIFT_Ransac_Naive", "SIFT_Ransac_Advanced", "Graph_TSP_MST", "Graph_TSP_Nearest_Neighbor"]

output_file = "temp.mp4"
frame_rate = 15
slide_duration = 5

first_video = f"{videoName}_{algorithms[0]}.mp4"
cap = cv2.VideoCapture(first_video)
if not cap.isOpened():
    raise FileNotFoundError(f"Unable to open the first video file: {first_video}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)
cap.release()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video = cv2.VideoWriter(output_file, fourcc, frame_rate, frame_size)

def create_text_slide(text, size, duration, fps):
    slide = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (size[0] - text_size[0]) // 2
    text_y = (size[1] + text_size[1]) // 2
    cv2.putText(slide, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
    return [slide.copy() for _ in range(int(duration * fps))]

for algorithm in algorithms:
    title = algorithm
    slide_frames = create_text_slide(title, frame_size, slide_duration, frame_rate)
    for frame in slide_frames:
        output_video.write(frame)

    filename = f"{videoName}_{algorithm}.mp4"
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print(f"Warning: Unable to open video file: {filename}")
        continue

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output_video.write(frame)

    cap.release()

output_video.release()

print(f"Final Temp video saved as {output_file}")

videos = {
    "Ground Truth": f"{videoName}_gt.mp4",
    "Algorithms": "temp.mp4"
}

output_file = f"{videoName}_output.mp4"
frame_rate = 15
slide_duration = 5

first_video = "temp.mp4"
cap = cv2.VideoCapture(first_video)
if not cap.isOpened():
    raise FileNotFoundError(f"Unable to open the first video file: {first_video}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)
cap.release()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video = cv2.VideoWriter(output_file, fourcc, frame_rate, frame_size)

def create_text_slide(text, size, duration, fps):
    slide = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (size[0] - text_size[0]) // 2
    text_y = (size[1] + text_size[1]) // 2
    cv2.putText(slide, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    return [slide.copy() for _ in range(int(duration * fps))]

for title, filename in videos.items():
    slide_frames = create_text_slide(title, frame_size, slide_duration, frame_rate)
    for frame in slide_frames:
        output_video.write(frame)

    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print(f"Warning: Unable to open video file: {filename}")
        continue

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output_video.write(frame)

    cap.release()

output_video.release()

print(f"Final video saved as {output_file}")
