import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import dlib
import glob
from tqdm import tqdm  # For progress bar

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_face_roi(gray_frame):
    """
    Detects face and extracts regions of interest (ROI) for micro-expression analysis.
    
    Parameters:
        gray_frame (numpy array): Grayscale frame from video.
    
    Returns:
        list: List of facial keypoint coordinates.
    """
    faces = detector(gray_frame)
    if len(faces) == 0:
        return None  # No face detected

    shape = predictor(gray_frame, faces[0])
    landmarks = [(p.x, p.y) for p in shape.parts()]
    return landmarks


def plot_micro_expression_scores(frame_scores, selected_frame_indices, output_dir, video_name):
    """
    Plots the micro-expression scores across all frames with keyframes highlighted.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_indices = [x[0] for x in frame_scores]
    scores = [x[1] for x in frame_scores]
    
    plt.figure(figsize=(10, 5))
    plt.plot(frame_indices, scores, color='blue', label='Frame Score')
    plt.scatter(selected_frame_indices, [scores[frame_indices.index(f)] for f in selected_frame_indices], color='red', label='Key Frames', marker='o')
    plt.xlabel("Frame Index")
    plt.ylabel("Micro-Expression Score")
    plt.title("Micro-Expression Analysis Over Frames")
    plt.legend()
    # plt.show()

    plot_filename = os.path.join(output_dir, video_name+"_plot.png")
    plt.savefig(plot_filename)
    plt.close()
   


def compute_micro_expression_flow(source_path, out_path,num_keyframes, save_keyframe=True, save_plot=True, save_record=True):
    """
    Computes optical flow for micro-expression detection and optionally saves keyframes, plots, and a record in a JSON file.
    
    Parameters:
        video_path (str): Path to the input video.
        num_keyframes (int): Number of keyframes to select.
        save_keyframe (bool): Whether to save the selected keyframes.
        save_plot (bool): Whether to save a plot of frame scores.
        save_record (bool): Whether to save the results as a JSON record.
    
    Returns:
        tuple: List of all frames, frame scores, selected frame indices, and selected frame scores.
    """
    video_name = os.path.basename(source_path).split('.')[0]
    
    cap = cv2.VideoCapture(source_path)
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the video file.")
        return [], [], [], []

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_scores = []
    frame_idx = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        landmarks = get_face_roi(gray)  # Replace with actual face ROI detection function

        if landmarks:
            motion_scores = []
            for (x, y) in landmarks:
                if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                    roi_prev = prev_gray[y-2:y+2, x-2:x+2]
                    roi_current = gray[y-2:y+2, x-2:x+2]
                    if roi_prev.shape == roi_current.shape and roi_prev.size > 0:
                        flow = cv2.calcOpticalFlowFarneback(roi_prev, roi_current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        motion_magnitude = np.sum(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
                        motion_scores.append(motion_magnitude)
            frame_scores.append((frame_idx, np.sum(motion_scores)))
            frames.append(frame)

        prev_gray = gray
        frame_idx += 1

    cap.release()

    # Sort frames by motion magnitude score in descending order and select top num_keyframes
    frame_scores_sorted = sorted(frame_scores, key=lambda x: x[1], reverse=True)
    selected_frames = frame_scores_sorted[:num_keyframes]
    
    # Sort selected frames by frame index in ascending order
    selected_frames_sorted = sorted(selected_frames, key=lambda x: x[0])


    selected_frame_indices = [x[0] for x in frame_scores_sorted[:num_keyframes]]

    # Optionally save the selected keyframes
    video_name = os.path.basename(source_path).split('.')[0]
    

    if save_keyframe:
        for idx, _ in selected_frames_sorted:
            keyframe_filename = os.path.join(out_path, f"{idx:04d}.jpg")
            cv2.imwrite(keyframe_filename, frames[idx])
        # print(f"save selected {num_keyframes} frames successfully")

    # save all trends for global perspective_
    if save_plot:
        plot_micro_expression_scores(frame_scores, selected_frame_indices, out_path, video_name)
        # print("save plot successfully")


    # Optionally save the results as a JSON record
    if save_record:
        record = {
        "video_name": video_name,
        "frame_scores": [(int(x[0]), float(x[1])) for x in frame_scores],
        "selected_frames": [(int(x[0]), float(x[1])) for x in selected_frames_sorted]
        }
        record_filename = os.path.join(out_path , f"{video_name}_record.json")
        with open(record_filename, 'w') as f:
            json.dump(record, f, indent=4)
        print("save json record successfully")

    # return frames, frame_scores, selected_frames_sorted
    # return selected_frames_sorted



def process_dataset(dataset_root, out_root, num_keyframes=60):
    """
    Process all .avi files in Train, Test, and Validation folders using compute_micro_expression_flow.
    
    Parameters:
        dataset_root (str): Root directory containing Train, Test, and Validation folders.
        num_keyframes (int): Number of keyframes to select.
    """
    # Iterate through Train, Test, and Validation folders
    for split in ["Test", "Validation","Train" ]:
        split_path = os.path.join(dataset_root, split)
        
        # Glob all .avi files recursively within the current split folder
        avi_files = glob.glob(os.path.join(split_path, "**", "*.avi"), recursive=True)
        
        print(f"Processing {len(avi_files)} videos in {split}")  # Test 1720; Val 1536; Train 4976

        # Process each .avi file
        # for video_path in avi_files:
        for video_path in tqdm(avi_files, desc=f"Processing {split}", unit="video"):

            video_name = os.path.basename(video_path).split('.')[0]  # Extract video name

            out_path = os.path.join(out_root, split, video_name)  # Define output directory

            print(f"Processing {video_path}...")
            
            # Ensure the output directory exists
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            # Call compute_micro_expression_flow
            compute_micro_expression_flow(video_path, out_path, num_keyframes=num_keyframes, 
                                          save_keyframe=True, save_plot=True, save_record=True)
        #     print(f"Finished processing {video_path}\n")

# Example usage
dataset_root = "/mnt/pub/Cognitive Dataset/DAiSEE/DataSet"
out_root = "/mnt/pub/Cognitive Dataset/DAiSEE_Process/DataSet"
process_dataset(dataset_root, out_root, num_keyframes=60)
