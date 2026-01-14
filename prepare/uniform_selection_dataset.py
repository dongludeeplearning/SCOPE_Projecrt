import cv2
import numpy as np
import os
import glob
import dlib
import argparse  # Added for command line args
from tqdm import tqdm

# Initialize dlib face detector (HOG-based)
# Global initialization to avoid overhead
detector = dlib.get_frontal_face_detector()

def crop_face_with_margin(frame, margin=0.2):
    """
    Detects face and crops it with a margin.
    Returns cropped frame if face detected, else returns original frame.
    """
    if frame is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return frame # Return original if no face found
    
    # Assume largest face is the target
    face = max(faces, key=lambda rect: rect.width() * rect.height())
    
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    
    # Calculate margin
    m_w = int(w * margin)
    m_h = int(h * margin)
    
    # Expand box with boundary checks
    x1 = max(0, x - m_w)
    y1 = max(0, y - m_h)
    x2 = min(frame.shape[1], x + w + m_w)
    y2 = min(frame.shape[0], y + h + m_h)
    
    # Ensure crop is valid
    if x2 > x1 and y2 > y1:
        return frame[y1:y2, x1:x2]
    else:
        return frame

def extract_uniform_frames(source_path, out_path, num_keyframes=60, save_keyframe=True, crop_face=False):
    """
    Uniformly samples frames from a video and saves them.
    Optionally crops faces.
    """
    video_name = os.path.basename(source_path).split('.')[0]
    
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {source_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Generate indices for uniform sampling
    if total_frames <= num_keyframes:
        selected_indices = list(range(total_frames))
    else:
        selected_indices = np.linspace(0, total_frames - 1, num_keyframes, dtype=int).tolist()
    
    selected_indices_set = set(selected_indices)
    
    if save_keyframe:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        frame_idx = 0
        saved_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx in selected_indices_set:
                frame_to_save = frame
                
                # Apply Face Cropping if enabled
                if crop_face:
                    cropped = crop_face_with_margin(frame, margin=0.2)
                    if cropped is not None:
                        frame_to_save = cropped
                
                # Save frame
                keyframe_filename = os.path.join(out_path, f"{frame_idx:04d}.jpg")
                cv2.imwrite(keyframe_filename, frame_to_save)
                saved_count += 1
                
                if saved_count >= len(selected_indices_set):
                    pass 
            
            frame_idx += 1

    cap.release()


def process_dataset(dataset_root, out_root, num_keyframes=60, crop_face=False):
    """
    Process all .avi files in Train, Test, and Validation folders.
    """
    for split in ["Test", "Validation", "Train"]:
        split_path = os.path.join(dataset_root, split)
        
        if not os.path.exists(split_path):
            print(f"Warning: Split folder {split_path} does not exist. Skipping.")
            continue
            
        avi_files = glob.glob(os.path.join(split_path, "**", "*.avi"), recursive=True)
        
        print(f"Processing {len(avi_files)} videos in {split} | Crop Face: {crop_face}")

        for video_path in tqdm(avi_files, desc=f"Processing {split}", unit="video"):
            video_name = os.path.basename(video_path).split('.')[0]
            out_path = os.path.join(out_root, split, video_name)
            
            extract_uniform_frames(video_path, out_path, num_keyframes=num_keyframes, save_keyframe=True, crop_face=crop_face)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Uniform Frames from DAiSEE")
    parser.add_argument("--crop", action="store_true", help="Enable face cropping with 20% margin")
    parser.add_argument("--root", type=str, default="/mnt/pub/Cognitive/DAiSEE/DataSet", help="Input Dataset Root")
    parser.add_argument("--out", type=str, default="/mnt/pub/Cognitive/DAiSEE_Process/DataSet_Uniform", help="Output Dataset Root")
    args = parser.parse_args()

    # If crop is enabled, append '_Cropped' to output folder name to avoid mixing
    if args.crop:
        if "Uniform" in args.out and "Cropped" not in args.out:
            args.out = args.out + "_Cropped"
    
    print("Starting Uniform Sampling Process...")
    print(f"Input: {args.root}")
    print(f"Output: {args.out}")
    print(f"Face Cropping: {args.crop}")
    
    process_dataset(args.root, args.out, num_keyframes=60, crop_face=args.crop)
    print("Processing Complete.")
