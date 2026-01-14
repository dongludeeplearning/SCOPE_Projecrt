"""
Uniform Frame Selection with RetinaFace Face Detection (v2)
Improved version using RetinaFace for better face detection accuracy.

Features:
- Uses RetinaFace (via insightface) instead of dlib for better accuracy
- Handles side faces, occlusions, and various lighting conditions better
- Adds statistics tracking for detection success rate
- Better error handling and logging

Author: Lu Dong
Date: 2026-01-14
"""

import cv2
import numpy as np
import os
import glob
import argparse
from tqdm import tqdm
import json
from collections import defaultdict

# Try to import RetinaFace
try:
    from insightface import app
    from insightface.utils import face_align
    RETINAFACE_AVAILABLE = True
except ImportError:
    try:
        # Alternative: use retinaface package directly
        from retinaface import RetinaFace
        RETINAFACE_AVAILABLE = True
        RETINAFACE_MODE = 'retinaface_package'
    except ImportError:
        RETINAFACE_AVAILABLE = False
        RETINAFACE_MODE = None
        print("Warning: RetinaFace not available. Please install:")
        print("  pip install insightface onnxruntime")
        print("  OR")
        print("  pip install retinaface")

# Global detector initialization
detector = None
detector_app = None
detector_mode = None  # 'insightface' or 'retinaface_package'

def init_retinaface_detector():
    """Initialize RetinaFace detector"""
    global detector, detector_app, detector_mode
    
    if not RETINAFACE_AVAILABLE:
        raise ImportError("RetinaFace is not available. Please install insightface or retinaface package.")
    
    # Try insightface first (recommended)
    try:
        detector_app = app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        detector_app.prepare(ctx_id=0, det_size=(640, 640))
        detector_mode = 'insightface'
        print("RetinaFace detector initialized (insightface)")
        return
    except Exception as e:
        print(f"Warning: Failed to initialize insightface: {e}")
        print("Trying retinaface package as fallback...")
    
    # Fallback to retinaface package
    try:
        from retinaface import RetinaFace
        detector = RetinaFace.build_model()
        detector_mode = 'retinaface_package'
        print("RetinaFace detector initialized (retinaface package)")
    except Exception as e:
        raise ImportError(f"Failed to initialize RetinaFace: {e}")


def crop_face_with_margin_retinaface(frame, margin=0.2, min_face_size=20):
    """
    Detects face using RetinaFace and crops it with a margin.
    Returns cropped frame if face detected, else returns original frame.
    
    Args:
        frame: Input BGR image (numpy array)
        margin: Margin ratio around face (default 0.2 = 20%)
        min_face_size: Minimum face size in pixels (default 20)
    
    Returns:
        Cropped frame if face detected, else original frame
    """
    if frame is None:
        return None, False
    
    if detector is None and detector_app is None:
        return frame, False
    
    try:
        faces = None
        
        if detector_mode == 'retinaface_package' and detector is not None:
            # Using retinaface package
            from retinaface import RetinaFace
            faces_dict = RetinaFace.detect_faces(frame)
            if faces_dict:
                # Convert to list format: [(bbox, landmarks, confidence), ...]
                # facial_area format: [x1, y1, x2, y2]
                faces = []
                for key, face_data in faces_dict.items():
                    facial_area = face_data['facial_area']  # [x1, y1, x2, y2]
                    landmarks = face_data.get('landmarks', {})
                    confidence = face_data['score']
                    faces.append((facial_area, landmarks, confidence))
        elif detector_mode == 'insightface' and detector_app is not None:
            # Using insightface
            faces = detector_app.get(frame)
        else:
            return frame, False
        
        if faces is None or len(faces) == 0:
            return frame, False
        
        # Select face with highest confidence (or largest if using retinaface package)
        if detector_mode == 'retinaface_package':
            # Sort by confidence (score)
            faces = sorted(faces, key=lambda x: x[2] if len(x) > 2 else 0, reverse=True)
            bbox = faces[0][0]  # (x1, y1, x2, y2)
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
        else:
            # insightface format: bbox is [x1, y1, x2, y2]
            # Sort by bbox area (largest face)
            faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
            bbox = faces[0].bbox
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            w = x2 - x1
            h = y2 - y1
        
        # Filter out too small faces
        if w < min_face_size or h < min_face_size:
            return frame, False
        
        # Calculate margin
        m_w = int(w * margin)
        m_h = int(h * margin)
        
        # Expand box with boundary checks
        x1_expanded = max(0, x1 - m_w)
        y1_expanded = max(0, y1 - m_h)
        x2_expanded = min(frame.shape[1], x2 + m_w)
        y2_expanded = min(frame.shape[0], y2 + m_h)
        
        # Ensure crop is valid
        if x2_expanded > x1_expanded and y2_expanded > y1_expanded:
            cropped = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
            return cropped, True
        else:
            return frame, False
            
    except Exception as e:
        print(f"Error in face detection: {e}")
        return frame, False


def extract_uniform_frames(source_path, out_path, num_keyframes=60, save_keyframe=True, 
                          crop_face=False, stats=None):
    """
    Uniformly samples frames from a video and saves them.
    Optionally crops faces using RetinaFace.
    
    Args:
        source_path: Path to input video file
        out_path: Output directory for frames
        num_keyframes: Number of frames to extract
        save_keyframe: Whether to save frames
        crop_face: Whether to crop faces
        stats: Statistics dictionary to update
    
    Returns:
        Dictionary with statistics for this video
    """
    video_name = os.path.basename(source_path).split('.')[0]
    video_stats = {
        'total_frames': 0,
        'faces_detected': 0,
        'faces_not_detected': 0,
        'crop_enabled': crop_face
    }
    
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {source_path}")
        return video_stats

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
                video_stats['total_frames'] += 1
                
                # Apply Face Cropping if enabled
                if crop_face:
                    cropped, face_detected = crop_face_with_margin_retinaface(frame, margin=0.2)
                    if face_detected:
                        frame_to_save = cropped
                        video_stats['faces_detected'] += 1
                    else:
                        video_stats['faces_not_detected'] += 1
                
                # Save frame
                keyframe_filename = os.path.join(out_path, f"{frame_idx:04d}.jpg")
                cv2.imwrite(keyframe_filename, frame_to_save)
                saved_count += 1
                
                if saved_count >= len(selected_indices_set):
                    break
            
            frame_idx += 1

    cap.release()
    
    # Update global stats if provided
    if stats is not None:
        stats['total_videos'] += 1
        stats['total_frames'] += video_stats['total_frames']
        stats['faces_detected'] += video_stats['faces_detected']
        stats['faces_not_detected'] += video_stats['faces_not_detected']
        stats['videos_with_faces'] += 1 if video_stats['faces_detected'] > 0 else 0
        stats['videos_without_faces'] += 1 if video_stats['faces_detected'] == 0 else 0
    
    return video_stats


def process_dataset(dataset_root, out_root, num_keyframes=60, crop_face=False, save_stats=True):
    """
    Process all .avi files in Train, Test, and Validation folders.
    
    Args:
        dataset_root: Root directory containing Train/Test/Validation folders
        out_root: Output root directory
        num_keyframes: Number of frames to extract per video
        crop_face: Whether to crop faces
        save_stats: Whether to save statistics to JSON file
    """
    # Initialize statistics
    global_stats = {
        'total_videos': 0,
        'total_frames': 0,
        'faces_detected': 0,
        'faces_not_detected': 0,
        'videos_with_faces': 0,
        'videos_without_faces': 0,
        'crop_enabled': crop_face,
        'detector': 'RetinaFace'
    }
    
    split_stats = defaultdict(lambda: {
        'total_videos': 0,
        'total_frames': 0,
        'faces_detected': 0,
        'faces_not_detected': 0,
        'videos_with_faces': 0,
        'videos_without_faces': 0
    })
    
    for split in ["Test", "Validation", "Train"]:
        split_path = os.path.join(dataset_root, split)
        
        if not os.path.exists(split_path):
            print(f"Warning: Split folder {split_path} does not exist. Skipping.")
            continue
            
        avi_files = glob.glob(os.path.join(split_path, "**", "*.avi"), recursive=True)
        
        print(f"\n{'='*60}")
        print(f"Processing {len(avi_files)} videos in {split} | Crop Face: {crop_face}")
        print(f"{'='*60}")

        for video_path in tqdm(avi_files, desc=f"Processing {split}", unit="video"):
            video_name = os.path.basename(video_path).split('.')[0]
            out_path = os.path.join(out_root, split, video_name)
            
            video_stats = extract_uniform_frames(
                video_path, out_path, 
                num_keyframes=num_keyframes, 
                save_keyframe=True, 
                crop_face=crop_face,
                stats=global_stats
            )
            
            # Update split stats
            split_stats[split]['total_videos'] += 1
            split_stats[split]['total_frames'] += video_stats['total_frames']
            split_stats[split]['faces_detected'] += video_stats['faces_detected']
            split_stats[split]['faces_not_detected'] += video_stats['faces_not_detected']
            if video_stats['faces_detected'] > 0:
                split_stats[split]['videos_with_faces'] += 1
            else:
                split_stats[split]['videos_without_faces'] += 1
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Processing Statistics Summary")
    print(f"{'='*60}")
    print(f"Total Videos: {global_stats['total_videos']}")
    print(f"Total Frames Processed: {global_stats['total_frames']}")
    if crop_face:
        print(f"Faces Detected: {global_stats['faces_detected']}")
        print(f"Faces Not Detected: {global_stats['faces_not_detected']}")
        if global_stats['total_frames'] > 0:
            detection_rate = global_stats['faces_detected'] / global_stats['total_frames'] * 100
            print(f"Detection Rate: {detection_rate:.2f}%")
        print(f"Videos with Faces: {global_stats['videos_with_faces']}")
        print(f"Videos without Faces: {global_stats['videos_without_faces']}")
    
    # Print per-split statistics
    print(f"\nPer-Split Statistics:")
    for split, stats in split_stats.items():
        print(f"\n{split}:")
        print(f"  Videos: {stats['total_videos']}")
        print(f"  Frames: {stats['total_frames']}")
        if crop_face:
            print(f"  Faces Detected: {stats['faces_detected']}")
            print(f"  Faces Not Detected: {stats['faces_not_detected']}")
            if stats['total_frames'] > 0:
                rate = stats['faces_detected'] / stats['total_frames'] * 100
                print(f"  Detection Rate: {rate:.2f}%")
            print(f"  Videos with Faces: {stats['videos_with_faces']}")
            print(f"  Videos without Faces: {stats['videos_without_faces']}")
    
    # Save statistics to JSON
    if save_stats:
        stats_file = os.path.join(out_root, 'detection_statistics.json')
        stats_data = {
            'global': global_stats,
            'per_split': dict(split_stats)
        }
        with open(stats_file, 'w') as f:
            json.dump(stats_data, f, indent=2)
        print(f"\nStatistics saved to: {stats_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Uniform Frames from DAiSEE using RetinaFace (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Without face cropping:
  python uniform_selection_dataset_v2.py --root /path/to/dataset --out /path/to/output
  
  # With face cropping:
  python uniform_selection_dataset_v2.py --crop --root /path/to/dataset --out /path/to/output
  
Installation:
  pip install insightface onnxruntime
  OR
  pip install retinaface
        """
    )
    parser.add_argument("--crop", action="store_true", 
                       help="Enable face cropping with 20%% margin using RetinaFace")
    parser.add_argument("--root", type=str, 
                       default="/mnt/pub/Cognitive/DAiSEE/DataSet", 
                       help="Input Dataset Root")
    parser.add_argument("--out", type=str, 
                       default="/mnt/pub/Cognitive/DAiSEE_Process/DataSet_Uniform_RetinaFace", 
                       help="Output Dataset Root")
    parser.add_argument("--num_keyframes", type=int, default=60,
                       help="Number of keyframes to extract per video (default: 60)")
    parser.add_argument("--no_stats", action="store_true",
                       help="Disable statistics saving")
    args = parser.parse_args()

    # Initialize RetinaFace detector if cropping is enabled
    if args.crop:
        print("Initializing RetinaFace detector...")
        try:
            init_retinaface_detector()
        except Exception as e:
            print(f"Error initializing RetinaFace: {e}")
            print("Please install RetinaFace:")
            print("  pip install insightface onnxruntime")
            print("  OR")
            print("  pip install retinaface")
            exit(1)
    
    # If crop is enabled, append '_RetinaFace' to output folder name to avoid mixing
    if args.crop:
        if "RetinaFace" not in args.out:
            if "Uniform" in args.out:
                args.out = args.out.replace("Uniform", "Uniform_RetinaFace")
            else:
                args.out = args.out + "_RetinaFace"
    
    print("\n" + "="*60)
    print("Uniform Frame Selection with RetinaFace (v2)")
    print("="*60)
    print(f"Input: {args.root}")
    print(f"Output: {args.out}")
    print(f"Face Cropping: {args.crop}")
    print(f"Number of Keyframes: {args.num_keyframes}")
    print(f"Detector: {'RetinaFace' if args.crop else 'None'}")
    print("="*60 + "\n")
    
    process_dataset(
        args.root, 
        args.out, 
        num_keyframes=args.num_keyframes, 
        crop_face=args.crop,
        save_stats=not args.no_stats
    )
    print("\nProcessing Complete.")

