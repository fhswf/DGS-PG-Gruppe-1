import os
import cv2
import numpy as np
import json
import glob
from pathlib import Path
from tqdm import tqdm
import argparse

# ---------------------------------------------------------
# 1. Imports / Klassen
# ---------------------------------------------------------
try:
    from pose_estimation_recognition_utils_rtmlib import RTMPoseEstimator2D
    from pose_estimation_recognition_utils_rtmlib import RTMPoseEstimationFrom3DFrame
except ImportError as e:
    print(f"WARNUNG: Externe Abhängigkeiten konnten nicht importiert werden: {e}")
    # Für Entwicklungszwecke könnten hier Mocks definiert werden
    pass

class DatasetGenerator:
    def __init__(self, source_dir, output_file_small, output_file_medium, baseline_m=0.06):
        self.source_dir = Path(source_dir)
        self.output_file_small = Path(output_file_small)
        self.output_file_medium = Path(output_file_medium)
        self.baseline = baseline_m
        
        # Data storage for the final JSON output
        self.data_small = {}
        self.data_medium = {}
        self.cnt_small = 0
        self.cnt_medium = 0
        
        # Constants from user
        self.CX_LEFT = 640
        self.CY_LEFT = 360
        self.FOCAL_LENGTH = 2710
        self.DISTANCE = 60 # Baseline

        # Initialisierung der Tools
        print("Initialisiere Pose Estimators...")
        try:
            # 1. Standard 2D Estimator (Explicitly requested by user)
            self.pose_estimator = RTMPoseEstimator2D(device="cuda")
            
            # 2. 3D Lifter (Replacement for SAD)
            self.lifter = RTMPoseEstimationFrom3DFrame(
                focal_length=self.FOCAL_LENGTH,
                distance=self.DISTANCE,
                cx_left=self.CX_LEFT,
                cy_left=self.CY_LEFT,
                with_confidence=True,
                device="cuda"
            )
        except NameError:
             print("Fehler: Klassen nicht definiert. Import fehlgeschlagen?")

    def scan_videos(self):
        """Findet alle Videodateien im Quellverzeichnis."""
        extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        video_files = []
        for ext in extensions:
            video_files.extend(list(self.source_dir.rglob(ext)))
        return sorted(video_files)

    def count_total_frames(self, video_files):
        """Zählt die Gesamtzahl der Frames in allen Videos."""
        total_frames = 0
        video_frames_map = {}
        
        print("Zähle Frames in allen Videos...")
        for video_path in tqdm(video_files):
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Warnung: Kann Video nicht öffnen: {video_path}")
                continue
            
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_frames_map[video_path] = n_frames
            total_frames += n_frames
            cap.release()
            
        return total_frames, video_frames_map

    def process_frame(self, frame):
        """
        Verarbeitet einen Frame:
        1. 2D: PoseEstimation Left + Right -> Mean (Mittelwert).
        2. 3D: RTMPoseEstimationFrom3DFrame.
        """
        
        # A. 3D Extraction (Using Stereo Frame)
        results_3d = self.lifter.extract_frame(frame)
        if not results_3d:
            # Wenn 3D fehlschlägt, geben wir meist auch nichts zurück, 
            # da das Dataset Lifting trainieren soll.
            return None

        # B. 2D Extraction (Explicit Mean per User Request)
        h, w, _ = frame.shape
        w_half = w // 2
        frame_left = frame[:, :w_half]
        frame_right = frame[:, w_half:]
        
        # Run Estimator on both sides
        res_left = self.pose_estimator(frame_left)
        res_right = self.pose_estimator(frame_right)
        
        # Helper to extract raw keypoints from result list
        def get_kps(res):
            if res and len(res) > 0:
                person = res[0]
                if isinstance(person, dict):
                    return person.get('keypoints')
                elif hasattr(person, 'keypoints'):
                    return person.keypoints
            return None

        kps_left = get_kps(res_left)
        kps_right = get_kps(res_right)

        # Build Output Dictionaries
        kp2d_dict = {}
        kp3d_dict = {}
        
        # 1. Calculate 2D Mean
        if kps_left is not None and kps_right is not None:
             # Assuming shapes match (usually 133, 2)
             # If lengths differ, take minimum? Usually standard model has fixed size.
             n_points = min(len(kps_left), len(kps_right))
             for i in range(n_points):
                 x_avg = (kps_left[i][0] + kps_right[i][0]) / 2.0
                 y_avg = (kps_left[i][1] + kps_right[i][1]) / 2.0
                 kp2d_dict[str(i)] = {"x": float(x_avg), "y": float(y_avg)}
                 
        elif kps_left is not None:
             # Fallback Left
             for i, p in enumerate(kps_left):
                 kp2d_dict[str(i)] = {"x": float(p[0]), "y": float(p[1])}
                 
        elif kps_right is not None:
             # Fallback Right
             for i, p in enumerate(kps_right):
                 kp2d_dict[str(i)] = {"x": float(p[0]), "y": float(p[1])}
        else:
            # Kein 2D gefunden -> Skip Frame
            return None

        # 2. Process 3D Keypoints
        for p in results_3d:
            kp3d_dict[str(p.id)] = {"x": float(p.x), "y": float(p.y), "z": float(p.z)}
            
        # Confidence
        # Prefer confidence from 3D lifter (which likely merges usage)
        max_id = 0
        if results_3d:
            max_id = max(max_id, max(p.id for p in results_3d))
            
        if kp2d_dict:
            max_id = max(max_id, max(int(k) for k in kp2d_dict.keys()))

        conf_array = np.zeros(max_id + 1)
        
        # Prefer 3D confidence (merged)
        for p in results_3d:
             if hasattr(p, 'confidence'):
                conf_array[p.id] = float(p.confidence)
             elif conf_2d_person is not None and len(conf_2d_person) > p.id:
                conf_array[p.id] = float(conf_2d_person[p.id])
                
        return {
            "keypoints_2d": kp2d_dict,
            "keypoints_3d": kp3d_dict,
            "confidence": conf_array,
            "intrinsics": self.FOCAL_LENGTH
        }

    def run(self, target_small=25000, target_medium=50000):
        video_files = self.scan_videos()
        total_frames_all_videos, video_frames_map = self.count_total_frames(video_files)
        
        print(f"Gefunden: {len(video_files)} Videos mit insgesamt {total_frames_all_videos} Frames.")
        
        # 3. Sampling Intervalle
        step_small = max(1, total_frames_all_videos // target_small)
        step_medium = max(1, total_frames_all_videos // target_medium)
        
        print(f"Sampling Step Small: {step_small}")
        print(f"Sampling Step Medium: {step_medium}")
        
        global_frame_counter = 0
        processed_count = 0
        
        # Ensure output directories exist
        self.output_file_small.parent.mkdir(parents=True, exist_ok=True)
        self.output_file_medium.parent.mkdir(parents=True, exist_ok=True)
        
        for video_path in video_files:
            cap = cv2.VideoCapture(str(video_path))
            n_frames = video_frames_map[video_path]
            
            pbar = tqdm(total=n_frames, desc=f"Verarbeite {video_path.name}")
            
            for f_idx in range(n_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check ob wir diesen Frame für eines der Datasets brauchen
                need_for_small = (global_frame_counter % step_small == 0)
                need_for_medium = (global_frame_counter % step_medium == 0)
                
                if need_for_small or need_for_medium:
                    result = self.process_frame(frame)
                    
                    if result:
                        # Qualitätsprüfung
                        if np.mean(result['confidence']) < 0.5:
                            # Skip schlechte Frames
                            pass 
                        else:
                            # Speichern
                            self._save_result(result, global_frame_counter, need_for_small, need_for_medium, video_path.name)
                            processed_count += 1
                
                global_frame_counter += 1
                pbar.update(1)
            
            pbar.close()
            cap.release()
            
        # Write final JSON files
        print(f"Saving {len(self.data_small)} samples to {self.output_file_small}...")
        with open(self.output_file_small, 'w') as f:
            json.dump(self.data_small, f, indent=2)
            
        print(f"Saving {len(self.data_medium)} samples to {self.output_file_medium}...")
        with open(self.output_file_medium, 'w') as f:
            json.dump(self.data_medium, f, indent=2)
            
        print(f"Fertig! Verarbeitete Samples: {processed_count}")

    def _save_result(self, result, frame_id, save_small, save_medium, source_video):
        # Result is already in dict format from process_frame
        
        packet = {
            "keypoints_2d": result['keypoints_2d'],
            "keypoints_3d": result['keypoints_3d'],
            "source_video": source_video,
            "frame_id": int(frame_id),
            "confidence": result['confidence'].tolist(),
             "meta": {
                "camera": {
                  "baseline_m": self.baseline,
                  "intrinsics": float(result['intrinsics'])
                }
            }
        }

        if save_small:
            self.data_small[str(self.cnt_small)] = packet
            self.cnt_small += 1
        
        if save_medium:
            self.data_medium[str(self.cnt_medium)] = packet
            self.cnt_medium += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo Dataset Generator")
    parser.add_argument("--source", type=str, required=True, help="Pfad zu den Videos")
    parser.add_argument("--out_small", type=str, default="dataset_small.json", help="Output Pfad Small (JSON)")
    parser.add_argument("--out_medium", type=str, default="dataset_medium.json", help="Output Pfad Medium (JSON)")
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(args.source, args.out_small, args.out_medium)
    generator.run()
