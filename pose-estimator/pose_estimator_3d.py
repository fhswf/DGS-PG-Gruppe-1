"""
Pose3DConverter: A class that converts existing 2D pose data to 3D using AI lifting

This module takes the 2D pose output from PoseEstimator2D and converts it to 3D
using machine learning-based pose lifting.
"""

import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict
from dataclasses import dataclass
import json
import torch
import torch.nn as nn

class PoseLiftingModel(nn.Module):
    """
    Neural Network for lifting 2D poses to 3D.
    Input: 266 values (133 keypoints * 2 coordinates)
    Output: 399 values (133 keypoints * 3 coordinates)
    """
    def __init__(self, input_dim=266, hidden_dims=[512, 512, 256], output_dim=399):
        super(PoseLiftingModel, self).__init__()
        
        # Vereinfachtes Netzwerk ohne BatchNorm für Stabilität
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)


@dataclass
class Pose3DResult:
    """
    Data class to store 3D pose estimation results.
    """
    frame_idx: int
    keypoints_3d: np.ndarray  # (num_persons, 133, 3)
    keypoints_2d: np.ndarray  # (num_persons, 133, 2) - original 2D input
    scores_3d: np.ndarray     # (num_persons, 133)
    bboxes_3d: np.ndarray     # (num_persons, 7)
    num_persons: int
    method: str
    confidence: float


class Pose3DConverter:
    """
    Converts 2D pose data to 3D using AI-based lifting.
    
    This class takes the JSON output from PoseEstimator2D and converts it to 3D poses
    using machine learning to "guess" the missing depth information.
    """
    
    def __init__(
        self,
        lifting_method: str = 'geometric',  # 'neural_network', 'geometric', 'hybrid' - geändert zu geometric als Default
        model_path: Optional[Union[str, Path]] = None,
        image_width: int = 1920,  # Standard image dimensions for normalization
        image_height: int = 1080
    ):
        self.lifting_method = lifting_method
        self.image_width = image_width
        self.image_height = image_height
        
        # Initialize the lifting model
        self._initialize_lifting_model(model_path)
        
        print(f"Initialized 3D Pose Converter with method: {lifting_method}")
    
    def _initialize_lifting_model(self, model_path: Optional[Union[str, Path]] = None):
        """Initialize the AI model for 2D-to-3D conversion"""
        if self.lifting_method in ['neural_network', 'hybrid']:
            self.lifting_model = PoseLiftingModel()
            self.lifting_model.eval()  # Immer im Evaluation-Modus
            
            if model_path and Path(model_path).exists():
                try:
                    checkpoint = torch.load(model_path, map_location='cpu')
                    if 'model_state_dict' in checkpoint:
                        self.lifting_model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.lifting_model.load_state_dict(checkpoint)
                    print(f"Loaded pre-trained lifting model from: {model_path}")
                    self.model_loaded = True
                except Exception as e:
                    print(f"Failed to load pre-trained model: {e}")
                    self.model_loaded = False
            else:
                print("No pre-trained model found. Using geometric fallback.")
                self.model_loaded = False
                self.lifting_method = 'geometric'  # Fallback auf geometrische Methode
        else:
            self.model_loaded = False
    
    def convert_2d_to_3d(
        self, 
        keypoints_2d: np.ndarray, 
        scores_2d: np.ndarray,
        image_size: Tuple[int, int] = None
    ) -> Pose3DResult:
        """
        Convert 2D keypoints to 3D using AI lifting.
        
        Args:
            keypoints_2d: Array of shape (num_persons, 133, 2) from PoseEstimator2D
            scores_2d: Array of shape (num_persons, 133) from PoseEstimator2D
            image_size: (width, height) of original image for normalization
            
        Returns:
            Pose3DResult with 3D coordinates
        """
        if len(keypoints_2d) == 0:
            return Pose3DResult(
                frame_idx=0,
                keypoints_3d=np.empty((0, 133, 3)),
                keypoints_2d=keypoints_2d,
                scores_3d=np.empty((0, 133)),
                bboxes_3d=np.empty((0, 7)),
                num_persons=0,
                method=self.lifting_method,
                confidence=0.0
            )
        
        # Use provided image size or default
        if image_size is None:
            w, h = self.image_width, self.image_height
        else:
            w, h = image_size
        
        # Convert each person's 2D pose to 3D
        keypoints_3d_list = []
        scores_3d_list = []
        
        for person_idx in range(len(keypoints_2d)):
            kpts_2d = keypoints_2d[person_idx]
            scores = scores_2d[person_idx]
            
            if self.lifting_method == 'neural_network' and self.model_loaded:
                # Use neural network for 3D lifting
                kpts_3d = self._neural_lift_2d_to_3d(kpts_2d, (h, w))
            elif self.lifting_method == 'geometric' or not self.model_loaded:
                # Use geometric/heuristic method
                kpts_3d = self._geometric_lift_2d_to_3d(kpts_2d, scores, (h, w))
            else:  # hybrid
                kpts_3d_nn = self._neural_lift_2d_to_3d(kpts_2d, (h, w))
                kpts_3d_geo = self._geometric_lift_2d_to_3d(kpts_2d, scores, (h, w))
                kpts_3d = (kpts_3d_nn + kpts_3d_geo) / 2
            
            # Apply anatomical constraints for realistic poses
            kpts_3d = self._apply_anatomical_constraints(kpts_3d, scores)
            
            keypoints_3d_list.append(kpts_3d)
            # Reduce confidence for 3D points (depth is estimated)
            scores_3d_list.append(scores * 0.8)
        
        keypoints_3d = np.array(keypoints_3d_list)
        scores_3d = np.array(scores_3d_list)
        
        # Calculate 3D bounding boxes
        bboxes_3d = self._calculate_3d_bboxes(keypoints_3d, scores_3d)
        
        # Overall confidence
        overall_confidence = float(np.mean(scores_3d)) if len(scores_3d) > 0 else 0.0
        
        return Pose3DResult(
            frame_idx=0,
            keypoints_3d=keypoints_3d,
            keypoints_2d=keypoints_2d,
            scores_3d=scores_3d,
            bboxes_3d=bboxes_3d,
            num_persons=len(keypoints_2d),
            method=self.lifting_method,
            confidence=overall_confidence
        )
    
    def _neural_lift_2d_to_3d(self, keypoints_2d: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """Use neural network to lift 2D points to 3D"""
        h, w = image_shape
        
        # Normalize 2D coordinates to [0, 1] range
        kpts_normalized = keypoints_2d.copy()
        kpts_normalized[:, 0] /= w  # x normalization
        kpts_normalized[:, 1] /= h  # y normalization
        
        # Flatten for neural network input
        input_flat = kpts_normalized.flatten()
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_flat).unsqueeze(0)
            output_3d_flat = self.lifting_model(input_tensor).numpy().flatten()
        
        # Reshape to (133, 3) and denormalize
        keypoints_3d = output_3d_flat.reshape(133, 3)
        keypoints_3d[:, 0] *= w  # x
        keypoints_3d[:, 1] *= h  # y
        # z remains in normalized coordinates
        
        return keypoints_3d
    
    def _geometric_lift_2d_to_3d(self, keypoints_2d: np.ndarray, scores: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """Use geometric heuristics to estimate 3D from 2D"""
        h, w = image_shape
        keypoints_3d = np.zeros((133, 3))
        keypoints_3d[:, :2] = keypoints_2d  # Copy x,y coordinates
        
        # Estimate body size for depth scaling
        body_size = self._estimate_body_size(keypoints_2d, scores, image_shape)
        
        # KI-basierte Heuristik: "Errate" die Tiefe basierend auf Anatomie
        for i in range(133):
            if scores[i] > 0.3:  # Nur für confident points
                # Körper-Punkte (0-22)
                if i < 23:
                    keypoints_3d[i, 2] = self._estimate_body_point_depth(i, keypoints_2d, body_size)
                # Gesichts-Punkte (23-91) 
                elif i < 92:
                    keypoints_3d[i, 2] = self._estimate_face_point_depth(i, keypoints_2d, body_size)
                # Hände-Punkte (92-133)
                else:
                    keypoints_3d[i, 2] = self._estimate_hand_point_depth(i, keypoints_2d, body_size)
        
        return keypoints_3d
    
    def _estimate_body_size(self, keypoints: np.ndarray, scores: np.ndarray, image_shape: Tuple[int, int]) -> float:
        """Estimate body size for depth scaling"""
        h, w = image_shape
        
        # Versuche Schulter-Hüfte Abstand zu finden
        shoulder_points = [5, 6]  # Left/right shoulder
        hip_points = [11, 12]     # Left/right hip
        
        valid_shoulders = [keypoints[i] for i in shoulder_points if scores[i] > 0.3]
        valid_hips = [keypoints[i] for i in hip_points if scores[i] > 0.3]
        
        if valid_shoulders and valid_hips:
            avg_shoulder = np.mean(valid_shoulders, axis=0)
            avg_hip = np.mean(valid_hips, axis=0)
            torso_height = abs(avg_shoulder[1] - avg_hip[1])
            return torso_height / h  # Normalized body size
        else:
            return 0.3  # Default size
    
    def _estimate_body_point_depth(self, point_idx: int, keypoints: np.ndarray, body_size: float) -> float:
        """KI-Heuristik: Errate Tiefe für Körperpunkte"""
        # Basierend auf anatomischem Wissen
        depth_rules = {
            # Shoulders - medium depth
            5: 0.0, 6: 0.0,
            # Elbows - forward
            7: 0.15, 8: 0.15,
            # Wrists - more forward
            9: 0.25, 10: 0.25,
            # Hips - medium depth
            11: 0.0, 12: 0.0,
            # Knees - backward
            #13: -0.1, 14: -0.1,
            # Ankles - more backward
            #15: -0.2, 16: -0.2,
        }
        
        default_depth = 0.0
        depth = depth_rules.get(point_idx, default_depth)
        
        # Skaliere mit Körpergröße
        return depth * body_size
    
    def _estimate_face_point_depth(self, point_idx: int, keypoints: np.ndarray, body_size: float) -> float:
        """Errate Tiefe für Gesichtspunkte"""
        # Gesicht ist normalerweise vor dem Körper
        return 0.1 * body_size
    
    def _estimate_hand_point_depth(self, point_idx: int, keypoints: np.ndarray, body_size: float) -> float:
        """Errate Tiefe für Handpunkte"""
        # Hände sind normalerweise vor dem Körper
        return 0.2 * body_size
    
    def _apply_anatomical_constraints(self, keypoints_3d: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Apply anatomical constraints for realistic 3D poses"""
        # Symmetrie-Konstraint für Schultern
        if scores[5] > 0.3 and scores[6] > 0.3:
            avg_z = (keypoints_3d[5, 2] + keypoints_3d[6, 2]) / 2
            keypoints_3d[5, 2] = avg_z
            keypoints_3d[6, 2] = avg_z
        
        # Glätte verbundene Körperteile
        connections = [
            [5, 7, 9],    # Linker Arm
            [6, 8, 10],   # Rechter Arm
            #[11, 13, 15], # Linkes Bein
            #[12, 14, 16]  # Rechtes Bein
        ]
        
        for connection in connections:
            valid_points = [i for i in connection if scores[i] > 0.3]
            if len(valid_points) > 1:
                avg_z = np.mean([keypoints_3d[i, 2] for i in valid_points])
                for i in valid_points:
                    keypoints_3d[i, 2] = avg_z
        
        return keypoints_3d
    
    def _calculate_3d_bboxes(self, keypoints_3d: np.ndarray, scores_3d: np.ndarray) -> np.ndarray:
        """Calculate 3D bounding boxes"""
        bboxes_3d = []
        
        for i in range(len(keypoints_3d)):
            kpts = keypoints_3d[i]
            valid_mask = scores_3d[i] > 0.3
            valid_kpts = kpts[valid_mask]
            
            if len(valid_kpts) > 0:
                min_coords = np.min(valid_kpts, axis=0)
                max_coords = np.max(valid_kpts, axis=0)
                center = (min_coords + max_coords) / 2
                dimensions = max_coords - min_coords
                confidence = np.mean(scores_3d[i][valid_mask])
                
                bbox = np.concatenate([center, dimensions, [confidence]])
                bboxes_3d.append(bbox)
            else:
                bboxes_3d.append(np.zeros(7))
        
        return np.array(bboxes_3d)
    
    def convert_2d_json_to_3d(
        self,
        input_json_path: Union[str, Path],
        output_json_path: Union[str, Path],
        image_size: Tuple[int, int] = None
    ) -> List[Dict]:
        """
        Convert a 2D pose JSON file to 3D poses.
        
        Args:
            input_json_path: Path to 2D pose JSON from PoseEstimator2D
            output_json_path: Path to save 3D pose JSON
            image_size: (width, height) of original images
            
        Returns:
            List of 3D pose results
        """
        input_json_path = Path(input_json_path)
        if not input_json_path.exists():
            raise FileNotFoundError(f"2D pose JSON not found: {input_json_path}")
        
        with open(input_json_path, 'r') as f:
            data_2d = json.load(f)
        
        results_3d = []
        
        for frame_data in data_2d:
            frame_idx = frame_data['frame']
            
            # Process left and right views separately
            left_3d = self._convert_single_view(frame_data['left'], image_size, frame_idx, 'left')
            right_3d = self._convert_single_view(frame_data['right'], image_size, frame_idx, 'right')
            
            # Combine results
            frame_result_3d = {
                "frame": frame_idx,
                "left_3d": left_3d,
                "right_3d": right_3d,
                "combined_3d": self._combine_stereo_views(left_3d, right_3d)
            }
            results_3d.append(frame_result_3d)
        
        # Save results
        with open(output_json_path, 'w') as f:
            json.dump(results_3d, f, indent=2)
        
        print(f"Converted 2D poses to 3D: {output_json_path}")
        return results_3d
    
    def _convert_single_view(self, view_data: Dict, image_size: Tuple[int, int], frame_idx: int, view: str) -> Dict:
        """Convert single view 2D data to 3D"""
        keypoints_2d = np.array(view_data['keypoints'])
        scores_2d = np.array(view_data['scores'])
        
        pose_3d = self.convert_2d_to_3d(keypoints_2d, scores_2d, image_size)
        
        return {
            "keypoints_3d": pose_3d.keypoints_3d.tolist(),
            "scores_3d": pose_3d.scores_3d.tolist(),
            "bboxes_3d": pose_3d.bboxes_3d.tolist(),
            "num_persons": pose_3d.num_persons,
            "method": pose_3d.method,
            "confidence": pose_3d.confidence
        }
    
    def _combine_stereo_views(self, left_3d: Dict, right_3d: Dict) -> Dict:
        """Combine left and right 3D views for better accuracy"""
        # Einfache Kombination: Nimm die bessere Schätzung
        if left_3d['confidence'] >= right_3d['confidence']:
            return left_3d
        else:
            return right_3d



# Convenience function
def convert_2d_poses_to_3d(
    input_json_path: Union[str, Path],
    output_json_path: Union[str, Path],
    lifting_method: str = 'geometric'  # Geändert zu geometric als Default
) -> List[Dict]:
    """
    Quick function to convert 2D pose JSON to 3D.
    
    Args:
        input_json_path: Path to 2D pose JSON from PoseEstimator2D
        output_json_path: Path to save 3D pose JSON
        lifting_method: Method for 2D-to-3D conversion
        
    Returns:
        List of 3D pose results
    """
    converter = Pose3DConverter(lifting_method=lifting_method)
    return converter.convert_2d_json_to_3d(input_json_path, output_json_path)