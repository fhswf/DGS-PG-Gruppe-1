"""
🔄 Pose3DConverter: 2D → 3D Körperpositionen konvertieren MIT EIGENEM TRAINIERTEM MODELL

EINFACHE ERKLÄRUNG:
Dieses Programm nimmt 2D-Bilder mit erkannten Körperpunkten 
und macht daraus 3D-Modelle mit einem eigenen trainierten KI-Modell!

🖼️ 2D-Foto → 🤖 KI-Modell → 🎯 3D-Figur
"""

# ===============================================
# 📦 IMPORTIEREN DER BENÖTIGTEN BIBLIOTHEKEN
# ===============================================
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict
from dataclasses import dataclass
import json
import warnings
import torch
import torch.nn as nn

# ===============================================
# 🤖 EIGENES TRAINIERTES MODELL (gleiche Struktur wie im Training)
# ===============================================
class Simple3DPoseEstimator(nn.Module):
    """Einfaches AI-Modell für 2D→3D Pose Estimation"""
    def __init__(self):
        super(Simple3DPoseEstimator, self).__init__()
        self.upscale = nn.Linear(133*2, 1024)
        self.fc1 = nn.Linear(1024, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.outputlayer = nn.Linear(1024, 133*3)

    def forward(self, x):
        x = self.upscale(x)
        x1 = nn.Dropout(p=0.5)(nn.ReLU()(self.bn1(self.fc1(x))))
        x1 = nn.Dropout(p=0.5)(nn.ReLU()(self.bn2(self.fc2(x1))))
        x = x + x1
        x1 = nn.Dropout(p=0.5)(nn.ReLU()(self.bn3(self.fc3(x))))
        x1 = nn.Dropout(p=0.5)(nn.ReLU()(self.bn4(self.fc4(x1))))
        x = x + x1
        x = self.outputlayer(x)
        return x

# ===============================================
# ⚙️ KONFIGURATION: WELCHE KÖRPERTEILE WEGLASSEN?
# ===============================================
DEFAULT_IGNORE_KEYPOINTS = list(range(13, 23))

# ===============================================
# 📦 DATENKLASSE: 3D-ERGEBNISSE SPEICHERN
# ===============================================
@dataclass
class Pose3DResult:
    frame_idx: int
    keypoints_3d: np.ndarray
    keypoints_2d: np.ndarray
    scores_3d: np.ndarray
    bboxes_3d: np.ndarray
    num_persons: int
    method: str
    confidence: float

# ===============================================
# 🔄 HAUPTKLASSE: DER 3D-KONVERTER MIT EIGENEM MODELL
# ===============================================
class Pose3DConverter:
    """
    🪄 DIE HAUPTKLASSE FÜR 2D→3D KONVERTIERUNG
    
    Nutzt jetzt das eigene trainierte Modell für präzisere 3D-Schätzungen!
    """
    
    def __init__(
        self,
        model_path: str = 'lifting2DTo3D.pth',  # 📁 Pfad zum trainierten Modell
        lifting_method: str = 'ai',  # 🔧 Methode: 'ai' (eigenes Modell) oder 'geometric' (Fallback)
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',  # 💻 Hardware
        ignore_keypoints: Optional[List[int]] = None,
        image_width: int = 1920,
        image_height: int = 1080
    ):
        """
        🏗️ KONSTRUKTOR: INITIALISIERT DEN 3D-KONVERTER MIT EIGENEM MODELL
        """
        self.model_path = model_path
        self.lifting_method = lifting_method
        self.device = device
        self.ignore_keypoints = ignore_keypoints if ignore_keypoints is not None else DEFAULT_IGNORE_KEYPOINTS
        self.image_width = image_width
        self.image_height = image_height
        
        # 🤖 Eigenes Modell laden
        self.model = None
        self._load_trained_model()
        
        print(f"✅ Pose3DConverter mit eigenem Modell bereit!")
        print(f"   Methode: {self.lifting_method}")
        print(f"   Modell: {model_path}")
        print(f"   Device: {device}")
        print(f"   Ignoriere Punkte: {self.ignore_keypoints}")
    
    def _load_trained_model(self):
        """Lädt das trainierte Modell von der Festplatte"""
        try:
            self.model = Simple3DPoseEstimator()
            
            if Path(self.model_path).exists():
                # 🚀 Modell auf korrektes Device laden
                map_location = torch.device(self.device)
                state_dict = torch.load(self.model_path, map_location=map_location)
                
                # 🔧 BatchNorm-Parameter anpassen falls nötig
                state_dict = self._fix_batchnorm_keys(state_dict)
                
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()  # ⚡ Evaluation-Modus (kein Dropout)
                
                print(f"✅ Eigenes Modell erfolgreich geladen von: {self.model_path}")
            else:
                print(f"⚠️  Modell-Datei nicht gefunden: {self.model_path}")
                print("   Verwende geometrische Methode als Fallback")
                self.lifting_method = 'geometric'
                
        except Exception as e:
            print(f"❌ Fehler beim Laden des Modells: {e}")
            print("   Verwende geometrische Methode als Fallback")
            self.lifting_method = 'geometric'
    
    def _fix_batchnorm_keys(self, state_dict):
        """Korrigiert BatchNorm-Schlüssel falls nötig"""
        new_state_dict = {}
        for key, value in state_dict.items():
            # Entferne 'module.' Präfix falls vorhanden (von DataParallel)
            if key.startswith('module.'):
                new_key = key[7:]
            else:
                new_key = key
            new_state_dict[new_key] = value
        return new_state_dict
    
    def _copy_keypoints_from_2d(self, keypoints_2d: np.ndarray, scores_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        📋 KOPIERT 2D-PUNKTE FÜR 3D - OHNE ÄNDERUNGEN!
        """
        return keypoints_2d.copy(), scores_2d.copy()
    
    def convert_2d_to_3d(
        self,
        keypoints_2d: np.ndarray,
        scores_2d: np.ndarray,
        image_size: Tuple[int, int] = None
    ) -> Pose3DResult:
        """
        🪄 HAUPT-FUNKTION: WANDELT 2D IN 3D UM MIT EIGENEM MODELL
        """
        if len(keypoints_2d) == 0:
            return self._empty_result()
        
        w, h = image_size if image_size else (self.image_width, self.image_height)
        
        # 📋 1. 2D-PUNKTE KOPIEREN
        kpts_2d, scores = self._copy_keypoints_from_2d(keypoints_2d, scores_2d)
        
        # 🔄 2. 2D → 3D KONVERTIEREN (MIT EIGENEM MODELL)
        keypoints_3d_list = []
        scores_3d_list = []
        
        for person_idx in range(len(kpts_2d)):
            kpts = kpts_2d[person_idx]
            scr = scores[person_idx]
            
            if self.lifting_method == 'ai' and self.model is not None:
                # 🤖 MIT EIGENEM MODELL
                kpts_3d = self._ai_lift_2d_to_3d(kpts, scr, (h, w))
            else:
                # 🪄 MIT GEOMETRISCHER METHODE (FALLBACK)
                kpts_3d = self._geometric_lift_2d_to_3d(kpts, scr, (h, w))
            
            keypoints_3d_list.append(kpts_3d)
            scores_3d_list.append(scr)
        
        keypoints_3d = np.array(keypoints_3d_list)
        scores_3d = np.array(scores_3d_list)
        
        # 🚫 3. PUNKTE FILTERN
        keypoints_3d, scores_3d = self._filter_keypoints(keypoints_3d, scores_3d)
        
        # 📦 4. 3D-BEGRENZUNGSRAHMEN
        bboxes_3d = self._calculate_3d_bboxes(keypoints_3d, scores_3d)
        
        # 📊 5. GESAMT-GENAUIGKEIT
        if np.any(scores_3d > 0):
            confidence = float(np.mean(scores_3d[scores_3d > 0]))
        else:
            confidence = 0.0
        
        return Pose3DResult(
            frame_idx=0,
            keypoints_3d=keypoints_3d,
            keypoints_2d=kpts_2d,
            scores_3d=scores_3d,
            bboxes_3d=bboxes_3d,
            num_persons=len(keypoints_2d),
            method=self.lifting_method,
            confidence=confidence
        )
    
    def _ai_lift_2d_to_3d(
        self, 
        keypoints_2d: np.ndarray, 
        scores: np.ndarray, 
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        🤖 EIGENES MODELL: WANDELT 2D IN 3D UM
        
        Verwendet das trainierte AI-Modell für präzisere 3D-Schätzungen.
        """
        h, w = image_shape
        keypoints_3d = np.zeros((133, 3))
        
        # 📋 1. X und Y KOORDINATEN KOPIEREN
        keypoints_3d[:, :2] = keypoints_2d
        
        # 🎯 2. NUR PUNKTE MIT AUSREICHENDER GENAUIGKEIT VERWENDEN
        valid_indices = np.where(scores > 0.3)[0]
        
        if len(valid_indices) > 0:
            # 🔢 Vorbereiten der Eingabedaten für das Modell
            input_2d = keypoints_2d.flatten()  # [266] Vektor
            
            # 📏 Normalisierung (wichtig für Modell-Performance)
            input_2d_normalized = self._normalize_keypoints(input_2d, w, h)
            
            # 🤖 Modell-Inferenz
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input_2d_normalized).unsqueeze(0).to(self.device)
                output_tensor = self.model(input_tensor)
                output_3d = output_tensor.cpu().numpy().flatten()
            
            # 🔄 Normalisierung rückgängig machen
            output_3d_reshaped = output_3d.reshape(133, 3)
            output_3d_denormalized = self._denormalize_keypoints(output_3d_reshaped, w, h)
            
            # 🎯 Nur gültige Punkte ersetzen
            for idx in valid_indices:
                # Verwende Z-Wert vom Modell, behalte originale X,Y bei
                keypoints_3d[idx, 2] = output_3d_denormalized[idx, 2]
                
                # Optional: X,Y vom Modell verwenden (falls gewünscht)
                # keypoints_3d[idx, :] = output_3d_denormalized[idx, :]
        
        # 🔧 3. FÜR NICHT-VALIDE PUNKTE: GEOMETRISCHE SCHÄTZUNG
        invalid_indices = np.where(scores <= 0.3)[0]
        for idx in invalid_indices:
            keypoints_3d[idx, 2] = self._estimate_z_by_type(idx)
        
        return keypoints_3d
    
    def _normalize_keypoints(self, keypoints: np.ndarray, w: int, h: int) -> np.ndarray:
        """Normalisiert Keypoints für das Modell"""
        # Normalisiere auf [-1, 1] Bereich
        normalized = keypoints.copy()
        for i in range(0, len(keypoints), 2):
            if i < len(keypoints):
                normalized[i] = (keypoints[i] / w) * 2 - 1  # X
            if i + 1 < len(keypoints):
                normalized[i + 1] = (keypoints[i + 1] / h) * 2 - 1  # Y
        return normalized
    
    def _denormalize_keypoints(self, keypoints_3d: np.ndarray, w: int, h: int) -> np.ndarray:
        """Macht die Normalisierung rückgängig"""
        denormalized = keypoints_3d.copy()
        for i in range(133):
            # X und Y denormalisieren (Z bleibt gleich)
            denormalized[i, 0] = (keypoints_3d[i, 0] + 1) / 2 * w
            denormalized[i, 1] = (keypoints_3d[i, 1] + 1) / 2 * h
        return denormalized
    
    def _estimate_z_by_type(self, point_idx: int) -> float:
        """Schätzt Z-Wert basierend auf Punkt-Typ (für nicht-valide Punkte)"""
        if point_idx == 0:  # 👃 Nase
            return 0.1
        elif 1 <= point_idx <= 4:  # 👀 Augen, 👂 Ohren
            return 0.1
        elif 5 <= point_idx <= 12:  # 💪 Schultern, Ellbogen, 🍑 Hüften
            return 0.0
        elif point_idx in [9, 10]:  # ✋ Handgelenke
            return 0.2
        elif 91 <= point_idx <= 111:  # ✋ Linke Hand
            return 0.25
        elif 112 <= point_idx <= 132:  # ✋ Rechte Hand
            return 0.25
        elif 23 <= point_idx <= 90:  # 😀 Gesicht
            return 0.1
        else:  # 🦵 Andere Punkte
            return 0.0
    
    def _geometric_lift_2d_to_3d(
        self, 
        keypoints_2d: np.ndarray, 
        scores: np.ndarray, 
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        🪄 GEOMETRISCHE 2D→3D KONVERTIERUNG (FALLBACK)
        """
        h, w = image_shape
        keypoints_3d = np.zeros((133, 3))
        keypoints_3d[:, :2] = keypoints_2d
        
        for i in range(133):
            if scores[i] > 0.3:
                keypoints_3d[i, 2] = self._estimate_z_by_type(i)
        
        return keypoints_3d
    
    def _filter_keypoints(
        self, 
        keypoints: np.ndarray, 
        scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filtert ignorierte Körperpunkte"""
        kpts = keypoints.copy()
        scrs = scores.copy()
        
        for idx in self.ignore_keypoints:
            if idx < kpts.shape[1]:
                kpts[:, idx, :] = 0
                scrs[:, idx] = 0
        
        return kpts, scrs
    
    def _calculate_3d_bboxes(
        self, 
        keypoints_3d: np.ndarray, 
        scores_3d: np.ndarray
    ) -> np.ndarray:
        """Berechnet 3D-Begrenzungsrahmen"""
        bboxes = []
        
        for i in range(len(keypoints_3d)):
            valid_mask = scores_3d[i] > 0.3
            valid_kpts = keypoints_3d[i][valid_mask]
            
            if len(valid_kpts) > 0:
                min_coords = np.min(valid_kpts, axis=0)
                max_coords = np.max(valid_kpts, axis=0)
                center = (min_coords + max_coords) / 2
                dimensions = max_coords - min_coords
                confidence = np.mean(scores_3d[i][valid_mask])
                bboxes.append(np.concatenate([center, dimensions, [confidence]]))
            else:
                bboxes.append(np.zeros(7))
        
        return np.array(bboxes)
    
    def _empty_result(self) -> Pose3DResult:
        """Gibt ein leeres Ergebnis zurück"""
        return Pose3DResult(
            frame_idx=0,
            keypoints_3d=np.empty((0, 133, 3)),
            keypoints_2d=np.empty((0, 133, 2)),
            scores_3d=np.empty((0, 133)),
            bboxes_3d=np.empty((0, 7)),
            num_persons=0,
            method=self.lifting_method,
            confidence=0.0
        )
    
    def convert_2d_json_to_3d(
        self,
        input_json_path: Union[str, Path],
        output_json_path: Union[str, Path],
        image_size: Tuple[int, int] = None
    ) -> List[Dict]:
        """Konvertiert eine ganze 2D-JSON-Datei zu 3D"""
        input_path = Path(input_json_path)
        if not input_path.exists():
            raise FileNotFoundError(f"❌ 2D JSON nicht gefunden: {input_path}")
        
        with open(input_path, 'r') as f:
            data_2d = json.load(f)
        
        results_3d = []
        
        print(f"🔄 Konvertiere {len(data_2d)} Bilder von 2D zu 3D mit {self.lifting_method}...")
        
        for frame_data in data_2d:
            frame_idx = frame_data['frame']
            
            left_3d = self._convert_single_view(frame_data['left'], image_size, frame_idx)
            right_3d = self._convert_single_view(frame_data['right'], image_size, frame_idx)
            
            frame_result = {
                "frame": frame_idx,
                "left_3d": left_3d,
                "right_3d": right_3d,
                "combined_3d": left_3d
            }
            results_3d.append(frame_result)
            
            if frame_idx % 10 == 0:
                print(f"  📊 Bild {frame_idx}/{len(data_2d)} konvertiert")
        
        with open(output_json_path, 'w') as f:
            json.dump(results_3d, f, indent=2)
        
        print(f"✅ 3D Posen gespeichert: {output_json_path}")
        return results_3d
    
    def _convert_single_view(
        self, 
        view_data: Dict, 
        image_size: Tuple[int, int], 
        frame_idx: int
    ) -> Dict:
        """Konvertiert eine einzelne Kamera-Ansicht"""
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

# ===============================================
# ⚡ BEQUEMLICHKEITSFUNKTION (Für Import)
# ===============================================

def convert_2d_poses_to_3d(
    input_json_path: Union[str, Path],
    output_json_path: Union[str, Path],
    model_path: str = 'lifting2DTo3D.pth',  # 📁 Standard-Pfad zum trainierten Modell
    lifting_method: str = 'ai',  # 🔧 'ai' für eigenes Modell, 'geometric' für Fallback
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> List[Dict]:
    """
    ⚡ SCHNELLE FUNKTION FÜR 2D→3D KONVERTIERUNG MIT EIGENEM MODELL
    
    Einfachste Nutzung:
        convert_2d_poses_to_3d("2d_poses.json", "3d_poses.json")
    """
    converter = Pose3DConverter(
        model_path=model_path,
        lifting_method=lifting_method,
        device=device
    )
    return converter.convert_2d_json_to_3d(input_json_path, output_json_path)

# ===============================================
# 🚀 START: WENN DAS PROGRAMM DIREKT GESTARTET WIRD
# ===============================================
if __name__ == "__main__":
    print("=" * 60)
    print("🔄 Pose3DConverter mit eigenem trainiertem Modell")
    print("=" * 60)
    print("🤖 Verwendet das trainierte AI-Modell für 2D→3D")
    print("")
    
    # 🔍 Test-Dateien
    test_input = "poses_2d_filtered.json"
    test_output = "poses_3d_ai.json"
    model_path = "lifting2DTo3D.pth"  # Ihr trainiertes Modell
    
    if Path(test_input).exists():
        print(f"✅ Testdatei gefunden: {test_input}")
        
        if Path(model_path).exists():
            print(f"✅ Modell gefunden: {model_path}")
            print("🔄 Starte Konvertierung mit eigenem Modell...")
            
            results = convert_2d_poses_to_3d(test_input, test_output, model_path)
            
            print(f"✅ Erfolgreich konvertiert mit eigenem Modell!")
            print(f"   📊 {len(results)} Bilder verarbeitet")
            print(f"   📁 Ergebnis: {test_output}")
        else:
            print(f"⚠️  Modell-Datei {model_path} nicht gefunden")
            print("🔄 Starte Konvertierung mit geometrischer Methode (Fallback)...")
            
            results = convert_2d_poses_to_3d(
                test_input, 
                test_output, 
                model_path, 
                lifting_method='geometric'
            )
            
            print(f"✅ Mit geometrischer Methode konvertiert!")
            print(f"   📊 {len(results)} Bilder verarbeitet")
            print(f"   📁 Ergebnis: {test_output}")
    else:
        print(f"⚠️  Testdatei {test_input} nicht gefunden")
        print("")
        print("ℹ️  So nutzt du es:")
        print("   1. Trainiere zuerst das Modell mit Ihrem Training-Skript")
        print("   2. Erstelle 2D-Posen mit PoseEstimator2D")
        print("   3. Konvertiere zu 3D:")
        print("      convert_2d_poses_to_3d('poses_2d.json', 'poses_3d.json', 'net_final.pth')")