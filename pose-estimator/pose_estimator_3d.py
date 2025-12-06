"""
🔄 Pose3DConverter: VERBESSERT mit H3WB Dataset Integration

ÄNDERUNGEN:
1. ✅ Methoden-Namen korrigiert ('ai' → 'mlp')
2. ✅ H3WB Dataset Loader integriert
3. ✅ Train/Test Split implementiert
4. ✅ Normalisierung verbessert

WAS DIESE DATEI MACHT:
- Definiert ein neuronales Netz für 2D→3D Pose Estimation
- Lädt und verarbeitet H3WB (Holistic 3D Whole-Body) Datasets
- Bietet Training-Funktionen für einzelne und multiple Dateien
- Stellt eine Inference-Klasse für die Konvertierung bereit
"""

import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict
from dataclasses import dataclass
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# ===============================================
# 🤖 MODELL-DEFINITION - DAS NEURONALE NETZ
# ===============================================
class Simple3DPoseEstimator(nn.Module):
    """
    🤖 MLP-Modell für 2D→3D Pose Estimation
    
    Architektur:
    - Eingang: 266 Werte (133 Keypoints × 2 Koordinaten)
    - Ausgang: 399 Werte (133 Keypoints × 3 Koordinaten)
    - Residual Connections: Vermeiden das Vanishing Gradient Problem
    - Batch Normalization: Stabilisiert das Training
    - Dropout: Verhindert Overfitting (50% Dropout Rate)
    """
    def __init__(self):
        super(Simple3DPoseEstimator, self).__init__()
        # Initialer Upscale von 266 auf 1024 Features
        self.upscale = nn.Linear(133*2, 1024)
        
        # Erster Residual Block
        self.fc1 = nn.Linear(1024, 1024)
        self.bn1 = nn.BatchNorm1d(1024)  # Normalisiert die Aktivierungen
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        
        # Zweiter Residual Block
        self.fc3 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        
        # Ausgangsschicht zu 3D-Koordinaten
        self.outputlayer = nn.Linear(1024, 133*3)

    def forward(self, x):
        """
        Forward-Pass des neuronalen Netzes
        x: Eingabe-Tensor der Form [Batch, 266]
        Rückgabe: 3D-Posen der Form [Batch, 399]
        """
        # Initialer Upscale
        x = self.upscale(x)
        
        # Erster Residual Block mit Skip Connection
        x1 = nn.Dropout(p=0.5)(nn.ReLU()(self.bn1(self.fc1(x))))
        x1 = nn.Dropout(p=0.5)(nn.ReLU()(self.bn2(self.fc2(x1))))
        x = x + x1  # Skip Connection: Ursprünglicher Input wird hinzugefügt
        
        # Zweiter Residual Block mit Skip Connection
        x1 = nn.Dropout(p=0.5)(nn.ReLU()(self.bn3(self.fc3(x))))
        x1 = nn.Dropout(p=0.5)(nn.ReLU()(self.bn4(self.fc4(x1))))
        x = x + x1  # Skip Connection
        
        # Finale Ausgabe zu 3D-Koordinaten
        x = self.outputlayer(x)
        return x

# ===============================================
# 📦 H3WB DATASET KLASSE - DATENLADER
# ===============================================
class H3WBDataset(Dataset):
    """
    🎯 PyTorch Dataset Klasse für H3WB (Holistic 3D Whole-Body Dataset)
    
    ERWARTETES JSON-FORMAT:
    {
        "frame_0": {
            "keypoints_2d": {"0": {"x": 123, "y": 456}, ...},
            "keypoints_3d": {"0": {"x": 1.23, "y": 4.56, "z": 0.12}, ...}
        },
        "frame_1": {...},
        ...
    }
    
    KEYPOINTS-BEZUG:
    - 0-4: Gesicht (5 Punkte)
    - 5-22: Körper (18 Punkte)
    - 23-90: Hände (je 21 Punkte × 2 = 68 Punkte)
    - 91-132: Füße (je 21 Punkte × 2 = 42 Punkte)
    - Total: 133 Keypoints
    """
    def __init__(self, json_path: str, normalize: bool = True):
        """
        Initialisiert den Dataset-Loader
        
        Args:
            json_path: Pfad zur JSON-Datei mit den Pose-Daten
            normalize: Wenn True, werden die Daten z-standardisiert (μ=0, σ=1)
        """
        self.normalize = normalize
        self.data = []  # Liste von Dictionarys mit kp2d und kp3d
        
        print(f"📂 Lade H3WB Dataset: {json_path}")
        with open(json_path, 'r') as f:
            data_dict = json.load(f)
        
        # Sammle Statistiken für Normalisierung
        all_2d_x, all_2d_y = [], []
        all_3d_x, all_3d_y, all_3d_z = [], [], []
        
        # Iteriere durch alle Frames im Dataset
        for frame_id, frame_data in data_dict.items():
            keypoints_2d = frame_data['keypoints_2d']
            keypoints_3d = frame_data['keypoints_3d']
            
            # 🔄 2D Keypoints extrahieren (133 Punkte × 2 Koordinaten)
            kp2d_list = []
            for i in range(133):  # Für alle 133 Keypoints
                if str(i) in keypoints_2d:
                    # Extrahiere x, y Koordinaten
                    x, y = keypoints_2d[str(i)]['x'], keypoints_2d[str(i)]['y']
                    kp2d_list.extend([x, y])  # Füge [x, y] zur Liste hinzu
                    # Sammle für Statistiken
                    all_2d_x.append(x)
                    all_2d_y.append(y)
                else:
                    # Wenn Keypoint nicht vorhanden, setze auf 0
                    kp2d_list.extend([0.0, 0.0])
            
            # 🔄 3D Keypoints extrahieren (133 Punkte × 3 Koordinaten)
            kp3d_list = []
            for i in range(133):
                if str(i) in keypoints_3d:
                    x = keypoints_3d[str(i)]['x']
                    y = keypoints_3d[str(i)]['y']
                    z = keypoints_3d[str(i)]['z']
                    kp3d_list.extend([x, y, z])  # Füge [x, y, z] zur Liste hinzu
                    # Sammle für Statistiken
                    all_3d_x.append(x)
                    all_3d_y.append(y)
                    all_3d_z.append(z)
                else:
                    kp3d_list.extend([0.0, 0.0, 0.0])
            
            # Speichere den Frame im Dataset
            self.data.append({
                'kp2d': np.array(kp2d_list, dtype=np.float32),
                'kp3d': np.array(kp3d_list, dtype=np.float32)
            })
        
        # 📊 Berechne Normalisierungs-Statistiken (Z-Standardisierung)
        if self.normalize:
            self.stats_2d = {
                'x_mean': np.mean(all_2d_x), 'x_std': np.std(all_2d_x),
                'y_mean': np.mean(all_2d_y), 'y_std': np.std(all_2d_y)
            }
            self.stats_3d = {
                'x_mean': np.mean(all_3d_x), 'x_std': np.std(all_3d_x),
                'y_mean': np.mean(all_3d_y), 'y_std': np.std(all_3d_y),
                'z_mean': np.mean(all_3d_z), 'z_std': np.std(all_3d_z)
            }
            
            # Zeige Normalisierungs-Statistiken an
            print(f"📊 Normalisierungs-Statistiken berechnet:")
            print(f"   2D X: μ={self.stats_2d['x_mean']:.2f}, σ={self.stats_2d['x_std']:.2f}")
            print(f"   2D Y: μ={self.stats_2d['y_mean']:.2f}, σ={self.stats_2d['y_std']:.2f}")
        
        print(f"✅ {len(self.data)} Samples geladen")
    
    def __len__(self):
        """Gibt die Anzahl der Samples im Dataset zurück"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Gibt ein Sample (2D-Pose, 3D-Pose) zurück
        
        Args:
            idx: Index des Samples (0 bis len(dataset)-1)
        
        Returns:
            kp2d: 2D-Pose als numpy array [266]
            kp3d: 3D-Pose als numpy array [399]
        """
        kp2d = self.data[idx]['kp2d'].copy()
        kp3d = self.data[idx]['kp3d'].copy()
        
        # 🔄 Normalisierung (Z-Standardisierung)
        if self.normalize:
            # 2D-Posen normalisieren (alle x- und y-Koordinaten)
            for i in range(0, len(kp2d), 2):
                kp2d[i] = (kp2d[i] - self.stats_2d['x_mean']) / self.stats_2d['x_std']
                kp2d[i+1] = (kp2d[i+1] - self.stats_2d['y_mean']) / self.stats_2d['y_std']
            
            # 3D-Posen normalisieren (alle x-, y-, z-Koordinaten)
            for i in range(0, len(kp3d), 3):
                kp3d[i] = (kp3d[i] - self.stats_3d['x_mean']) / self.stats_3d['x_std']
                kp3d[i+1] = (kp3d[i+1] - self.stats_3d['y_mean']) / self.stats_3d['y_std']
                kp3d[i+2] = (kp3d[i+2] - self.stats_3d['z_mean']) / self.stats_3d['z_std']
        
        return kp2d, kp3d

# ===============================================
# 🎓 TRAINING-FUNKTIONEN MIT INKREMENTELLEM LADEN
# ===============================================
def train_on_h3wb_incremental(
    train_json_files: List[str],
    test_json: Optional[str] = None,
    epochs: int = 75,
    batch_size: int = 256,
    learning_rate: float = 0.002,
    output_model: str = 'lifting2DTo3D.pth',
    train_split: float = 0.8,
    checkpoint_interval: int = 10
):
    """
    🎓 TRAINIERT DAS MODELL MIT MEHREREN DATASET-TEILEN (INKREMENTELL)
    
    Diese Methode:
    1. Lädt jeden Dataset-Teil einzeln (spart RAM)
    2. Trainiert auf jedem Teil für 'epochs' Epochen
    3. Führt Evaluation auf Test-Set durch
    4. Speichert regelmäßig Checkpoints
    
    Args:
        train_json_files: Liste von JSON-Dateien (werden nacheinander geladen)
        test_json: Optional separates Test-Set (wenn None, wird gesplittet)
        epochs: Epochen pro Dataset-Teil
        batch_size: Anzahl Samples pro Optimierungsschritt
        learning_rate: Anfängliche Lernrate für den Optimierer
        output_model: Pfad, wo das trainierte Modell gespeichert wird
        train_split: Anteil für Training (0.8 = 80% Training, 20% Test)
        checkpoint_interval: Speichere Checkpoint alle N Epochen
    
    Beispiel:
        train_on_h3wb_incremental(
            train_json_files=[
                'h3wb_part1.json',
                'h3wb_part2.json', 
                'h3wb_part3.json'
            ],
            test_json='h3wb_test.json'
        )
    """
    # 🔧 Hardware-Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    print(f"📦 Anzahl Dataset-Teile: {len(train_json_files)}")
    
    # 🤖 Modell initialisieren (nur einmal!)
    model = Simple3DPoseEstimator().to(device)
    
    # 📂 Lade existierendes Modell falls vorhanden (für fortgesetztes Training)
    if Path(output_model).exists():
        print(f"📂 Lade existierendes Modell: {output_model}")
        model.load_state_dict(torch.load(output_model, map_location=device))
    
    # 📈 Verlustfunktion und Optimierer
    criterion = nn.MSELoss()  # Mean Squared Error für Regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 📉 Lernrate Scheduler: Reduziert LR wenn Loss stagniert
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # 📊 Test-Dataset laden (nur einmal, wenn separate Test-Datei)
    if test_json:
        print(f"📂 Lade Test-Dataset: {test_json}")
        test_dataset = H3WBDataset(test_json, normalize=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        test_loader = None  # Wird später pro Dataset-Teil erstellt
    
    best_loss = float('inf')  # Initialer bester Loss (unendlich hoch)
    global_epoch = 0  # Zählt alle Epochen über alle Dataset-Teile
    
    # 🔄 ITERIERE ÜBER ALLE DATASET-TEILE
    for file_idx, json_file in enumerate(train_json_files):
        print(f"\n{'='*60}")
        print(f"📦 Dataset-Teil {file_idx+1}/{len(train_json_files)}: {json_file}")
        print(f"{'='*60}")
        
        # 📂 Lade aktuellen Dataset-Teil
        full_dataset = H3WBDataset(json_file, normalize=True)
        
        # 🔀 Train/Test Split für diesen Teil (wenn kein separates Test-Set)
        if test_json is None:
            train_size = int(train_split * len(full_dataset))
            test_size = len(full_dataset) - train_size
            # Random Split mit festem Seed für Reproduzierbarkeit
            train_dataset, current_test = random_split(
                full_dataset, [train_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            test_loader = DataLoader(current_test, batch_size=batch_size, shuffle=False)
            print(f"📊 Split: {train_size} Training / {test_size} Test")
        else:
            # Wenn separates Test-Set, verwende gesamten Teil für Training
            train_dataset = full_dataset
        
        # 📥 Erstelle DataLoader für Training
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 🏃 TRAINING LOOP FÜR DIESEN DATASET-TEIL
        for epoch in range(epochs):
            global_epoch += 1
            model.train()  # Setze Modell in Trainingsmodus
            train_loss = 0
            
            # 🔄 Iteriere über alle Batches im Trainingsloader
            for inputs_2d, targets_3d in tqdm(train_loader, 
                desc=f'Teil {file_idx+1}/{len(train_json_files)} | Epoch {epoch+1}/{epochs}',
                leave=False):
                
                # 🖥️ Verschiebe Daten auf GPU falls verfügbar
                inputs_2d = inputs_2d.to(device)
                targets_3d = targets_3d.to(device)
                
                # 🧹 Setze Gradienten zurück
                optimizer.zero_grad()
                
                # 🤖 Forward Pass: Berechne Vorhersagen
                outputs = model(inputs_2d)
                
                # 📉 Berechne Verlust
                loss = criterion(outputs, targets_3d)
                
                # 📈 Backward Pass: Berechne Gradienten
                loss.backward()
                
                # 🚀 Update Modellparameter
                optimizer.step()
                
                # 📊 Akkumuliere Verlust für Statistik
                train_loss += loss.item()
            
            # 📊 Berechne durchschnittlichen Trainingsverlust
            train_loss /= len(train_loader)
            
            # 📊 VALIDATION (EVALUATION)
            if test_loader:
                model.eval()  # Setze Modell in Evaluierungsmodus
                test_loss = 0
                with torch.no_grad():  # Deaktiviere Gradientenberechnung für Evaluation
                    for inputs_2d, targets_3d in test_loader:
                        inputs_2d = inputs_2d.to(device)
                        targets_3d = targets_3d.to(device)
                        outputs = model(inputs_2d)
                        loss = criterion(outputs, targets_3d)
                        test_loss += loss.item()
                
                test_loss /= len(test_loader)
                print(f"Global Epoch {global_epoch}: Train={train_loss:.6f}, Test={test_loss:.6f}")
                
                # 📉 Passe Lernrate basierend auf Test-Loss an
                scheduler.step(test_loss)
                
                # 💾 SPEICHERE BESTES MODELL
                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(model.state_dict(), output_model)
                    print(f"✅ Neues bestes Modell gespeichert: {output_model}")
            else:
                # Nur Training, kein Test
                print(f"Global Epoch {global_epoch}: Train={train_loss:.6f}")
            
            # 💾 CHECKPOINT SPEICHERN (regelmäßige Sicherungen)
            if global_epoch % checkpoint_interval == 0:
                checkpoint_path = f'checkpoint_epoch_{global_epoch}.pth'
                torch.save({
                    'epoch': global_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss if test_loader else None,
                }, checkpoint_path)
                print(f"💾 Checkpoint gespeichert: {checkpoint_path}")
        
        # 🧹 SPEICHER FREIGEBEN nach jedem Dataset-Teil
        del train_dataset
        del train_loader
        if test_json is None:
            del test_loader
        # Leere GPU-Cache falls CUDA verfügbar
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"✅ Dataset-Teil {file_idx+1} abgeschlossen!")
    
    print(f"\n🎉 KOMPLETTES Training abgeschlossen!")
    print(f"   Gesamt-Epochen: {global_epoch}")
    print(f"   Bestes Test-Loss: {best_loss:.6f}")
    return model

def train_on_h3wb(
    train_json: str,
    test_json: Optional[str] = None,
    epochs: int = 75,
    batch_size: int = 256,
    learning_rate: float = 0.002,
    output_model: str = 'lifting2DTo3D.pth',
    train_split: float = 0.8
):
    """
    🎓 TRAINIERT DAS MODELL AUF EINZELNEM H3WB DATASET
    
    Vereinfachte Version für einzelne Dateien.
    Für inkrementelles Training mit mehreren Dateien nutze:
    train_on_h3wb_incremental()
    """
    # Nutze die inkrementelle Funktion mit nur einer Datei
    return train_on_h3wb_incremental(
        train_json_files=[train_json],
        test_json=test_json,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_model=output_model,
        train_split=train_split
    )

# ===============================================
# 📦 ERGEBNIS-DATENKLASSE - STRUKTURIERTE AUSGABE
# ===============================================
@dataclass
class Pose3DResult:
    """
    📦 Datenklasse für 3D-Pose Ergebnisse
    
    Enthält alle relevanten Informationen für eine 3D-Pose.
    """
    frame_idx: int                  # Frame-Nummer
    keypoints_3d: np.ndarray       # 3D-Koordinaten [Personen, 133, 3]
    keypoints_2d: np.ndarray       # 2D-Koordinaten [Personen, 133, 2]
    scores_3d: np.ndarray          # Konfidenz-Scores [Personen, 133]
    bboxes_3d: np.ndarray          # 3D Bounding Boxes [Personen, 7]
    num_persons: int               # Anzahl detektierter Personen
    method: str                    # Verwendete Methode ('mlp' oder 'geometric')
    confidence: float              # Durchschnitts-Konfidenz

# ===============================================
# 🔄 HAUPTKLASSE: POSE3D CONVERTER (KORRIGIERT)
# ===============================================
DEFAULT_IGNORE_KEYPOINTS = list(range(13, 23))

class Pose3DConverter:
    """
    🪄 2D→3D KONVERTER MIT TRAINIERTEM MODELL
    
    KERN-FUNKTIONALITÄT:
    1. Lädt ein trainiertes MLP-Modell
    2. Verarbeitet 2D-Posen (z.B. von OpenPose)
    3. Konvertiert diese zu 3D-Posen
    4. Bietet Fallback-Methode (geometrisch) wenn kein Modell verfügbar
    
    ✅ KORRIGIERT: lifting_method='mlp' für AI-Modell
    """
    
    def __init__(
        self,
        model_path: str = 'lifting2DTo3D.pth',
        lifting_method: str = 'mlp',  # ✅ KORRIGIERT: 'mlp' statt 'ai'
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        ignore_keypoints: Optional[List[int]] = None,
        image_width: int = 1920,
        image_height: int = 1080,
        normalization_stats: Optional[Dict] = None  # ✅ NEU: Für Normalisierung
    ):
        """
        Initialisiert den 3D-Pose Converter
        
        Args:
            model_path: Pfad zum trainierten Modell
            lifting_method: 'mlp' für neuronales Netz, 'geometric' für Fallback
            device: 'cuda' oder 'cpu'
            ignore_keypoints: Liste von Keypoint-Indizes, die ignoriert werden sollen
            image_width: Standard-Bildbreite für Normalisierung
            image_height: Standard-Bildhöhe für Normalisierung
            normalization_stats: Vorberechnete Normalisierungsstatistiken
        """
        self.model_path = model_path
        self.lifting_method = lifting_method
        self.device = device
        self.ignore_keypoints = ignore_keypoints or DEFAULT_IGNORE_KEYPOINTS
        self.image_width = image_width
        self.image_height = image_height
        self.normalization_stats = normalization_stats
        
        self.model = None
        self._load_trained_model()  # Lade das Modell sofort
        
        print(f"✅ Pose3DConverter bereit!")
        print(f"   Methode: {self.lifting_method}")
        print(f"   Modell: {model_path}")
        print(f"   Device: {device}")
    
    def _load_trained_model(self):
        """Lädt das trainierte Modell von Festplatte"""
        try:
            self.model = Simple3DPoseEstimator()
            
            if Path(self.model_path).exists():
                # Lade Modellgewichte
                state_dict = torch.load(
                    self.model_path, 
                    map_location=torch.device(self.device)
                )
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()  # Wichtig für Inference (kein Training)
                print(f"✅ Modell geladen: {self.model_path}")
            else:
                print(f"⚠️  Modell nicht gefunden: {self.model_path}")
                # Fallback auf geometrische Methode
                self.lifting_method = 'geometric'
        except Exception as e:
            print(f"❌ Fehler: {e}")
            self.lifting_method = 'geometric'  # Fallback
    
    def convert_2d_to_3d(
        self,
        keypoints_2d: np.ndarray,
        scores_2d: np.ndarray,
        image_size: Tuple[int, int] = None
    ) -> Pose3DResult:
        """🪄 Konvertiert 2D-Posen zu 3D-Posen"""
        if len(keypoints_2d) == 0:
            return self._empty_result()  # Leeres Ergebnis wenn keine Personen
        
        # Bestimme Bildgröße
        w, h = image_size if image_size else (self.image_width, self.image_height)
        
        keypoints_3d_list = []
        scores_3d_list = []
        
        # 🔄 KONVERTIERE JEDE PERSON EINZELN
        for person_idx in range(len(keypoints_2d)):
            kpts = keypoints_2d[person_idx]  # 2D Keypoints dieser Person
            scr = scores_2d[person_idx]     # Konfidenz-Scores
            
            # ✅ KORRIGIERT: Prüft jetzt auf 'mlp'
            if self.lifting_method == 'mlp' and self.model is not None:
                kpts_3d = self._mlp_lift_2d_to_3d(kpts, scr, (h, w))
            else:
                kpts_3d = self._geometric_lift_2d_to_3d(kpts, scr, (h, w))
            
            keypoints_3d_list.append(kpts_3d)
            scores_3d_list.append(scr)
        
        # Erstelle numpy Arrays für alle Personen
        keypoints_3d = np.array(keypoints_3d_list)
        scores_3d = np.array(scores_3d_list)
        
        # Filtere ignorierte Keypoints
        keypoints_3d, scores_3d = self._filter_keypoints(keypoints_3d, scores_3d)
        
        # Berechne 3D Bounding Boxes
        bboxes_3d = self._calculate_3d_bboxes(keypoints_3d, scores_3d)
        
        # Berechne Gesamt-Konfidenz
        confidence = float(np.mean(scores_3d[scores_3d > 0])) if np.any(scores_3d > 0) else 0.0
        
        return Pose3DResult(
            frame_idx=0,
            keypoints_3d=keypoints_3d,
            keypoints_2d=keypoints_2d,
            scores_3d=scores_3d,
            bboxes_3d=bboxes_3d,
            num_persons=len(keypoints_2d),
            method=self.lifting_method,
            confidence=confidence
        )
    
    def _mlp_lift_2d_to_3d(
        self, 
        keypoints_2d: np.ndarray, 
        scores: np.ndarray, 
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """🤖 Verwendet das trainierte MLP für 2D→3D Konvertierung"""
        h, w = image_shape
        
        # Input vorbereiten: Flache 266-Werte Vektor (133×2)
        input_2d = keypoints_2d.flatten()
        
        # 🔄 NORMALISIERUNG (sehr wichtig für gute Ergebnisse!)
        if self.normalization_stats:
            # Verwende gespeicherte Z-Standardisierung
            input_2d = self._normalize_input(input_2d, w, h)
        else:
            # Einfache Min-Max Normalisierung auf [0, 1]
            input_2d_norm = np.zeros_like(input_2d)
            input_2d_norm[0::2] = input_2d[0::2] / w  # X-Koordinaten
            input_2d_norm[1::2] = input_2d[1::2] / h  # Y-Koordinaten
            input_2d = input_2d_norm
        
        # 🤖 MODELL-INFERENZ
        with torch.no_grad():  # Keine Gradientenberechnung für Inference
            # Konvertiere zu PyTorch Tensor
            input_tensor = torch.FloatTensor(input_2d).unsqueeze(0).to(self.device)
            # Forward-Pass durch das Modell
            output_tensor = self.model(input_tensor)
            # Zurück zu numpy
            output_3d = output_tensor.cpu().numpy().flatten()
        
        # Reshape zu (133, 3) Matrix
        keypoints_3d = output_3d.reshape(133, 3)
        
        # 🔄 DENORMALISIERUNG (falls nötig)
        if self.normalization_stats:
            keypoints_3d = self._denormalize_output(keypoints_3d, w, h)
        
        return keypoints_3d
    
    def _normalize_input(self, input_2d: np.ndarray, w: int, h: int) -> np.ndarray:
        """Normalisiert Input mit gespeicherten Statistiken (Z-Standardisierung)"""
        if not self.normalization_stats:
            return input_2d
        
        normalized = input_2d.copy()
        stats = self.normalization_stats['2d']
        
        # Normalisiere jede Koordinate: (x - μ) / σ
        for i in range(0, len(input_2d), 2):
            normalized[i] = (input_2d[i] - stats['x_mean']) / stats['x_std']
            normalized[i+1] = (input_2d[i+1] - stats['y_mean']) / stats['y_std']
        
        return normalized
    
    def _denormalize_output(self, output_3d: np.ndarray, w: int, h: int) -> np.ndarray:
        """Denormalisiert Output mit gespeicherten Statistiken"""
        if not self.normalization_stats:
            return output_3d
        
        denormalized = output_3d.copy()
        stats = self.normalization_stats['3d']
        
        # Umkehrung der Z-Standardisierung: x * σ + μ
        denormalized[:, 0] = output_3d[:, 0] * stats['x_std'] + stats['x_mean']
        denormalized[:, 1] = output_3d[:, 1] * stats['y_std'] + stats['y_mean']
        denormalized[:, 2] = output_3d[:, 2] * stats['z_std'] + stats['z_mean']
        
        return denormalized
    
    def _geometric_lift_2d_to_3d(
        self, 
        keypoints_2d: np.ndarray, 
        scores: np.ndarray, 
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """🪄 Fallback: Geometrische Methode (einfache Heuristik)"""
        keypoints_3d = np.zeros((133, 3))
        
        # Kopiere 2D-Koordinaten (X, Y bleiben gleich)
        keypoints_3d[:, :2] = keypoints_2d
        
        # Einfache Z-Schätzung basierend auf Körperteil
        for i in range(133):
            if scores[i] > 0.3:  # Nur wenn Score hoch genug
                keypoints_3d[i, 2] = self._estimate_z_by_type(i)
        
        return keypoints_3d
    
    def _estimate_z_by_type(self, point_idx: int) -> float:
        """Schätzt Z-Wert nach Körperteil (einfache Heuristik)"""
        if point_idx == 0:
            return 0.1    # Nase
        elif 1 <= point_idx <= 4:
            return 0.1    # Augen, Ohren
        elif 5 <= point_idx <= 12:
            return 0.0    # Schultern, Ellbogen, Handgelenke
        elif point_idx in [9, 10]:
            return 0.2    # Handgelenke weiter vorne
        elif 91 <= point_idx <= 132:
            return 0.25   # Füße (weiter vorne im 3D-Raum)
        elif 23 <= point_idx <= 90:
            return 0.1    # Hände
        else:
            return 0.0
    
    def _filter_keypoints(self, keypoints, scores):
        """Filtert ignorierte Keypoints (setzt sie auf 0)"""
        kpts = keypoints.copy()
        scrs = scores.copy()
        
        for idx in self.ignore_keypoints:
            if idx < kpts.shape[1]:
                kpts[:, idx, :] = 0
                scrs[:, idx] = 0
        
        return kpts, scrs
    
    def _calculate_3d_bboxes(self, keypoints_3d, scores_3d):
        """Berechnet 3D Bounding Boxes für jede Person"""
        bboxes = []
        
        for i in range(len(keypoints_3d)):
            # Nur Keypoints mit hohem Score berücksichtigen
            valid_mask = scores_3d[i] > 0.3
            valid_kpts = keypoints_3d[i][valid_mask]
            
            if len(valid_kpts) > 0:
                # Min- und Max-Koordinaten
                min_coords = np.min(valid_kpts, axis=0)
                max_coords = np.max(valid_kpts, axis=0)
                
                # Bounding Box: Center + Dimensions + Confidence
                center = (min_coords + max_coords) / 2
                dimensions = max_coords - min_coords
                confidence = np.mean(scores_3d[i][valid_mask])
                
                bboxes.append(np.concatenate([center, dimensions, [confidence]]))
            else:
                # Keine validen Keypoints → leere BBox
                bboxes.append(np.zeros(7))
        
        return np.array(bboxes)
    
    def convert_2d_json_to_3d(
        self,
        input_json_path: Union[str, Path],
        output_json_path: Union[str, Path],
        image_size: Tuple[int, int] = None
    ) -> List[Dict]:
        """
        🔄 Konvertiert eine komplette 2D-JSON-Datei zu 3D
        
        Format der Eingabe-JSON:
        [
            {
                "frame": 0,
                "left": {"keypoints": [...], "scores": [...]},
                "right": {"keypoints": [...], "scores": [...]}
            }
        ]
        """
        input_path = Path(input_json_path)
        if not input_path.exists():
            raise FileNotFoundError(f"❌ 2D JSON nicht gefunden: {input_path}")
        
        # Lade 2D-Daten
        with open(input_path, 'r') as f:
            data_2d = json.load(f)
        
        results_3d = []
        
        print(f"🔄 Konvertiere {len(data_2d)} Frames von 2D zu 3D mit Methode '{self.lifting_method}'...")
        
        # 🔄 KONVERTIERE JEDEN FRAME
        for frame_data in data_2d:
            frame_idx = frame_data['frame']
            
            # Konvertiere beide Kamera-Ansichten
            left_3d = self._convert_single_view(
                frame_data['left'], 
                image_size, 
                frame_idx
            )
            right_3d = self._convert_single_view(
                frame_data['right'], 
                image_size, 
                frame_idx
            )
            
            # Erstelle Ergebnis-Struktur
            frame_result = {
                "frame": frame_idx,
                "left_3d": left_3d,
                "right_3d": right_3d,
                "combined_3d": left_3d  # Nutze left als combined
            }
            results_3d.append(frame_result)
            
            # Fortschritt anzeigen
            if frame_idx % 10 == 0:
                print(f"  📊 Frame {frame_idx}/{len(data_2d)} konvertiert")
        
        # 💾 Speichere Ergebnisse
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
        """Konvertiert eine einzelne Kamera-Ansicht von 2D zu 3D"""
        keypoints_2d = np.array(view_data['keypoints'])
        scores_2d = np.array(view_data['scores'])
        
        # Nutze die Haupt-Konvertierungs-Funktion
        pose_3d = self.convert_2d_to_3d(keypoints_2d, scores_2d, image_size)
        
        # Erstelle strukturierte Ausgabe
        return {
            "keypoints_3d": pose_3d.keypoints_3d.tolist(),
            "scores_3d": pose_3d.scores_3d.tolist(),
            "bboxes_3d": pose_3d.bboxes_3d.tolist(),
            "num_persons": pose_3d.num_persons,
            "method": pose_3d.method,
            "confidence": pose_3d.confidence
        }
    
    def _empty_result(self):
        """Gibt leeres Ergebnis zurück (wenn keine Personen)"""
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

# ===============================================
# ⚡ CONVENIENCE FUNCTIONS - EINFACHE SCHNITTSTELLEN
# ===============================================

def train_new_model(
    h3wb_json: Union[str, List[str]],
    output_model: str = 'lifting2DTo3D.pth',
    epochs: int = 75,
    test_json: Optional[str] = None
):
    """
    🎓 Trainiert neues Modell auf H3WB Dataset
    
    Intelligente Funktion, die automatisch entscheidet:
    - Einzelne Datei → Standard Training
    - Mehrere Dateien → Inkrementelles Training
    
    Beispiele:
        # Einzelne Datei:
        train_new_model('h3wb_full.json')
        
        # Mehrere Teile (inkrementell):
        train_new_model([
            'h3wb_part1.json',
            'h3wb_part2.json',
            'h3wb_part3.json'
        ])
        
        # Mit separatem Test-Set:
        train_new_model(
            ['h3wb_part1.json', 'h3wb_part2.json'],
            test_json='h3wb_test.json'
        )
    """
    if isinstance(h3wb_json, list):
        # Mehrere Dateien = Inkrementelles Training
        return train_on_h3wb_incremental(
            train_json_files=h3wb_json,
            test_json=test_json,
            output_model=output_model,
            epochs=epochs
        )
    else:
        # Einzelne Datei = Standard Training
        return train_on_h3wb(
            train_json=h3wb_json,
            test_json=test_json,
            output_model=output_model,
            epochs=epochs
        )

def convert_poses(
    input_json: str,
    output_json: str,
    model_path: str = 'lifting2DTo3D.pth'
):
    """⚡ Schnelle 2D→3D Konvertierung für einfache JSONs"""
    converter = Pose3DConverter(model_path=model_path, lifting_method='mlp')
    
    # Lade 2D Posen
    with open(input_json, 'r') as f:
        poses_2d = json.load(f)
    
    results = []
    # Konvertiere jeden Frame
    for frame_data in tqdm(poses_2d, desc="Converting"):
        # Extrahiere 2D Keypoints und Scores
        keypoints = np.array(frame_data['keypoints'])
        scores = np.array(frame_data['scores'])
        
        # Konvertiere zu 3D
        result = converter.convert_2d_to_3d(keypoints, scores)
        
        # Speichere Ergebnis
        results.append({
            'frame': frame_data['frame'],
            'keypoints_3d': result.keypoints_3d.tolist(),
            'scores': result.scores_3d.tolist(),
            'method': result.method
        })
    
    # Speichere Ergebnisse
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ {len(results)} Frames konvertiert → {output_json}")

# ===============================================
# 🚀 MAIN - NUTZUNGSBEISPIELE
# ===============================================
if __name__ == "__main__":
    print("=" * 60)
    print("🔄 Pose3DConverter - INKREMENTELLES TRAINING")
    print("=" * 60)
    print()
    print("📋 Verwendung:")
    print()
    print("1️⃣ TRAINING MIT EINZELNER DATEI:")
    print("   train_new_model('h3wb_dataset.json')")
    print()
    print("2️⃣ TRAINING MIT MEHREREN TEILEN (INKREMENTELL):")
    print("   train_new_model([")
    print("       'h3wb_part1.json',")
    print("       'h3wb_part2.json',")
    print("       'h3wb_part3.json'")
    print("   ])")
    print()
    print("3️⃣ MIT SEPARATEM TEST-SET:")
    print("   train_new_model(")
    print("       ['part1.json', 'part2.json'],")
    print("       test_json='test.json'")
    print("   )")
    print()
    print("4️⃣ POSEN KONVERTIEREN:")
    print("   convert_poses('2d.json', '3d.json', 'lifting2DTo3D.pth')")
    print()
    print("💡 VORTEILE INKREMENTELLES TRAINING:")
    print("   ✅ Geringerer RAM-Verbrauch")
    print("   ✅ Größere Datasets möglich")
    print("   ✅ Flexiblere Daten-Organisation")
    print("   ✅ Checkpoints bei jedem Teil")
    print("=" * 60)