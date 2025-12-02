"""
🖼️ PoseEstimator2D: Eine Python-Klasse für 2D-Körperpositions-Erkennung

EINFACHE ERKLÄRUNG:
Dieses Programm analysiert Bilder und Videos und findet darin Menschen.
Es zeigt an, wo sich Körperteile wie Kopf, Arme, Hände befinden.
Das ist wie eine digitale Version von "Mensch-ärgere-dich-nicht"-Figuren erkennen!

Funktioniert mit RTMLib (einer KI-Bibliothek für Posenschätzung)
Liefert 133 Körperpunkte pro Person:
- 17 Punkte für den Körper
- 68 Punkte für das Gesicht  
- 42 Punkte für die Hände
- 6 Punkte für die Füße

Autor: DGS Project Group 1
Datum: September 2025
"""

# ===============================================
# 📦 IMPORTIEREN DER BENÖTIGTEN BIBLIOTHEKEN
# ===============================================
import cv2  # 🖼️ Für Bilder und Videos (OpenCV - Computer Vision)
import numpy as np  # 🔢 Für Zahlen und Berechnungen
from pathlib import Path  # 📁 Für Dateipfade und Ordner
from typing import Union, List, Tuple, Optional  # 📝 Für bessere Code-Lesbarkeit
from dataclasses import dataclass  # 🏗️ Für strukturierte Daten-Container
import json  # 📄 Zum Speichern im JSON-Format (lesbar für Mensch und Computer)
import time  # ⏱️ Für Zeitmessungen

try:
    # 🤖 Versuche RTMLib zu laden (die KI-Motor)
    from rtmlib import Wholebody, draw_skeleton
except ImportError:
    # ❌ Falls nicht installiert: Installationsanleitung zeigen
    raise ImportError("RTMLib nicht gefunden. Installiere mit: pip install rtmlib")

# ===============================================
# ⚙️ KONFIGURATION: WELCHE KÖRPERTEILE WEGLASSEN?
# ===============================================
# Standardmäßig ignorierte Körperpunkte: Beine, Füße, Zehen (Punkte 13-22)
# Warum? Manchmal wollen wir uns nur auf Oberkörper konzentrieren!
DEFAULT_IGNORE_KEYPOINTS = list(range(13, 23))  # 🔢 Von Punkt 13 bis 22

# ===============================================
# 🔧 HILFSFUNKTION 1: BESTIMMTE KÖRPERPUNKTE AUSSCHALTEN
# ===============================================
def filter_keypoints(keypoints, scores, ignore_indices=None):
    """
    🎯 SETZT BESTIMMTE KÖRPERPUNKTE AUF "UNSICHTBAR"
    
    EINFACH GESAGT:
    Diese Funktion macht bestimmte Körperteile (z.B. Beine) unsichtbar,
    indem sie ihre Position auf (0,0) setzt und die Genauigkeit auf 0.
    
    BEISPIEL:
    Wenn wir nur Oberkörper analysieren wollen, schalten wir Beine aus.
    
    🔧 Parameter (Eingaben):
        keypoints: Liste von Körperpunkt-Positionen
        scores: Liste von Genauigkeitswerten (wie sicher ist die KI?)
        ignore_indices: Welche Punkte sollen ignoriert werden?
    
    📤 Rückgabe:
        Gefilterte keypoints und scores (Kopien der Originaldaten)
    """
    if ignore_indices is None:
        # 🚫 Keine Filterung: Einfach Kopien zurückgeben
        return keypoints.copy(), scores.copy()
    
    # 📋 Kopien der Originaldaten erstellen (wir ändern Original NICHT!)
    keypoints_filtered = keypoints.copy()
    scores_filtered = scores.copy()
    
    # 🔄 Für jeden zu ignorierenden Punkt...
    for idx in ignore_indices:
        if idx < keypoints_filtered.shape[1]:  # ✅ Prüfen ob Punkt existiert
            keypoints_filtered[:, idx, :] = 0  # 🎯 Position auf (0,0) setzen
            scores_filtered[:, idx] = 0        # 🎯 Genauigkeit auf 0 setzen
    
    return keypoints_filtered, scores_filtered

# ===============================================
# 🎨 HILFSFUNKTION 2: SKELETT-LINIEN ZEICHNEN
# ===============================================
def draw_skeleton_filtered(image, keypoints, scores, ignore_indices=None, kpt_thr=0.3):
    """
    🖍️ ZEICHNET KÖRPER-LINIEN OHNE IGNORIERTE BEREICHE
    
    EINFACH GESAGT:
    Malt grüne Linien zwischen Körperpunkten und rote Punkte auf die Positionen.
    Überspringt dabei Körperteile, die wir nicht sehen wollen (z.B. Beine).
    
    🖼️ Beispiel-Output:
        ○ Kopf
        ├──○ Linke Schulter
        │  └──○ Linker Ellbogen
        │     └──○ Linkes Handgelenk
        └──○ Rechte Schulter
           └──○ Rechter Ellbogen
              └──○ Rechtes Handgelenk
    
    🔧 Parameter:
        image: Das Original-Bild (wird nicht verändert!)
        keypoints: Körperpunkt-Positionen
        scores: Genauigkeitswerte
        ignore_indices: Zu ignorierende Punkte
        kpt_thr: Mindest-Genauigkeit zum Zeichnen (0.3 = 30% sicher)
    
    📤 Rückgabe:
        Annotiertes Bild mit gezeichnetem Skelett
    """
    if ignore_indices is None:
        # 🎨 Fallback: Verwende Standard-Zeichenfunktion von RTMLib
        from rtmlib import draw_skeleton
        return draw_skeleton(image, keypoints, scores, kpt_thr=kpt_thr)
    
    # 🦴 DEFINITION DER KÖRPER-VERBINDUNGEN (OHNE BEINE!)
    # Welche Punkte sollen mit Linien verbunden werden?
    BODY_CONNECTIONS = [
        (53, 1), (53, 2), (1, 3), (2, 4),  # 👤 Kopf (Punkt 53 = Nase)
        (3, 5), (4, 6), (5, 6),           # 🎯 Schultern
        (5, 7), (7, 91),                  # 💪 Linker Arm
        (6, 8), (8, 112),                 # 💪 Rechter Arm
        (5, 11), (6, 12), (11, 12),       # 🏋️ Torso (Oberkörper)
    ]
    
    # 📋 Kopie des Originalbildes (wir malen auf die Kopie!)
    annotated = image.copy()
    # ⚡ Schneller Zugriff: Set aus ignore_indices machen
    ignore_set = set(ignore_indices)
    
    # 👥 Für jede Person im Bild...
    for person_idx in range(len(keypoints)):
        kpts = keypoints[person_idx]  # 📍 Punkte dieser Person
        conf = scores[person_idx]     # 🎯 Genauigkeiten dieser Person
        
        # 🖍️ LINIEN ZEICHNEN (Verbindungen zwischen Punkten)
        for start_idx, end_idx in BODY_CONNECTIONS:
            # ✅ Prüfen: Beide Punkte NICHT ignoriert?
            if start_idx not in ignore_set and end_idx not in ignore_set:
                # ✅ Prüfen: Beide Punkte genug sicher?
                if conf[start_idx] > kpt_thr and conf[end_idx] > kpt_thr:
                    pt1 = tuple(kpts[start_idx].astype(int))  # 🎯 Start-Punkt
                    pt2 = tuple(kpts[end_idx].astype(int))    # 🎯 End-Punkt
                    # 🟢 Grüne Linie zeichnen (Farbe: 0,255,0, Dicke: 1)
                    cv2.line(annotated, pt1, pt2, (0, 255, 0), 1)
        
        # 🔴 PUNKTE ZEICHNEN (Einzelne Körperpunkte)
        for idx in range(len(kpts)):
            # ✅ Prüfen: Punkt nicht ignoriert und genug sicher?
            if idx not in ignore_set and conf[idx] > kpt_thr:
                pt = tuple(kpts[idx].astype(int))  # 🎯 Punkt-Position
                # 🔴 Roten Punkt zeichnen (Radius: 1, komplett ausgefüllt)
                cv2.circle(annotated, pt, 1, (0, 0, 255), -1)
    
    return annotated

# ===============================================
# 📦 DATENKLASSE 1: ERGEBNIS FÜR EIN EINZELBILD
# ===============================================
@dataclass
class PoseResult:
    """
    🏷️ EIN "DATEN-BEHÄLTER" FÜR EINZELBILD-ERGEBNISSE
    
    Stell dir das vor wie ein digitales Formular, das alle Infos zu 
    einer Posenerkennung in einem Bild speichert.
    
    📋 INHALT:
        frame_idx:     Bild-Nummer (bei Videos)
        keypoints:     Körperpunkt-Positionen [Personen, 133 Punkte, X/Y]
        scores:        Genauigkeiten für jeden Punkt [Personen, 133 Punkte]
        bboxes:        Begrenzungsrahmen um Personen [Personen, 5 Werte]
        num_persons:   Anzahl der gefundenen Personen
    """
    frame_idx: int
    keypoints: np.ndarray
    scores: np.ndarray
    bboxes: np.ndarray
    num_persons: int

# ===============================================
# 📦 DATENKLASSE 2: ERGEBNIS FÜR EIN GANZES VIDEO
# ===============================================
@dataclass
class VideoResult:
    """
    🎞️ EIN "DATEN-BEHÄLTER" FÜR VIDEO-ERGEBNISSE
    
    Speichert alle Einzelbild-Ergebnisse eines Videos plus Video-Infos.
    
    📋 INHALT:
        frame_results:    Liste von PoseResult für jedes Bild
        total_frames:     Anzahl aller verarbeiteten Bilder
        fps:              Bilder pro Sekunde im Original-Video
        processing_time:  Verarbeitungszeit in Sekunden
    """
    frame_results: List[PoseResult]
    total_frames: int
    fps: float
    processing_time: float

# ===============================================
# 🚀 HAUPTKLASSE: DER POSE-ESTIMATOR
# ===============================================
class PoseEstimator2D:
    """
    🤖 DIE HAUPTKLASSE FÜR 2D-POSENERKENNUNG
    
    EINFACH GESAGT:
    Dies ist unser "digitaler Body-Detektor". Er kann:
    1. 🖼️ In Bildern Menschen finden
    2. 🎞️ In Videos Menschen verfolgen
    3. 📍 Genau zeigen, wo Körperteile sind
    4. 💾 Ergebnisse speichern und exportieren
    
    So nutzt du es:
        estimator = PoseEstimator2D(device='cpu')
        result = estimator.process_image("mein_bild.jpg")
    """
    
    def __init__(
        self,
        mode: str = 'performance',      # 🥇 Beste Genauigkeit
        backend: str = 'onnxruntime',   # 🏗️ KI-Ausführungs-Engine
        device: str = 'cpu',            # 💻 Hardware (cpu, cuda für NVIDIA, mps für Apple)
        to_openpose: bool = False,      # 🔀 OpenPose-Format konvertieren?
        kpt_threshold: float = 0.8      # 🎯 Mindest-Genauigkeit für Punkte (80%)
    ):
        """
        🏗️ KONSTRUKTOR: INITIALISIERT DEN ESTIMATOR
        
        Hier wird der KI-Motor (RTMLib) gestartet und konfiguriert.
        """
        self.mode = mode
        self.backend = backend
        self.devend = device
        self.to_openpose = to_openpose
        self.kpt_threshold = kpt_threshold
        
        try:
            # 🤖 RTMLib KI-Modell laden (133-Punkte-Ganzkörper-Modell)
            self.model = Wholebody(
                mode=mode,
                backend=backend,
                device=device,
                to_openpose=to_openpose
            )
            print(f"✅ RTMLib Wholebody geladen mit:")
            print(f"   Modus: {mode}, Backend: {backend}, Gerät: {device}")
        except Exception as e:
            raise RuntimeError(f"❌ RTMLib konnte nicht geladen werden: {e}")
    
    def _replace_keypoints(self, keypoints: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        🔄 ERSETZT BESTIMMTE KÖRPERPUNKTE DURCH GENAUERE
        
        WARUM?
        Die KI hat zwei Arten von Nasen- und Handgelenk-Punkten:
        1. Von Körper-Erkennung (weniger genau)
        2. Von Gesichts-/Hand-Erkennung (genauer)
        
        👃 Beispiel Nase:
            - Punkt 0: Körper-Nase (ungefähr)
            - Punkt 53: Gesichts-Nase (genau)
            → Wir nehmen Punkt 53!
        
        ✋ Beispiel Handgelenke:
            - Punkt 9: Linkes Körper-Handgelenk
            - Punkt 91: Linkes Hand-Handgelenk (aus Hand-Erkennung)
            → Wir nehmen Punkt 91!
        
        🔧 Parameter:
            keypoints: Alle 133 Punkte pro Person
            scores: Genauigkeiten aller Punkte
            
        📤 Rückgabe:
            Verbesserte keypoints und scores
        """
        keypoints_modified = keypoints.copy()  # 📋 Kopie
        scores_modified = scores.copy()        # 📋 Kopie
        
        # 👥 Für jede erkannte Person...
        for person_idx in range(len(keypoints)):
            # 1. 👃 NASE ERSETZEN (Punkt 0 durch 53)
            if scores[person_idx, 53] > 0:  # ✅ Wenn Gesichts-Nase erkannt
                keypoints_modified[person_idx, 0] = keypoints[person_idx, 53]
                scores_modified[person_idx, 0] = scores[person_idx, 53]
            else:
                # 🚫 Keine Gesichts-Nase: Körper-Nase unsichtbar machen
                keypoints_modified[person_idx, 0] = 0
                scores_modified[person_idx, 0] = 0
            
            # 2. ✋ LINKES HANDGELENK ERSETZEN (9 durch 91)
            if scores[person_idx, 91] > 0:
                keypoints_modified[person_idx, 9] = keypoints[person_idx, 91]
                scores_modified[person_idx, 9] = scores[person_idx, 91]
            else:
                keypoints_modified[person_idx, 9] = 0
                scores_modified[person_idx, 9] = 0
            
            # 3. ✋ RECHTES HANDGELENK ERSETZEN (10 durch 112)
            if scores[person_idx, 112] > 0:
                keypoints_modified[person_idx, 10] = keypoints[person_idx, 112]
                scores_modified[person_idx, 10] = scores[person_idx, 112]
            else:
                keypoints_modified[person_idx, 10] = 0
                scores_modified[person_idx, 10] = 0
        
        return keypoints_modified, scores_modified
    
    def _process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> PoseResult:
        """
        🎯 KERN-FUNKTION: ANALYSIERT EIN EINZELBILD
        
        Hier passiert die Magie: KI analysiert Bild → findet Menschen → berechnet Punkte.
        
        🔧 Parameter:
            frame: Das Bild als numpy Array (BGR Format)
            frame_idx: Bild-Nummer (für Videos wichtig)
            
        📤 Rückgabe:
            PoseResult mit allen Ergebnissen
        """
        try:
            # ===============================================
            # 📥 SCHRITT 1: BILD MIT KI ANALYSIEREN
            # ===============================================
            keypoints, scores = self.model(frame)  # 🤖 KI sagt: "Hier sind Menschen!"
            
            # 🚫 Prüfen: Wurden überhaupt Personen gefunden?
            if keypoints is None or len(keypoints) == 0:
                return PoseResult(
                    frame_idx=frame_idx,
                    keypoints=np.empty((0, 133, 2)),  # 📭 Leeres Array: 0 Personen
                    scores=np.empty((0, 133)),
                    bboxes=np.empty((0, 5)),
                    num_persons=0
                )
            
            # ===============================================
            # 🔢 SCHRITT 2: DATEN IN RICHTIGES FORMAT BRINGEN
            # ===============================================
            keypoints = np.array(keypoints)  # 🔄 In numpy Array umwandeln
            scores = np.array(scores)        # 🔄 Genauigkeiten umwandeln
            
            # 📊 KI-Interne Werte in Prozente (0-100%) umrechnen
            logits = np.array(scores)
            confidence_scores = 1 / (1 + np.exp(-logits))  # 🧮 Mathe-Formel
            
            # 🔧 Sicherstellen: Arrays haben richtige Dimensionen
            if keypoints.ndim == 2:
                keypoints = keypoints[np.newaxis, ...]  # 👥 Person-Dimension hinzufügen
            
            if confidence_scores.ndim == 1:
                confidence_scores = confidence_scores[np.newaxis, ...]
            
            num_persons = keypoints.shape[0]  # 👥 Wie viele Personen?
            
            # ===============================================
            # 🔄 SCHRITT 3: PUNKTE VERBESSERN (Genauere Versionen nehmen)
            # ===============================================
            keypoints, confidence_scores = self._replace_keypoints(keypoints, confidence_scores)
            
            # ===============================================
            # 📦 SCHRITT 4: BEGRENZUNGSRAHMEN BERECHNEN
            # ===============================================
            # 🔲 Grüne Rechtecke um jede Person berechnen
            bboxes = []
            for i in range(num_persons):
                kpts = keypoints[i].copy()  # 📍 Punkte dieser Person
                conf_scores_flat = confidence_scores[i]  # 🎯 Genauigkeiten
                
                # 🚫 Punkte mit niedriger Genauigkeit ignorieren
                low_confidence_mask = conf_scores_flat <= self.kpt_threshold
                kpts[low_confidence_mask, 0] = 0  # X-Koordinate auf 0
                kpts[low_confidence_mask, 1] = 0  # Y-Koordinate auf 0
                keypoints[i] = kpts  # 📋 Zurückspeichern
                
                # ✅ Finde gültige Punkte (nicht 0,0)
                non_zero_mask = (kpts != 0).any(axis=1)
                valid_kpts = kpts[non_zero_mask]
                
                if len(valid_kpts) > 0:
                    # 📐 Rechteck berechnen: min/max von X und Y
                    x_coords = valid_kpts[:, 0]
                    y_coords = valid_kpts[:, 1]
                    x1, y1 = np.min(x_coords), np.min(y_coords)  # ↖️ Oben links
                    x2, y2 = np.max(x_coords), np.max(y_coords)  # ↘️ Unten rechts
                    
                    # ⬜ 20 Pixel Rand hinzufügen
                    padding = 20
                    x1 = max(0, x1 - padding)          # Nicht kleiner als 0
                    y1 = max(0, y1 - padding)          # Nicht kleiner als 0
                    x2 = min(frame.shape[1], x2 + padding)  # Nicht breiter als Bild
                    y2 = min(frame.shape[0], y2 + padding)  # Nicht höher als Bild
                    
                    # 🎯 Durchschnitts-Genauigkeit berechnen
                    high_confidence_scores = conf_scores_flat[conf_scores_flat > self.kpt_threshold]
                    confidence = np.mean(high_confidence_scores) if len(high_confidence_scores) > 0 else 0
                    
                    # 📦 Rahmen zur Liste hinzufügen [x1, y1, x2, y2, confidence]
                    bboxes.append([x1, y1, x2, y2, confidence])
                else:
                    # 🚫 Keine gültigen Punkte: Leeren Rahmen
                    bboxes.append([0, 0, 0, 0, 0])
            
            bboxes_array = np.array(bboxes)  # 🔄 In numpy Array
            
            # ===============================================
            # 📊 SCHRITT 5: DEBUG-AUSGABE (Für Entwickler)
            # ===============================================
            print(f"Frame {frame_idx}: Punkt 0 (Nase) = {keypoints[0, 0]}")
            print(f"Frame {frame_idx}: Punkt 53 (Gesichts-Nase) = {keypoints[0, 53]}")
            print(f"Frame {frame_idx}: Punkt 91 (Hand-Handgelenk) = {keypoints[0, 91]}")
            
            # ===============================================
            # 📤 SCHRITT 6: ERGEBNIS ZURÜCKGEBEN
            # ===============================================
            return PoseResult(
                frame_idx=frame_idx,
                keypoints=keypoints,          # 📍 Verbesserte Punkte
                scores=confidence_scores,     # 🎯 Genauigkeiten
                bboxes=bboxes_array,          # 🔲 Begrenzungsrahmen
                num_persons=num_persons       # 👥 Anzahl Personen
            )
            
        except Exception as e:
            # ❌ Falls Fehler: Fehlermeldung und leeres Ergebnis
            print(f"❌ Fehler bei Frame {frame_idx}: {e}")
            return PoseResult(
                frame_idx=frame_idx,
                keypoints=np.empty((0, 133, 2)),
                scores=np.empty((0, 133)),
                bboxes=np.empty((0, 5)),
                num_persons=0
            )
    
    def process_image(self, image_path: Union[str, Path]) -> PoseResult:
        """
        🖼️ ANALYSIERT EIN EINZELNES BILD
        
        🔧 Parameter:
            image_path: Pfad zum Bild (jpg, png, etc.)
            
        📤 Rückgabe:
            PoseResult mit den gefundenen Personen
            
        🚨 Mögliche Fehler:
            FileNotFoundError: Bild existiert nicht
            ValueError: Bild kann nicht geladen werden
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"❌ Bild nicht gefunden: {image_path}")
        
        # 🖼️ Bild laden
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"❌ Bild kann nicht geladen werden: {image_path}")
        
        print(f"📸 Verarbeite Bild: {image_path}")
        print(f"📐 Größe: {frame.shape[1]}x{frame.shape[0]} Pixel")
        
        # 🎯 Bild analysieren
        result = self._process_frame(frame, frame_idx=0)
        
        print(f"👤 Gefunden: {result.num_persons} Person(en)")
        return result
    
    def process_image_with_annotation(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        draw_bbox: bool = True,           # 🔲 Grüne Rechtecke zeichnen?
        draw_keypoints: bool = True,      # 🎯 Punkte und Linien zeichnen?
        keypoint_threshold: float = 0.3,  # 🎯 Mindest-Genauigkeit für Zeichnen
        ignore_keypoints: Optional[List[int]] = None  # 🚫 Zu ignorierende Punkte
    ) -> PoseResult:
        """
        🖼️📝 ANALYSIERT BILD UND SPEICHERT ANNOTIERTE VERSION
        
        🔧 Parameter:
            image_path: Pfad zum Eingabebild
            output_path: Wo annotiertes Bild speichern? (optional)
            draw_bbox: Begrenzungsrahmen zeichnen?
            draw_keypoints: Skelett zeichnen?
            keypoint_threshold: Wie sicher muss Punkt sein zum Zeichnen?
            ignore_keypoints: Welche Punkte ignorieren? (z.B. Beine)
            
        📤 Rückgabe:
            PoseResult + gespeichertes Bild (falls output_path)
        """
        image_path = Path(image_path)
        
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)  # 📁 Ordner erstellen
        
        # 🖼️ Bild laden
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"❌ Bild kann nicht geladen werden: {image_path}")
        
        print(f"📸 Verarbeite Bild: {image_path}")
        
        # 🎯 Bild analysieren
        result = self._process_frame(frame, frame_idx=0)
        
        # 🚫 Optional: Bestimmte Punkte filtern (z.B. Beine)
        if ignore_keypoints is not None:
            result.keypoints, result.scores = filter_keypoints(
                result.keypoints, 
                result.scores, 
                ignore_keypoints
            )
        
        # 🖍️ Kopie für Annotationen erstellen
        annotated_frame = frame.copy()
        
        # 👥 Falls Personen gefunden...
        if result.num_persons > 0:
            # 🔲 Grüne Rechtecke zeichnen
            if draw_bbox and len(result.bboxes) > 0:
                for bbox in result.bboxes:
                    x1, y1, x2, y2 = bbox[:4].astype(int)  # 📐 Koordinaten
                    # 🟩 Grünes Rechteck (Farbe: 0,255,0, Dicke: 2)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 🎯 Punkte und Skelett zeichnen
            if draw_keypoints:
                annotated_frame = draw_skeleton_filtered(
                    annotated_frame,
                    result.keypoints,
                    result.scores,
                    ignore_keypoints,
                    kpt_thr=keypoint_threshold
                )
        
        # 💾 Annotiertes Bild speichern (falls gewünscht)
        if output_path is not None:
            cv2.imwrite(str(output_path), annotated_frame)
            print(f"💾 Annotiertes Bild gespeichert: {output_path}")
        
        return result
    
    def process_video(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_frames: bool = False,       # 🖼️ Einzelbilder speichern?
        max_frames: Optional[int] = None # 🔢 Maximale Anzahl Bilder
    ) -> VideoResult:
        """
        🎞️ ANALYSIERT EIN GANZES VIDEO
        
        🔧 Parameter:
            video_path: Pfad zum Video (mp4, avi, etc.)
            output_dir: Wo Ergebnisse speichern? (optional)
            save_frames: Einzelbilder mit Annotationen speichern?
            max_frames: Nur erste X Bilder analysieren (schneller Test)
            
        📤 Rückgabe:
            VideoResult mit allen Einzelbild-Ergebnissen
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"❌ Video nicht gefunden: {video_path}")
        
        # 🎬 Video öffnen
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"❌ Video kann nicht geöffnet werden: {video_path}")
        
        # 📊 Video-Eigenschaften lesen
        fps = cap.get(cv2.CAP_PROP_FPS)  # 🎞️ Bilder pro Sekunde
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 🔢 Gesamtanzahl
        
        if max_frames:
            total_frames = min(total_frames, max_frames)  # 🔢 Begrenzen
        
        print(f"🎬 Verarbeite Video: {video_path}")
        print(f"📊 FPS: {fps}, Gesamte Bilder: {total_frames}")
        
        # 📁 Ausgabeordner vorbereiten
        if output_dir and save_frames:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 📋 Liste für alle Ergebnisse
        frame_results = []
        start_time = time.time()  # ⏱️ Startzeit messen
        
        # 🔄 Alle Bilder/Frames durchgehen
        for frame_idx in range(total_frames):
            ret, frame = cap.read()  # 📷 Nächstes Bild lesen
            if not ret:  # 🏁 Videoende erreicht?
                break
            
            # 🎯 Bild analysieren
            result = self._process_frame(frame, frame_idx)
            frame_results.append(result)
            
            # 💾 Annotiertes Bild speichern (falls gewünscht)
            if save_frames and output_dir and result.num_persons > 0:
                annotated_frame = draw_skeleton_filtered(
                    frame.copy(),
                    result.keypoints,
                    result.scores,
                    kpt_thr=self.kpt_threshold
                )
                frame_filename = output_dir / f"frame_{frame_idx:05d}.jpg"
                cv2.imwrite(str(frame_filename), annotated_frame)
            
            # 📊 Fortschritt anzeigen (alle 30 Bilder)
            if frame_idx % 30 == 0:
                print(f"📊 Verarbeitet: {frame_idx}/{total_frames} Bilder")
        
        cap.release()  # 🎬 Video schließen
        
        processing_time = time.time() - start_time
        print(f"✅ Fertig in {processing_time:.2f} Sekunden")
        
        return VideoResult(
            frame_results=frame_results,
            total_frames=len(frame_results),
            fps=fps,
            processing_time=processing_time
        )
    
    def export_to_json(
        self,
        result: Union[PoseResult, VideoResult],
        output_path: Union[str, Path],
        include_scores: bool = True  # 🎯 Genauigkeiten mit speichern?
    ) -> None:
        """
        📄 EXPORTIERT ERGEBNISSE ALS JSON-DATEI
        
        JSON ist wie ein digitales Notizbuch:
        - Menschen-lesbar
        - Computer-lesbar
        - Universell kompatibel
        
        🔧 Parameter:
            result: PoseResult oder VideoResult
            output_path: Wo JSON speichern?
            include_scores: Genauigkeiten mit exportieren?
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 🖼️ Einzelbild-Result
        if isinstance(result, PoseResult):
            data = {
                "frame_idx": int(result.frame_idx),
                "num_persons": int(result.num_persons),
                "keypoints": result.keypoints.tolist(),  # 🔄 numpy → Liste
                "bboxes": result.bboxes.tolist()
            }
            if include_scores:
                data["scores"] = result.scores.tolist()
        
        # 🎞️ Video-Result
        elif isinstance(result, VideoResult):
            data = {
                "total_frames": result.total_frames,
                "fps": result.fps,
                "processing_time": result.processing_time,
                "frames": []
            }
            
            # 🔄 Für jedes Bild im Video...
            for frame_result in result.frame_results:
                frame_data = {
                    "frame_idx": int(frame_result.frame_idx),
                    "num_persons": int(frame_result.num_persons),
                    "keypoints": frame_result.keypoints.tolist(),
                    "bboxes": frame_result.bboxes.tolist()
                }
                if include_scores:
                    frame_data["scores"] = frame_result.scores.tolist()
                
                data["frames"].append(frame_data)
        
        else:
            raise ValueError("❌ Result muss PoseResult oder VideoResult sein")
        
        # 💾 In Datei speichern
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)  # 📝 Schön formatiert (Einrückung: 2)
        
        print(f"💾 JSON exportiert: {output_path}")
    
    def get_summary(self, result: Union[PoseResult, VideoResult]) -> str:
        """
        📋 ERSTELLT EINE ZUSAMMENFASSUNG
        
        🔧 Parameter:
            result: PoseResult oder VideoResult
            
        📤 Rückgabe:
            Formatierte Zusammenfassung als Text
        """
        if isinstance(result, PoseResult):
            # 🖼️ Einzelbild-Zusammenfassung
            summary = f"=== Pose Estimation Summary ===\n"
            summary += f"Bild: {result.frame_idx}\n"
            summary += f"Gefundene Personen: {result.num_persons}\n"
            
            if result.num_persons > 0:
                for i in range(result.num_persons):
                    # 📊 Wie viele sichere Punkte?
                    valid_kpts = np.sum(result.scores[i] > self.kpt_threshold)
                    # 🎯 Durchschnittliche Genauigkeit
                    avg_confidence = np.mean(result.scores[i][result.scores[i] > self.kpt_threshold])
                    summary += f"Person {i+1}: {valid_kpts}/133 Punkte, Genauigkeit: {avg_confidence:.1%}\n"
        
        elif isinstance(result, VideoResult):
            # 🎞️ Video-Zusammenfassung
            total_persons = sum(fr.num_persons for fr in result.frame_results)
            frames_with_detection = sum(1 for fr in result.frame_results if fr.num_persons > 0)
            
            summary = f"=== Video Processing Summary ===\n"
            summary += f"Gesamte Bilder: {result.total_frames}\n"
            summary += f"Bilder pro Sekunde: {result.fps:.2f}\n"
            summary += f"Verarbeitungszeit: {result.processing_time:.2f}s\n"
            summary += f"Bilder mit Personen: {frames_with_detection}/{result.total_frames}\n"
            summary += f"Gesamt Personen-Erkennungen: {total_persons}\n"
            
            if total_persons > 0:
                avg_persons_per_frame = total_persons / result.total_frames
                summary += f"Durchschnitt pro Bild: {avg_persons_per_frame:.2f} Personen\n"
        
        else:
            summary = "❌ Ungültiger Result-Typ"
        
        return summary

# ===============================================
# ⚡ BEQUEMLICHKEITSFUNKTIONEN (Schnellstart)
# ===============================================
# Diese Funktionen sind für "Ich will jetzt sofort loslegen!"

def estimate_pose_image(
    image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    mode: str = 'performance',  # 🥇 Beste Genauigkeit
    device: str = 'cpu'         # 💻 Auf CPU laufen lassen
) -> PoseResult:
    """
    ⚡ SCHNELLE FUNKTION FÜR EINZELBILD-ANALYSE
    
    BEISPIEL:
        result = estimate_pose_image("urlaub.jpg", "urlaub_pose.jpg")
    
    🔧 Parameter:
        image_path: Pfad zum Bild
        output_path: Wo annotiertes Bild speichern? (optional)
        mode: KI-Modus ('performance', 'balanced', 'lightweight')
        device: Hardware ('cpu', 'cuda', 'mps')
        
    📤 Rückgabe:
        PoseResult
    """
    # 🤖 Estimator erstellen
    estimator = PoseEstimator2D(mode=mode, device=device)
    
    # 🎯 Bild analysieren (mit oder ohne Annotation)
    if output_path:
        return estimator.process_image_with_annotation(image_path, output_path)
    else:
        return estimator.process_image(image_path)

def estimate_pose_video(
    video_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    mode: str = 'performance',
    device: str = 'cpu',
    max_frames: Optional[int] = None
) -> VideoResult:
    """
    ⚡ SCHNELLE FUNKTION FÜR VIDEO-ANALYSE
    
    BEISPIEL:
        result = estimate_pose_video("tanzen.mp4", "ergebnisse/")
    
    🔧 Parameter:
        video_path: Pfad zum Video
        output_dir: Wo Ergebnisse speichern? (optional)
        mode: KI-Modus
        device: Hardware
        max_frames: Maximale Anzahl Bilder
        
    📤 Rückgabe:
        VideoResult
    """
    estimator = PoseEstimator2D(mode=mode, device=device)
    return estimator.process_video(
        video_path,
        output_dir=output_dir,
        save_frames=bool(output_dir),  # 💾 Nur speichern wenn output_dir gegeben
        max_frames=max_frames
    )

# ===============================================
# 🚀 START: WENN DAS PROGRAMM DIREKT GESTARTET WIRD
# ===============================================
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 RTMLib Pose Estimator 2D - Test Script")
    print("=" * 60)
    print("📝 Testet die Posenerkennung mit einem Beispielbild")
    print("")
    
    # 🔍 Testbild suchen
    test_image = Path("../data/test_pose.png")
    
    if test_image.exists():
        print(f"✅ Testbild gefunden: {test_image}")
        print("")
        
        # 🤖 Estimator erstellen (ausgewogener Modus, auf CPU)
        print("1️⃣  Erstelle Pose-Estimator...")
        estimator = PoseEstimator2D(mode='balanced', device='cpu')
        
        # 🎯 Bild analysieren
        print("2️⃣  Analysiere Bild...")
        result = estimator.process_image(test_image)
        
        # 📊 Zusammenfassung anzeigen
        print("3️⃣  Zeige Zusammenfassung:")
        print(estimator.get_summary(result))
        
        # 🖍️ Annotiertes Ergebnis speichern
        print("4️⃣  Speichere annotiertes Bild...")
        output_path = Path("../output/pose-estimation/test_result.png")
        estimator.process_image_with_annotation(test_image, output_path)
        
        # 📄 In JSON exportieren
        print("5️⃣  Exportiere als JSON...")
        json_path = Path("../output/pose-estimation/test_result.json")
        estimator.export_to_json(result, json_path)
        
        print("")
        print("=" * 40)
        print("🎉 Alles fertig! Ergebnisse im Ordner 'output/'")
        print("")
        print("📁 Du findest:")
        print("   - test_result.png (Bild mit eingezeichneten Personen)")
        print("   - test_result.json (Daten aller Körperpunkte)")
        
    else:
        print(f"⚠️  Kein Testbild gefunden: {test_image}")
        print("")
        print("ℹ️  So kannst du es trotzdem nutzen:")
        print("")
        print("🖼️  FÜR BILDER:")
        print("    result = estimate_pose_image('mein_bild.jpg')")
        print("    result = estimate_pose_image('mein_bild.jpg', 'ergebnis.jpg')")
        print("")
        print("🎞️  FÜR VIDEOS:")
        print("    result = estimate_pose_video('mein_video.mp4', 'ergebnisse/')")
        print("")
        print("💡 TIPPS:")
        print("    - Verwende mode='lightweight' für schnellere Analyse")
        print("    - Verwende device='cuda' falls du NVIDIA GPU hast")
        print("    - max_frames=100 für schnellen Test mit Videos")