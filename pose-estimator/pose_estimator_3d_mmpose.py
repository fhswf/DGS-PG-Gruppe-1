"""
🔄 Pose3DConverter: 2D → 3D Körperpositionen konvertieren

EINFACHE ERKLÄRUNG:
Dieses Programm nimmt 2D-Bilder mit erkannten Körperpunkten 
und macht daraus 3D-Modelle! Stell es dir vor wie:

🖼️ 2D-Foto → 🪄 Magie → 🎯 3D-Figur

Es funktioniert so:
1. Nimmt Punkte von einem 2D-Bild (X, Y Koordinaten)
2. Schätzt die Tiefe (Z-Koordinate)
3. Erstellt daraus eine 3D-Figur, die man von allen Seiten betrachten kann

Besonderheit: Behält alle wichtigen Punkte bei - genau wie im 2D-Wrapper!
"""

# ===============================================
# 📦 IMPORTIEREN DER BENÖTIGTEN BIBLIOTHEKEN
# ===============================================
import numpy as np  # 🔢 Für Mathe und 3D-Berechnungen
from pathlib import Path  # 📁 Für Dateipfade
from typing import Union, List, Tuple, Optional, Dict  # 📝 Für bessere Code-Lesbarkeit
from dataclasses import dataclass  # 🏗️ Für strukturierte Daten-Container
import json  # 📄 Zum Speichern im JSON-Format
import warnings  # ⚠️ Für Warnmeldungen

try:
    # 🤖 Versuche MMPose zu laden (fortgeschrittene 3D-Posenschätzung)
    from mmpose.apis import MMPoseInferencer
    MMPOSE_AVAILABLE = True
except ImportError:
    # ℹ️ Falls nicht verfügbar: trotzdem weitermachen (geometrische Methode geht immer)
    MMPOSE_AVAILABLE = False

# ===============================================
# ⚙️ KONFIGURATION: WELCHE KÖRPERTEILE WEGLASSEN?
# ===============================================
# Standardmäßig ignorierte Körperpunkte: Beine, Füße, Zehen (Punkte 13-22)
DEFAULT_IGNORE_KEYPOINTS = list(range(13, 23))

# ===============================================
# 📦 DATENKLASSE: 3D-ERGEBNISSE SPEICHERN
# ===============================================
@dataclass
class Pose3DResult:
    """
    🏷️ EIN "DATEN-BEHÄLTER" FÜR 3D-ERGEBNISSE
    
    Speichert alle Informationen zu einer 3D-Körperposition.
    
    📋 INHALT:
        frame_idx:      Bild-Nummer
        keypoints_3d:   3D-Körperpunkte [Personen, 133 Punkte, X/Y/Z]
        keypoints_2d:   Original 2D-Punkte [Personen, 133 Punkte, X/Y]
        scores_3d:      Genauigkeiten in 3D [Personen, 133 Punkte]
        bboxes_3d:      3D-Begrenzungsrahmen [Personen, 7 Werte]
        num_persons:    Anzahl der Personen
        method:         Welche Methode wurde verwendet?
        confidence:     Durchschnittliche Genauigkeit
    """
    frame_idx: int
    keypoints_3d: np.ndarray
    keypoints_2d: np.ndarray
    scores_3d: np.ndarray
    bboxes_3d: np.ndarray
    num_persons: int
    method: str
    confidence: float

# ===============================================
# 🔄 HAUPTKLASSE: DER 3D-KONVERTER
# ===============================================
class Pose3DConverter:
    """
    🪄 DIE HAUPTKLASSE FÜR 2D→3D KONVERTIERUNG
    
    EINFACH GESAGT:
    Nimmt flache 2D-Bilder und macht sie "tief" - wie aus einem Foto 
    eine kleine 3D-Figur für ein Computerspiel zu machen.
    
    So funktioniert die "Magie":
    1. 📍 Kopiert alle 2D-Punkte (X, Y)
    2. 🔍 Schätzt für jeden Punkt die Tiefe (Z)
    3. 🎯 Behält alle wichtigen Punkte bei (Hände, Gesicht)
    4. 🚫 Entfernt Beine (wenn gewünscht)
    """
    
    def __init__(
        self,
        lifting_method: str = 'geometric',  # 🔧 Methode: 'geometric' (einfach & zuverlässig)
        mmpose_model: str = 'human3d',      # 🤖 Fortgeschrittenes KI-Modell (optional)
        mmpose_weights: Optional[str] = None,  # ⚖️ KI-Gewichte (falls KI verwendet)
        device: str = 'cpu',                # 💻 Hardware (cpu, cuda für NVIDIA)
        ignore_keypoints: Optional[List[int]] = None,  # 🚫 Zu ignorierende Punkte
        image_width: int = 1920,            # 📐 Standard-Bildbreite
        image_height: int = 1080            # 📐 Standard-Bildhöhe
    ):
        """
        🏗️ KONSTRUKTOR: INITIALISIERT DEN 3D-KONVERTER
        
        Hier wird festgelegt, wie die Konvertierung funktionieren soll.
        """
        self.lifting_method = lifting_method
        self.device = device
        self.ignore_keypoints = ignore_keypoints if ignore_keypoints is not None else DEFAULT_IGNORE_KEYPOINTS
        self.image_width = image_width
        self.image_height = image_height
        self.lifting_method = 'geometric'  # 🎯 IMMER geometrische Methode (konsistent)
        
        print(f"✅ Pose3DConverter bereit!")
        print(f"   Methode: {self.lifting_method}")
        print(f"   Ignoriere Punkte: {self.ignore_keypoints}")
        print(f"   Bildgröße: {image_width}x{image_height}")
    
    def _copy_keypoints_from_2d(self, keypoints_2d: np.ndarray, scores_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        📋 KOPIERT 2D-PUNKTE FÜR 3D - OHNE ÄNDERUNGEN!
        
        WICHTIGSTE REGEL: "Mach genau das gleiche wie im 2D-Wrapper!"
        
        Warum?
        - Im 2D-Wrapper wurden schon Punkte ersetzt (Nase, Handgelenke)
        - Diese Änderungen müssen in 3D beibehalten werden
        - Sonst passen 2D und 3D nicht zusammen!
        
        🔧 Parameter:
            keypoints_2d: 2D-Punkte vom Wrapper
            scores_2d:    2D-Genauigkeiten vom Wrapper
            
        📤 Rückgabe:
            GLEICHE keypoints_2d und scores_2d (als Kopien)
        """
        return keypoints_2d.copy(), scores_2d.copy()  # 📋 Einfach kopieren!
    
    def convert_2d_to_3d(
        self,
        keypoints_2d: np.ndarray,  # 📍 Eingabe: 2D-Punkte
        scores_2d: np.ndarray,     # 🎯 Eingabe: 2D-Genauigkeiten
        image_size: Tuple[int, int] = None  # 📐 Optional: Bildgröße überschreiben
    ) -> Pose3DResult:
        """
        🪄 HAUPT-FUNKTION: WANDELT 2D IN 3D UM
        
        Ablauf der "Magie":
        1. 📋 Kopiere 2D-Punkte (KEINE Änderungen!)
        2. 🔍 Füge Tiefe (Z-Koordinate) hinzu
        3. 🚫 Filtere unerwünschte Punkte (Beine)
        4. 📦 Berechne 3D-Begrenzungsrahmen
        5. 📊 Berechne Gesamt-Genauigkeit
        
        🔧 Parameter:
            keypoints_2d: 2D-Körperpunkte [Personen, 133, 2]
            scores_2d:    2D-Genauigkeiten [Personen, 133]
            image_size:   (Breite, Höhe) des Originalbildes
            
        📤 Rückgabe:
            Pose3DResult mit allen 3D-Daten
        """
        # 🚫 Prüfen: Sind überhaupt Personen vorhanden?
        if len(keypoints_2d) == 0:
            return self._empty_result()  # 📭 Leeres Ergebnis zurückgeben
        
        # 📐 Bildgröße festlegen (Standard oder angegeben)
        w, h = image_size if image_size else (self.image_width, self.image_height)
        
        # ===============================================
        # 📋 SCHRITT 1: 2D-PUNKTE KOPIEREN (OHNE ÄNDERUNGEN!)
        # ===============================================
        kpts_2d, scores = self._copy_keypoints_from_2d(keypoints_2d, scores_2d)
        
        # ===============================================
        # 🔄 SCHRITT 2: 2D → 3D KONVERTIEREN ("Magie!")
        # ===============================================
        keypoints_3d_list = []  # 📋 Für 3D-Punkte jeder Person
        scores_3d_list = []     # 📋 Für 3D-Genauigkeiten
        
        # 👥 Für jede Person...
        for person_idx in range(len(kpts_2d)):
            kpts = kpts_2d[person_idx]  # 📍 2D-Punkte dieser Person
            scr = scores[person_idx]    # 🎯 Genauigkeiten dieser Person
            
            # 🪄 Geometrische Konvertierung: 2D → 3D
            kpts_3d = self._geometric_lift_2d_to_3d(kpts, scr, (h, w))
            
            keypoints_3d_list.append(kpts_3d)  # 💾 3D-Punkte speichern
            scores_3d_list.append(scr)         # 💾 Genauigkeiten behalten
        
        # 🔢 In numpy Arrays umwandeln
        keypoints_3d = np.array(keypoints_3d_list)
        scores_3d = np.array(scores_3d_list)
        
        # ===============================================
        # 🚫 SCHRITT 3: PUNKTE FILTERN (Beine entfernen)
        # ===============================================
        keypoints_3d, scores_3d = self._filter_keypoints(keypoints_3d, scores_3d)
        
        # ===============================================
        # 📦 SCHRITT 4: 3D-BEGRENZUNGSRAHMEN BERECHNEN
        # ===============================================
        bboxes_3d = self._calculate_3d_bboxes(keypoints_3d, scores_3d)
        
        # ===============================================
        # 📊 SCHRITT 5: GESAMT-GENAUIGKEIT BERECHNEN
        # ===============================================
        if np.any(scores_3d > 0):  # 🎯 Falls gültige Punkte existieren
            confidence = float(np.mean(scores_3d[scores_3d > 0]))
        else:
            confidence = 0.0  # 🚫 Keine gültigen Punkte
        
        # ===============================================
        # 🔍 SCHRITT 6: DEBUG-AUSGABE (Für Entwickler)
        # ===============================================
        if len(keypoints_3d) > 0:
            print(f"\n🔍 3D-Konvertierung - Wichtige Punkte prüfen:")
            print(f"   Punkt 9 (L-Handgelenk): Pos={keypoints_3d[0, 9]}, Score={scores_3d[0, 9]:.3f}")
            print(f"   Punkt 91 (L-Handwurzel): Pos={keypoints_3d[0, 91]}, Score={scores_3d[0, 91]:.3f}")
            print(f"   ✅ Sind sie gleich? {np.array_equal(keypoints_3d[0, 9], keypoints_3d[0, 91])}")
            
            print(f"\n   Punkt 10 (R-Handgelenk): Pos={keypoints_3d[0, 10]}, Score={scores_3d[0, 10]:.3f}")
            print(f"   Punkt 112 (R-Handwurzel): Pos={keypoints_3d[0, 112]}, Score={scores_3d[0, 112]:.3f}")
            print(f"   ✅ Sind sie gleich? {np.array_equal(keypoints_3d[0, 10], keypoints_3d[0, 112])}")
        
        # ===============================================
        # 📤 SCHRITT 7: ERGEBNIS ZURÜCKGEBEN
        # ===============================================
        return Pose3DResult(
            frame_idx=0,
            keypoints_3d=keypoints_3d,      # 🎯 3D-Punkte
            keypoints_2d=kpts_2d,           # 📍 Original 2D-Punkte
            scores_3d=scores_3d,            # 🎯 3D-Genauigkeiten
            bboxes_3d=bboxes_3d,            # 📦 3D-Begrenzungsrahmen
            num_persons=len(keypoints_2d),  # 👥 Personen-Anzahl
            method=self.lifting_method,     # 🔧 Verwendete Methode
            confidence=confidence           # 📊 Gesamt-Genauigkeit
        )
    
    def _geometric_lift_2d_to_3d(
        self, 
        keypoints_2d: np.ndarray, 
        scores: np.ndarray, 
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        🪄 GEOMETRISCHE 2D→3D KONVERTIERUNG
        
        EINFACH GESAGT:
        "Wie tief ist jeder Körperteil?" - Schätzung basierend auf Position.
        
        🎮 SO WIRD TIEFE GESCHÄTZT:
        - Nase, Augen, Ohren:      Etwas vorne (Z = 0.1)
        - Schultern, Ellbogen:     In der Mitte (Z = 0.0)
        - Hände, Finger:           Weit vorne (Z = 0.2-0.25)
        - Gesicht:                 Vorne (Z = 0.1)
        - Beine:                   Weiter hinten (aber werden ignoriert)
        
        🔧 Parameter:
            keypoints_2d: Einzelne Person, 133 Punkte, [X, Y]
            scores:       Genauigkeiten der Punkte
            image_shape:  (Höhe, Breite) des Bildes
            
        📤 Rückgabe:
            3D-Punkte [133 Punkte, X/Y/Z]
        """
        h, w = image_shape
        keypoints_3d = np.zeros((133, 3))  # 🆕 Leeres 3D-Array
        
        # 📋 1. X und Y KOORDINATEN KOPIEREN (aus 2D)
        keypoints_3d[:, :2] = keypoints_2d
        
        # 🔍 2. TIEFE (Z) FÜR JEDEN PUNKT SCHÄTZEN
        for i in range(133):
            if scores[i] > 0.3:  # 🎯 Nur bei ausreichender Genauigkeit
                if i == 0:  # 👃 Nase (vorne)
                    keypoints_3d[i, 2] = 0.1
                elif 1 <= i <= 4:  # 👀 Augen, 👂 Ohren (vorne)
                    keypoints_3d[i, 2] = 0.1
                elif 5 <= i <= 12:  # 💪 Schultern, Ellbogen, 🍑 Hüften (Mitte)
                    keypoints_3d[i, 2] = 0.0
                elif i in [9, 10]:  # ✋ Handgelenke (WICHTIG! vorne)
                    keypoints_3d[i, 2] = 0.2
                elif 91 <= i <= 111:  # ✋ Linke Hand (Finger, sehr vorne)
                    keypoints_3d[i, 2] = 0.25
                elif 112 <= i <= 132:  # ✋ Rechte Hand (Finger, sehr vorne)
                    keypoints_3d[i, 2] = 0.25
                elif 23 <= i <= 90:  # 😀 Gesicht (vorne)
                    keypoints_3d[i, 2] = 0.1
                else:  # 🦵 Andere Punkte (Beine, werden ignoriert)
                    keypoints_3d[i, 2] = 0.0
        
        return keypoints_3d
    
    def _filter_keypoints(
        self, 
        keypoints: np.ndarray, 
        scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        🚫 FILTERT IGNORIERTE KÖRPERPUNKTE (Z.B. BEINE)
        
        Setzt ignorierte Punkte auf (0,0,0) und Genauigkeit auf 0.
        Das macht sie praktisch "unsichtbar" in der 3D-Visualisierung.
        """
        kpts = keypoints.copy()  # 📋 Kopie (Original bleibt unverändert)
        scrs = scores.copy()     # 📋 Kopie
        
        # 🔄 Für jeden zu ignorierenden Punkt...
        for idx in self.ignore_keypoints:
            if idx < kpts.shape[1]:  # ✅ Prüfen ob Punkt existiert
                kpts[:, idx, :] = 0  # 🎯 Position auf (0,0,0)
                scrs[:, idx] = 0     # 🎯 Genauigkeit auf 0
        
        return kpts, scrs
    
    def _calculate_3d_bboxes(
        self, 
        keypoints_3d: np.ndarray, 
        scores_3d: np.ndarray
    ) -> np.ndarray:
        """
        📦 BERECHNET 3D-BEGRENZUNGSRAHMEN ("BOUNDING BOXES")
        
        Ein 3D-Rahmen ist wie eine imaginäre Schachtel um die Person.
        Enthält: [Mittelpunkt-X, Mittelpunkt-Y, Mittelpunkt-Z, 
                  Breite, Höhe, Tiefe, Genauigkeit]
        """
        bboxes = []
        
        # 👥 Für jede Person...
        for i in range(len(keypoints_3d)):
            # 🎯 Nur Punkte mit guter Genauigkeit berücksichtigen
            valid_mask = scores_3d[i] > 0.3
            valid_kpts = keypoints_3d[i][valid_mask]
            
            if len(valid_kpts) > 0:
                # 📐 Minimum und Maximum in allen 3 Dimensionen
                min_coords = np.min(valid_kpts, axis=0)  # 🔽 Kleinste X,Y,Z
                max_coords = np.max(valid_kpts, axis=0)  # 🔼 Größte X,Y,Z
                
                # 🎯 Mittelpunkt berechnen
                center = (min_coords + max_coords) / 2
                
                # 📏 Abmessungen (Breite, Höhe, Tiefe)
                dimensions = max_coords - min_coords
                
                # 🎯 Durchschnittliche Genauigkeit
                confidence = np.mean(scores_3d[i][valid_mask])
                
                # 📦 Alles zusammenfügen [X,Y,Z, Breite, Höhe, Tiefe, Genauigkeit]
                bboxes.append(np.concatenate([center, dimensions, [confidence]]))
            else:
                # 🚫 Keine gültigen Punkte: leeren Rahmen
                bboxes.append(np.zeros(7))
        
        return np.array(bboxes)  # 🔄 In numpy Array
    
    def _empty_result(self) -> Pose3DResult:
        """
        📭 GIBT EIN LEERES ERGEBNIS ZURÜCK
        
        Wird verwendet, wenn keine Personen gefunden wurden.
        """
        return Pose3DResult(
            frame_idx=0,
            keypoints_3d=np.empty((0, 133, 3)),  # 📭 0 Personen
            keypoints_2d=np.empty((0, 133, 2)),
            scores_3d=np.empty((0, 133)),
            bboxes_3d=np.empty((0, 7)),
            num_persons=0,
            method=self.lifting_method,
            confidence=0.0
        )
    
    def convert_2d_json_to_3d(
        self,
        input_json_path: Union[str, Path],  # 📁 Eingabe: 2D-JSON
        output_json_path: Union[str, Path],  # 📁 Ausgabe: 3D-JSON
        image_size: Tuple[int, int] = None   # 📐 Optional: Bildgröße
    ) -> List[Dict]:
        """
        📁 KONVERTIERT EINE GANZE 2D-JSON-DATEI ZU 3D
        
        Liest eine JSON-Datei mit 2D-Posen (von PoseEstimator2D)
        und erstellt eine neue JSON-Datei mit 3D-Posen.
        
        🔧 Parameter:
            input_json_path:  Pfad zur 2D-JSON-Datei
            output_json_path: Wo 3D-JSON gespeichert werden soll
            image_size:       Bildgröße für alle Frames
            
        📤 Rückgabe:
            Liste mit allen 3D-Ergebnissen
        """
        input_path = Path(input_json_path)
        if not input_path.exists():
            raise FileNotFoundError(f"❌ 2D JSON nicht gefunden: {input_path}")
        
        # 📖 2D-Daten laden
        with open(input_path, 'r') as f:
            data_2d = json.load(f)
        
        results_3d = []  # 📋 Für alle 3D-Ergebnisse
        
        print(f"🔄 Konvertiere {len(data_2d)} Bilder von 2D zu 3D...")
        
        # 🔄 Für jedes Bild in der 2D-Datei...
        for frame_data in data_2d:
            frame_idx = frame_data['frame']  # 🎞️ Bild-Nummer
            
            # 🎯 Linke Ansicht konvertieren (von Stereo-Kamera)
            left_3d = self._convert_single_view(frame_data['left'], image_size, frame_idx)
            
            # 🎯 Rechte Ansicht konvertieren
            right_3d = self._convert_single_view(frame_data['right'], image_size, frame_idx)
            
            # 🎯 Kombinierte Ansicht (hier einfach linke nehmen)
            frame_result = {
                "frame": frame_idx,
                "left_3d": left_3d,      # 👈 Linke Kamera-Ansicht
                "right_3d": right_3d,    # 👉 Rechte Kamera-Ansicht
                "combined_3d": left_3d   # 🎯 Beste kombinierte Ansicht
            }
            results_3d.append(frame_result)
            
            # 📊 Fortschritt anzeigen (alle 10 Bilder)
            if frame_idx % 10 == 0:
                print(f"  📊 Bild {frame_idx}/{len(data_2d)} konvertiert")
        
        # 💾 3D-Daten speichern
        with open(output_json_path, 'w') as f:
            json.dump(results_3d, f, indent=2)  # 📝 Schön formatiert
        
        print(f"✅ 3D Posen gespeichert: {output_json_path}")
        return results_3d
    
    def _convert_single_view(
        self, 
        view_data: Dict, 
        image_size: Tuple[int, int], 
        frame_idx: int
    ) -> Dict:
        """
        🔄 KONVERTIERT EINE EINZELNE KAMERA-ANSICHT
        
        Wird für linke und rechte Kamera-Ansicht aufgerufen.
        """
        # 📍 2D-Punkte extrahieren
        keypoints_2d = np.array(view_data['keypoints'])
        scores_2d = np.array(view_data['scores'])
        
        # 🪄 2D → 3D konvertieren
        pose_3d = self.convert_2d_to_3d(keypoints_2d, scores_2d, image_size)
        
        # 📦 Ergebnis als Dictionary zurückgeben
        return {
            "keypoints_3d": pose_3d.keypoints_3d.tolist(),  # 🔄 numpy → Liste
            "scores_3d": pose_3d.scores_3d.tolist(),
            "bboxes_3d": pose_3d.bboxes_3d.tolist(),
            "num_persons": pose_3d.num_persons,
            "method": pose_3d.method,
            "confidence": pose_3d.confidence
        }

# ===============================================
# ⚡ BEQUEMLICHKEITSFUNKTION (Für Import)
# ===============================================
# Diese Funktion wird vom Test-Skript importiert!

def convert_2d_poses_to_3d(
    input_json_path: Union[str, Path],  # 📁 Eingabe: 2D-Posen
    output_json_path: Union[str, Path],  # 📁 Ausgabe: 3D-Posen
    lifting_method: str = 'geometric',   # 🔧 Methode (immer geometric)
    mmpose_model: str = 'human3d',       # 🤖 KI-Modell (optional)
    device: str = 'cpu'                  # 💻 Hardware
) -> List[Dict]:
    """
    ⚡ SCHNELLE FUNKTION FÜR 2D→3D KONVERTIERUNG
    
    Diese Funktion wird vom Test-Skript aufgerufen.
    Einfachste Nutzung:
        convert_2d_poses_to_3d("2d_poses.json", "3d_poses.json")
    
    🔧 Parameter:
        input_json_path:  2D-JSON-Datei von PoseEstimator2D
        output_json_path: Wo 3D-JSON gespeichert werden soll
        lifting_method:   Konvertierungs-Methode
        mmpose_model:     KI-Modell für fortgeschrittene Methoden
        device:           Hardware (cpu, cuda)
        
    📤 Rückgabe:
        Liste mit allen 3D-Ergebnissen
    """
    # 🤖 Konverter erstellen
    converter = Pose3DConverter(
        lifting_method=lifting_method,
        mmpose_model=mmpose_model,
        device=device
    )
    # 🔄 Konvertierung durchführen
    return converter.convert_2d_json_to_3d(input_json_path, output_json_path)

# ===============================================
# 🚀 START: WENN DAS PROGRAMM DIREKT GESTARTET WIRD
# ===============================================
if __name__ == "__main__":
    print("=" * 60)
    print("🔄 Pose3DConverter - Test Script")
    print("=" * 60)
    print("📝 Testet die 2D→3D Konvertierung")
    print("")
    
    # 🔍 Test-Dateien
    test_input = "poses_2d_filtered.json"  # 📁 Erwartet: 2D-Posen
    test_output = "poses_3d_test.json"     # 📁 Wird erstellt: 3D-Posen
    
    if Path(test_input).exists():
        print(f"✅ Testdatei gefunden: {test_input}")
        print("🔄 Starte Konvertierung...")
        
        # 🪄 Konvertierung durchführen
        results = convert_2d_poses_to_3d(test_input, test_output)
        
        print(f"✅ Erfolgreich konvertiert!")
        print(f"   📊 {len(results)} Bilder verarbeitet")
        print(f"   📁 Ergebnis: {test_output}")
        
    else:
        print(f"⚠️  Testdatei {test_input} nicht gefunden")
        print("")
        print("ℹ️  So nutzt du es:")
        print("   1. Erstelle zuerst 2D-Posen mit PoseEstimator2D")
        print("   2. Speichere sie als JSON (z.B. 'poses_2d.json')")
        print("   3. Konvertiere zu 3D:")
        print("      convert_2d_poses_to_3d('poses_2d.json', 'poses_3d.json')")
        print("")
        print("💡 TIPP: Die 3D-Datei kann mit dem 3D Visualizer angezeigt werden!")