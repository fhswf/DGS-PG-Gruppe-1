"""
🔄 Aktualisiertes Test-Skript für die komplette 2D→3D Pipeline

EINFACHE ERKLÄRUNG:
Dies ist unser "Hauptprogramm", das alle Teile zusammenfügt wie ein Rezept:
1. 🖼️  Nimm ein Bild
2. 🤖  Finde Menschen darin (2D)
3. 🔄  Mache es 3D-fähig mit TRAINIERTEM MODELL
4. 🎨  Zeichne und zeige das Ergebnis

ZUSAMMENHANG:
Dieses Skript verbindet alle Komponenten unserer 3D-Pose-Pipeline:
- 2D Pose Estimation (OpenPose/MediaPipe)
- 3D Pose Lifting (trainiertes neuronales Netz)
- Visualisierung der Ergebnisse
"""

# ===============================================
# 📦 IMPORTIEREN DER EIGENEN MODULE
# ===============================================
from pose_estimator_2d import PoseEstimator2D, DEFAULT_IGNORE_KEYPOINTS, filter_keypoints
from pose_estimator_3d import Pose3DConverter  # 🔄 Neue Converter-Klasse
# from pose_3d_visualizer_updated import plot_3d_pose_from_json, plot_multiple_views  # Optional
import json
import numpy as np

# ===============================================
# 🚀 WILLKOMMENSNACHRICHT
# ===============================================
print("=" * 80)
print("🎯 TEST: Komplette 2D → 3D Körperpositions-Pipeline mit AI-Modell")
print("=" * 80)
print("📋 Dieses Programm führt alle Schritte automatisch durch:")
print("   1. 🖼️  Finde Menschen im Bild (2D)")
print("   2. 🤖 Nutze trainiertes MLP-Modell für 3D-Konvertierung")
print("   3. 🎨 Speichere und zeige die Ergebnisse")
print("=" * 80)

# ===============================================
# 🖼️ SCHRITT 0: BILD AUSWÄHLEN
# ===============================================
print("\n📸 SCHRITT 0: Wähle ein Testbild aus")

# 🖼️ TESTBILDER - Nur eine Zeile sollte aktiv sein (kein # davor):
file = "V.png"        # 🟦 Beispielbild 1 - Person mit V-Hand-Zeichen
# file = "hocke.jpg"    # 🟦 Beispielbild 2 - Person in der Hocke
# file = "mensch.jpg"   # 🟦 Beispielbild 3 - Standard-Person

print(f"✅ Ausgewähltes Bild: {file}")

# ===============================================
# 🎯 SCHRITT 1: 2D-KÖRPERPOSITIONEN FINDEN (OPENPOSE/MEDIAPIPE)
# ===============================================
print("\n" + "="*80)
print("📸 SCHRITT 1: Finde Menschen im Bild (2D Pose Estimation)")
print("="*80)

print(f"🚫 Ignoriere Punkte: {DEFAULT_IGNORE_KEYPOINTS} (Beine/Füße)")
"""
WARUM WIR BEINE IGNORIEREN:
- Bein-Posen sind oft ungenau in 2D-Detektoren
- Sie machen die 3D-Rekonstruktion instabil
- Fokus auf Oberkörper für bessere Ergebnisse
- Punkte 13-22: Knie, Knöchel, Füße
"""

# 🤖 Erstelle den 2D-"Body-Detektor" (OpenPose oder MediaPipe)
estimator_2d = PoseEstimator2D(kpt_threshold=0.9)
"""
KONFIGURATION:
- kpt_threshold=0.9: Nur Keypoints mit 90%+ Confidence werden akzeptiert
- Höherer Threshold = präzisere aber weniger Keypoints
- Niedrigerer Threshold = mehr Keypoints aber möglicherweise ungenauer
"""
print("✅ 2D-Pose-Estimator erstellt (sehr hohe Genauigkeit)")

# 🎯 Bild analysieren - Hier passiert die eigentliche 2D-Erkennung
result_2d = estimator_2d.process_image(file)
"""
WAS PASSIERT HIER:
1. Bild wird geladen und vorverarbeitet
2. Neuronales Netz findet Personen und ihre Gelenke
3. Keypoints werden in Koordinaten umgewandelt
4. Confidence-Scores werden berechnet
"""
print(f"👤 Gefunden: {result_2d.num_persons} Person(en) im Bild")

# 🔍 Zeige Ersetzungen (falls welche durchgeführt wurden)
if result_2d.num_persons > 0:
    print("\n🔍 KI-Ersetzungen (automatisch korrigiert):")
    print(f"   👃 Nase (Punkt 0):  Punkt 53 (Gesichts-Nase)")
    print(f"   ✋ Linkes Handgelenk: Punkt 9 → Punkt 91")
    print(f"   ✋ Rechtes Handgelenk: Punkt 10 → Punkt 112")
    """
    WARUM ERSETZUNGEN:
    - Standard-Body-Modelle haben oft Probleme mit bestimmten Keypoints
    - MediaPipe/Gesichts-Modelle sind für Gesichts-Posen besser
    - Hand-Modelle sind für Hand-Posen besser
    - Kombination verschiedener Modelle für bessere Ergebnisse
    """

# 🚫 Filtere Beine heraus - Entferne unzuverlässige Keypoints
print(f"\n🚫 Filtere Beine/Füße (Punkte 13-22) aus...")
result_2d.keypoints, result_2d.scores = filter_keypoints(
    result_2d.keypoints,      # 2D-Koordinaten aller Personen
    result_2d.scores,         # Confidence-Scores
    DEFAULT_IGNORE_KEYPOINTS  # Welche Punkte zu filtern sind
)
"""
WAS filter_keypoints MACHT:
- Setzt die Koordinaten der ignorierten Keypoints auf 0
- Setzt ihre Confidence-Scores auf 0
- Erhaltene Keypoints bleiben unverändert
"""

# ✅ Überprüfung der Filterung
if result_2d.num_persons > 0:
    # Extrahiere Scores der gefilterten Keypoints
    filtered_scores = result_2d.scores[0][DEFAULT_IGNORE_KEYPOINTS]
    # Zähle wie viele auf 0 gesetzt wurden
    filtered_count = np.sum(filtered_scores == 0)
    print(f"✅ {filtered_count} Bein-Punkte wurden 'unsichtbar' gemacht")

# ===============================================
# 💾 SCHRITT 2: 2D-DATEN SPEICHERN (JSON-FORMAT)
# ===============================================
print("\n💾 Speichere 2D-Daten für 3D-Konvertierung...")

"""
JSON-STRUKTUR FÜR 3D-KONVERTIERUNG:
[
    {
        "frame": 0,                    # Frame-Nummer
        "left": {                      # Linke Kamera-Ansicht
            "num_persons": 1,         # Anzahl Personen
            "keypoints": [[x,y], ...], # 133×2 Koordinaten
            "scores": [0.9, ...],     # 133 Confidence-Werte
            "bboxes": [...]            # Bounding Boxes
        },
        "right": {...}                 # Rechte Kamera-Ansicht (gleiche Daten)
    }
]
"""
results_2d_list = [{
    "frame": 0,
    "left": {
        "num_persons": result_2d.num_persons,
        "keypoints": result_2d.keypoints.tolist(),  # np.array → Liste
        "scores": result_2d.scores.tolist(),
        "bboxes": result_2d.bboxes.tolist()
    },
    "right": {  # Gleiche Daten für rechte Ansicht (Stereo-Kamera-Simulation)
        "num_persons": result_2d.num_persons,
        "keypoints": result_2d.keypoints.tolist(),
        "scores": result_2d.scores.tolist(),
        "bboxes": result_2d.bboxes.tolist()
    }
}]

# Speichere JSON-Datei
with open("poses_2d_filtered.json", "w") as f:
    json.dump(results_2d_list, f, indent=2)  # indent=2 für lesbares Format
print("✅ Gespeichert: poses_2d_filtered.json")

# ===============================================
# 🖍️ SCHRITT 3: 2D-BILD ANNOTIEREN (VISUALISIERUNG)
# ===============================================
print("\n" + "="*80)
print("🖍️ SCHRITT 3: Zeichne Körperlinien auf das Originalbild")
print("="*80)

"""
WAS DIE ANNOTATION MACHT:
- Zeichnet Gelenke als Punkte
- Verbindet Gelenke mit Linien (Skelett)
- Zeichnet Bounding Boxes um Personen
- Ignoriert gefilterte Keypoints (Beine)
"""
bild = estimator_2d.process_image_with_annotation(
    image_path=file,                    # Eingabebild
    output_path="image_annotated_filtered.png",  # Ausgabedatei
    ignore_keypoints=DEFAULT_IGNORE_KEYPOINTS,  # Nicht zeichnen
    draw_bbox=True,                     # Bounding Box zeichnen
    draw_keypoints=True,                # Keypoints zeichnen
    keypoint_threshold=0.3              # Min. Confidence für Anzeige
)
print("💾 Gespeichert: image_annotated_filtered.png")

# ===============================================
# 🔄 SCHRITT 4: 2D → 3D MIT TRAINIERTEM MODELL (KERN-SCHRITT)
# ===============================================
print("\n" + "="*80)
print("🪄 SCHRITT 4: 2D → 3D mit trainiertem MLP-Modell")
print("="*80)
print("🤖 HIER PASSIERT DIE 'MAGIE':")
print("   - Das trainierte neuronale Netz wird geladen")
print("   - Es wandelt 2D-Koordinaten in 3D-Koordinaten um")
print("   - Tiefeninformationen (Z-Achse) werden berechnet")
print("   - Ergebnisse werden gespeichert")
print("="*80)

# 🤖 Erstelle 3D-Converter mit trainiertem Modell
converter_3d = Pose3DConverter(
    model_path='lifting2DTo3D.pth',  # 📁 Dein trainiertes Modell
    lifting_method='mlp',             # 🤖 Nutze das AI-Modell!
    device='cuda',                    # 💻 GPU falls verfügbar, sonst 'cpu'
    ignore_keypoints=DEFAULT_IGNORE_KEYPOINTS  # Gleiche Filter wie bei 2D
)
"""
PARAMETER-ERKLÄRUNG:
- model_path: Pfad zur .pth Datei mit den trainierten Gewichten
- lifting_method: 'mlp' für neuronales Netz, 'geometric' für Fallback
- device: 'cuda' für NVIDIA GPU (schneller), 'cpu' für CPU (langsamer)
- ignore_keypoints: Welche Keypoints in der Ausgabe ignoriert werden sollen
"""

print(f"✅ 3D-Converter initialisiert")
print(f"   Methode: {converter_3d.lifting_method}")
print(f"   Device: {converter_3d.device}")

# 🔄 Konvertiere die JSON-Datei von 2D zu 3D
print("\n🔄 Starte Konvertierung...")
results_3d = converter_3d.convert_2d_json_to_3d(
    input_json_path="poses_2d_filtered.json",   # Eingabe: 2D-Posen
    output_json_path="poses_3d_mlp.json",       # Ausgabe: 3D-Posen
    image_size=(1920, 1080)  # Passe an deine Bildgröße an!
)
"""
WAS convert_2d_json_to_3d MACHT:
1. Lädt die 2D-JSON-Datei
2. Für jeden Frame und jede Person:
   a. Extrahiert 2D-Koordinaten und Scores
   b. Führt Forward-Pass durch das neuronale Netz durch
   c. Berechnet 3D-Koordinaten (x, y, z)
   d. Speichert Ergebnisse mit Konfidenzen und Metadaten
3. Speichert alles in einer neuen JSON-Datei
"""

print("✅ 3D-Modelle erfolgreich erstellt mit MLP!")

# 📊 Zeige detaillierte Statistiken der Ergebnisse
if len(results_3d) > 0:
    frame0 = results_3d[0]['combined_3d']  # Extrahiere Daten von Frame 0
    print(f"\n📊 3D-Ergebnis-Statistiken:")
    print(f"   👥 Personen: {frame0['num_persons']}")
    print(f"   🔧 Methode: {frame0['method']}")
    print(f"   🎯 Genauigkeit: {frame0['confidence']:.1%}")
    
    # 🔍 Analysiere Tiefeninformationen (Z-Koordinaten)
    kpts_3d = np.array(frame0['keypoints_3d'])  # Konvertiere zu numpy array
    if len(kpts_3d) > 0 and frame0['num_persons'] > 0:
        z_coords = kpts_3d[0, :, 2]  # Extrahiere Z-Werte der ersten Person
        z_valid = z_coords[z_coords != 0]  # Ignoriere 0-Werte (gefilterte/ungültige)
        
        if len(z_valid) > 0:
            print(f"   📏 Tiefenbereich: {z_valid.min():.2f} bis {z_valid.max():.2f}")
            print(f"   📐 Durchschnitt Z: {z_valid.mean():.2f}")
            """
            INTERPRETATION DER Z-WERTE:
            - Positive Z: Vorwärts (weg von der Kamera)
            - Negative Z: Rückwärts (zur Kamera hin)
            - Größere Werte: Weiter entfernt
            - Kleinere Werte: Näher an der Kamera
            - 0: Keine Tiefeninformation verfügbar
            """

# ===============================================
# 🎨 SCHRITT 5: 3D DIREKT VISUALISIEREN (Optional)
# ===============================================
print("\n" + "="*80)
print("🎨 SCHRITT 5: Visualisierung der 3D-Posen")
print("="*80)

# Versuche den 3D-Visualizer zu importieren (optional)
try:
    from pose_3d_visualizer_updated import plot_3d_pose_from_json
    
    print("🖥️  Erstelle 3D-Visualisierung...")
    plot_3d_pose_from_json(
        "poses_3d_mlp.json",   # Eingabe: 3D-Posen JSON
        frame_idx=0,           # Welcher Frame visualisiert werden soll
        view='combined_3d',    # Welche Ansicht (left_3d, right_3d, combined_3d)
        output_path="image_3d_mlp.png",  # Wo gespeichert wird
        z_scale=5.0,           # Skalierung der Z-Achse für bessere Darstellung
        show_plot=True,        # Plot direkt anzeigen
        show_hands=True,       # Hände anzeigen
        show_face=True         # Gesicht anzeigen
    )
    print("💾 Gespeichert: image_3d_mlp.png")
except ImportError:
    print("⚠️  Visualizer nicht gefunden - überspringe Visualisierung")
    print("   3D-Daten wurden aber gespeichert in: poses_3d_mlp.json")
    print("   Du kannst sie mit anderen Tools visualisieren (z.B. Blender, Unity)")

# ===============================================
# 🎉 ZUSAMMENFASSUNG DER ERGEBNISSE
# ===============================================
print("\n" + "="*80)
print("🎉 FERTIG! Pipeline mit trainiertem MLP-Modell erfolgreich!")
print("="*80)

print("\n📁 ERSTELLTE DATEIEN:")
print("   1. 📄 poses_2d_filtered.json        - Rohdaten der 2D-Erkennung")
print("   2. 🖼️  image_annotated_filtered.png  - Bild mit 2D-Skelett")
print("   3. 📄 poses_3d_mlp.json             - 3D-Koordinaten (neuronales Netz)")
print("   4. 🎨 image_3d_mlp.png              - 3D-Visualisierung (falls verfügbar)")

print("\n🤖 VERWENDETES MODELL:")
print("   - Typ: Multi-Layer Perceptron (MLP)")
print("   - Architektur: 6 Schichten mit Residual Connections")
print("   - Datei: lifting2DTo3D.pth (trainierte Gewichte)")
print("   - Trainiert auf: H3WB Dataset (133 Keypoints)")
print("   - Eingabe: 266 Werte (133×2), Ausgabe: 399 Werte (133×3)")

print("\n💡 NÄCHSTE SCHRITTE / PROBLEMLÖSUNG:")
print("   1. Falls Modell fehlt: Trainiere es mit train_new_model()")
print("   2. Falls ungenau: Trainiere mit mehr Daten oder mehr Epochen")
print("   3. Teste verschiedene Bilder für verschiedene Posen")
print("   4. Vergleiche 'geometric' vs 'mlp' Methode für Qualitätscheck")

# ===============================================
# 📊 ALTERNATIVE: DIREKTER VERGLEICH BEIDER METHODEN
# ===============================================
print("\n" + "="*80)
print("🔬 BONUS: Vergleich Geometric vs MLP Methode")
print("="*80)
print("🤔 WARUM VERGLEICHEN?")
print("   - Geometric: Einfache Heuristik (Fallback)")
print("   - MLP: Neuronales Netz (trainiert auf echten Daten)")
print("   - Vergleich zeigt Qualitätsunterschiede")
print("="*80)

# Geometric Methode zum Vergleich (Fallback ohne Training)
converter_geometric = Pose3DConverter(
    lifting_method='geometric',  # Einfache Heuristik statt MLP
    ignore_keypoints=DEFAULT_IGNORE_KEYPOINTS
)

# Konvertiere mit geometrischer Methode
results_3d_geometric = converter_geometric.convert_2d_json_to_3d(
    input_json_path="poses_2d_filtered.json",
    output_json_path="poses_3d_geometric.json",
    image_size=(1920, 1080)
)

print("\n📊 VERGLEICH DER KONFIDENZEN:")
# Extrahiere Konfidenzwerte beider Methoden
mlp_conf = results_3d[0]['combined_3d']['confidence']
geo_conf = results_3d_geometric[0]['combined_3d']['confidence']

print(f"   🤖 MLP Confidence:       {mlp_conf:.1%}")
print(f"   📐 Geometric Confidence: {geo_conf:.1%}")
print(f"   {'✅ MLP ist besser!' if mlp_conf > geo_conf else '⚠️  Geometric ist besser - Modell nachtrainieren?'}")

"""
INTERPRETATION:
- Confidence < 50%: Schlechte Erkennung, Modell benötigt mehr Training
- Confidence 50-70%: Akzeptable Ergebnisse
- Confidence 70-90%: Gute Ergebnisse
- Confidence > 90%: Exzellente Ergebnisse
"""

print("\n✅ Alle Tests abgeschlossen!")
print("\n🔧 TECHNISCHE HINWEISE:")
print("   - 2D-Erkennung: OpenPose/MediaPipe (CPU/GPU)")
print("   - 3D-Lifting: Eigenes neuronales Netz (PyTorch)")
print("   - Datenformat: JSON für einfache Weiterverarbeitung")
print("   - Visualisierung: Matplotlib/Open3D (optional)")