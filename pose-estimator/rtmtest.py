"""
🔄 Aktualisiertes Test-Skript für die komplette 2D→3D Pipeline

EINFACHE ERKLÄRUNG:
Dies ist unser "Hauptprogramm", das alle Teile zusammenfügt wie ein Rezept:
1. 🖼️  Nimm ein Bild
2. 🤖  Finde Menschen darin (2D)
3. 🔄  Mache es 3D-fähig
4. 🎨  Zeichne und zeige das Ergebnis

So wie ein Küchenrezept: Zutaten → Schritte → fertiges Gericht!
"""

# ===============================================
# 📦 IMPORTIEREN DER EIGENEN MODULE
# ===============================================
# Hier laden wir unsere selbstgeschriebenen "Werkzeugkästen"
from pose_estimator_2d import PoseEstimator2D, DEFAULT_IGNORE_KEYPOINTS, filter_keypoints
from pose_estimator_3d import convert_2d_poses_to_3d  # 🔄 Neues 3D-Modul
from pose_3d_visualizer_updated import plot_3d_pose_from_json, plot_multiple_views  # 🎨 Verbesserter Visualizer
import json  # 📄 Zum Lesen/Schreiben von Daten-Dateien
import numpy as np  # 🔢 Für Mathe-Berechnungen

# ===============================================
# 🚀 WILLKOMMENSNACHRICHT
# ===============================================
print("=" * 80)
print("🎯 TEST: Komplette 2D → 3D Körperpositions-Pipeline")
print("=" * 80)
print("📋 Dieses Programm führt alle Schritte automatisch durch:")
print("   1. 🖼️  Finde Menschen im Bild (2D)")
print("   2. 🔄  Mache 3D-Modelle daraus")
print("   3. 🎨  Zeichne und zeige die Ergebnisse")
print("=" * 80)

# ===============================================
# 🖼️ SCHRITT 0: BILD AUSWÄHLEN
# ===============================================
print("\n📸 SCHRITT 0: Wähle ein Testbild aus")

# 🖼️ Liste der verfügbaren Testbilder (nur eins aktiv lassen!)
file = "V.png"        # 🟦 Beispielbild 1 - Person mit V-Hand
#file = "hocke.jpg"    # 🟦 Beispielbild 2 - Person in der Hocke
#file = "mensch.jpg"   # 🟦 Beispielbild 3 - Standard-Person
#file = "merkel.jpg"   # 🟦 Beispielbild 4 - Angela Merkel
#file = "merz.jpg"     # 🟦 Beispielbild 5 - Friedrich Merz

print(f"✅ Ausgewähltes Bild: {file}")
print("   ℹ️  Um ein anderes Bild zu testen, ändere Zeile 31-35")

# ===============================================
# 🎯 SCHRITT 1: 2D-KÖRPERPOSITIONEN FINDEN
# ===============================================
print("\n" + "="*80)
print("📸 SCHRITT 1: Finde Menschen im Bild (2D)")
print("="*80)
print("🤖 Die KI schaut sich das Bild an und sagt:")
print('   "Hier ist ein Mensch, hier sind seine Körperteile!"')

# 🚫 Welche Körperteile werden ignoriert? (Beine/Füße)
print(f"🚫 Ignoriere Punkte: {DEFAULT_IGNORE_KEYPOINTS} (Beine/Füße)")
print("   Warum? Manchmal wollen wir uns nur auf den Oberkörper konzentrieren.")

# 🤖 Erstelle den 2D-"Body-Detektor"
estimator_2d = PoseEstimator2D(kpt_threshold=0.9)  # 🎯 90% Mindest-Genauigkeit
print("✅ 2D-Pose-Estimator erstellt (sehr hohe Genauigkeit)")

# 🎯 Bild analysieren (KI-Magie passiert hier!)
result_2d = estimator_2d.process_image(file)
print(f"👤 Gefunden: {result_2d.num_persons} Person(en) im Bild")

# 🔍 Zeige was die KI gemacht hat (für Entwickler)
if result_2d.num_persons > 0:
    print("\n🔍 KI-Ersetzungen (automatisch korrigiert):")
    print(f"   👃 Nase (Punkt 0):  Wurde durch Punkt 53 (Gesichts-Nase) ersetzt")
    print(f"   ✋ Linkes Handgelenk: Punkt 9 → Punkt 91 (genauer)")
    print(f"   ✋ Rechtes Handgelenk: Punkt 10 → Punkt 112 (genauer)")

# 🚫 Filtere Beine heraus (machen sie "unsichtbar")
print(f"\n🚫 Filtere Beine/Füße (Punkte 13-22) aus...")
result_2d.keypoints, result_2d.scores = filter_keypoints(
    result_2d.keypoints,      # 📍 Original-Punkte
    result_2d.scores,         # 🎯 Original-Genauigkeiten
    DEFAULT_IGNORE_KEYPOINTS  # 🚫 Welche Punkte ignorieren?
)

# ✅ Überprüfe ob Filterung funktioniert hat
if result_2d.num_persons > 0:
    filtered_scores = result_2d.scores[0][DEFAULT_IGNORE_KEYPOINTS]
    filtered_count = np.sum(filtered_scores == 0)  # 🎯 Wie viele sind jetzt 0?
    print(f"✅ {filtered_count} Bein-Punkte wurden 'unsichtbar' gemacht")

# ===============================================
# 💾 SCHRITT 2: 2D-DATEN SPEICHERN (für 3D)
# ===============================================
print("\n💾 Speichere 2D-Daten für 3D-Konvertierung...")

# 📋 Erstelle spezielles Format (linke & rechte "Kamera")
results_2d_list = [{
    "frame": 0,  # 🎞️ Bild-Nummer (0 für einzelnes Bild)
    "left": {    # 👈 Linke "Kamera"-Ansicht
        "num_persons": result_2d.num_persons,
        "keypoints": result_2d.keypoints.tolist(),  # 🔄 In Liste umwandeln
        "scores": result_2d.scores.tolist(),
        "bboxes": result_2d.bboxes.tolist()
    },
    "right": {   # 👉 Rechte "Kamera"-Ansicht (bei Einzelbild: Kopie)
        "num_persons": result_2d.num_persons,
        "keypoints": result_2d.keypoints.tolist(),
        "scores": result_2d.scores.tolist(),
        "bboxes": result_2d.bboxes.tolist()
    }
}]

# 📁 In Datei speichern
with open("poses_2d_filtered.json", "w") as f:
    json.dump(results_2d_list, f, indent=2)  # 📝 Schön formatiert
print("✅ Gespeichert: poses_2d_filtered.json")

# ===============================================
# 🖍️ SCHRITT 3: 2D-BILD ANNOTIEREN (zeichnen)
# ===============================================
print("\n" + "="*80)
print("🖍️ SCHRITT 3: Zeichne Körperlinien auf das Originalbild")
print("="*80)
print("🎨 Jetzt malen wir die gefundenen Menschen ein:")
print("   - 🟩 Grüne Linien für Körperverbindungen")
print("   - 🔴 Rote Punkte für Körperpositionen")
print("   - 🚫 KEINE Bein-Linien (weil gefiltert)")

# 🖍️ Bild mit Körperlinien zeichnen
bild = estimator_2d.process_image_with_annotation(
    image_path=file,                     # 🖼️ Welches Bild?
    output_path="image_annotated_filtered.png",  # 💾 Wo speichern?
    ignore_keypoints=DEFAULT_IGNORE_KEYPOINTS,   # 🚫 Welche Punkte ignorieren?
    draw_bbox=True,                      # 🟩 Grüne Rahmen zeichnen?
    draw_keypoints=True,                 # 🔴 Punkte zeichnen?
    keypoint_threshold=0.3               # 🎯 Mindest-Genauigkeit
)
print("💾 Gespeichert: image_annotated_filtered.png")
print("   Öffne die Datei um das Ergebnis zu sehen!")

# ===============================================
# 🔄 SCHRITT 4: 2D → 3D KONVERTIEREN (Magie!)
# ===============================================
print("\n" + "="*80)
print("🪄 SCHRITT 4: Mache aus dem 2D-Bild ein 3D-Modell")
print("="*80)
print("🎮 Stell dir vor:")
print("   🖼️  2D-Foto → 🪄 Magie → 🎯 3D-Figur")
print("")
print("⚙️  Verwendete Methode: geometric (mathematische Schätzung)")
print("   Alternative: 'mmpose' (fortgeschrittene KI, wenn installiert)")

# 🔄 Die eigentliche 2D→3D Konvertierung
results_3d = convert_2d_poses_to_3d(
    "poses_2d_filtered.json",        # 📁 Eingabe: Unsere 2D-Daten
    "poses_3d_filtered.json",        # 📁 Ausgabe: Werden 3D-Daten
    lifting_method='ai'       # 🔧 Methode: geometric (Mathe) oder ai (KI)
)
print("✅ 3D-Modelle erfolgreich erstellt!")

# 📊 Zeige Statistiken über die 3D-Daten
if len(results_3d) > 0:
    frame0 = results_3d[0]['combined_3d']  # 🎯 Beste 3D-Ansicht
    print(f"\n📊 3D-Ergebnis-Statistiken:")
    print(f"   👥 Personen: {frame0['num_persons']}")
    print(f"   🔧 Methode: {frame0['method']}")
    print(f"   🎯 Genauigkeit: {frame0['confidence']:.1%}")
    
    # 🔍 Zeige Tiefen-Informationen
    kpts_3d = np.array(frame0['keypoints_3d'])
    if len(kpts_3d) > 0 and frame0['num_persons'] > 0:
        z_coords = kpts_3d[0, :, 2]  # 📏 Z-Koordinaten (Tiefe)
        z_valid = z_coords[z_coords != 0]  # 🚫 Filtere 0-Werte
        if len(z_valid) > 0:
            print(f"   📏 Tiefenbereich: {z_valid.min():.2f} bis {z_valid.max():.2f}")
            print(f"   📐 Durchschnittliche Tiefe: {z_valid.mean():.2f}")

# ===============================================
# 🎨 SCHRITT 5: 3D-MODELLE VISUALISIEREN
# ===============================================
print("\n" + "="*80)
print("🎨 SCHRITT 5: Zeige die 3D-Modelle an")
print("="*80)
print("🖥️  Jetzt kommt der coole Teil: Interaktive 3D-Grafik!")
print("")
print("🎯 Was gezeichnet wird:")
print("   - 👤 Korrekte Schulter-Linien (Nase → Schultern)")
print("   - ✋ Handgelenke an richtiger Position (Punkte 91 & 112)")
print("   - 🚫 KEINE Bein-Verbindungen")
print("   - 🔍 Tiefe 5x vergrößert für bessere Sichtbarkeit")

# 🎨 3D-Visualisierung erstellen
z_scale = 5.0  # 🔍 Tiefe verstärken (für bessere 3D-Wirkung)
print(f"\n⚙️  Einstellungen:")
print(f"   - Tiefen-Verstärkung: {z_scale}x")
print(f"   - Hände anzeigen: Ja")
print(f"   - Gesicht anzeigen: Ja")

# 🖼️ Erstelle und zeige 3D-Grafik
plot_3d_pose_from_json(
    "poses_3d_filtered.json",  # 📁 Unsere 3D-Daten
    frame_idx=0,               # 🎞️ Erstes Bild
    view='combined_3d',        # 🎯 Beste Ansicht
    output_path="image_3d_filtered.png",  # 💾 Speichere Bild
    z_scale=z_scale,           # 🔍 Tiefen-Verstärkung
    show_plot=True,            # 👀 Sofort anzeigen?
    show_hands=True,           # ✋ Hände zeigen?
    show_face=True             # 😀 Gesicht zeigen?
)
print("💾 Gespeichert: image_3d_filtered.png")

# ===============================================
# 🎉 ZUSAMMENFASSUNG: WAS WURDE GEMACHT?
# ===============================================
print("\n" + "="*80)
print("🎉 FERTIG! Die komplette Pipeline wurde erfolgreich durchlaufen!")
print("="*80)

print("\n📁 ALLE ERSTELLTEN DATEIEN:")
print("   1. 📄 poses_2d_filtered.json        - 2D-Daten (ohne Beine)")
print("   2. 🖼️  image_annotated_filtered.png  - Bild mit Körperlinien")
print("   3. 📄 poses_3d_filtered.json        - 3D-Modelle")
print("   4. 🎨 image_3d_filtered.png         - 3D-Visualisierung")

print("\n✅ ALLE KORREKTUREN WURDEN DURCHGEFÜHRT:")
print("   👃 Nase:             Punkt 0 → Punkt 53 (genauer)")
print("   ✋ Linkes Handgelenk: Punkt 9 → Punkt 91 (genauer)")
print("   ✋ Rechtes Handgelenk: Punkt 10 → Punkt 112 (genauer)")
print("   🚫 Beine/Füße:       Punkte 13-22 gefiltert (unsichtbar)")
print("   🔗 Schultern:        Korrekte Verbindungen (Nase→Schultern)")

print("\n🔧 EINSTELLUNGEN DIESES LAUFS:")
print(f"   - Tiefen-Verstärkung: {z_scale}x")
print("   - Mindest-Genauigkeit 2D: 90%")
print("   - 3D-Methode: geometric (mathematisch)")
print("   - Hände angezeigt: Ja")
print("   - Gesicht angezeigt: Ja")