from pose_estimator_2d import PoseEstimator2D, DEFAULT_IGNORE_KEYPOINTS, filter_keypoints
from pose_estimator_3d import convert_2d_poses_to_3d
from pose_3d_visualizer import plot_3d_pose_from_json, plot_multiple_views
import json

print("=== Testing 2D to 3D Pose Conversion with Filtering ===")

file = "V.png"
#file = "hocke.jpg"
#file = "mensch.jpg"

# 1. Zuerst 2D-Posen mit Filterung berechnen
print("\n1. Calculating 2D poses with filtering...")
print(f"Ignoring keypoints: {DEFAULT_IGNORE_KEYPOINTS} (Füße/Zehen)")

estimator_2d = PoseEstimator2D(kpt_threshold=0.9)

# Für ein Bild verwenden wir process_image statt process_side_by_side_video
result_2d = estimator_2d.process_image(file)

# ===== NEU: Keypoints filtern BEVOR wir das JSON erstellen =====
print(f"Original: Detected {result_2d.num_persons} person(s)")
result_2d.keypoints, result_2d.scores = filter_keypoints(
    result_2d.keypoints,
    result_2d.scores,
    DEFAULT_IGNORE_KEYPOINTS
)
print(f"Filtered keypoints 14-23 (feet/toes)")

# Manuell das JSON im gleichen Format wie process_side_by_side_video erstellen
results_2d_list = [{
    "frame": 0,
    "left": {
        "num_persons": result_2d.num_persons,
        "keypoints": result_2d.keypoints.tolist(),
        "scores": result_2d.scores.tolist(), 
        "bboxes": result_2d.bboxes.tolist()
    },
    "right": {  # Für Bilder kopieren wir einfach die linke Ansicht
        "num_persons": result_2d.num_persons,
        "keypoints": result_2d.keypoints.tolist(),
        "scores": result_2d.scores.tolist(),
        "bboxes": result_2d.bboxes.tolist()
    }
}]

# 2D-Ergebnisse speichern
with open("poses_2d_filtered.json", "w") as f:
    json.dump(results_2d_list, f, indent=2)
print(f"Saved: poses_2d_filtered.json")

# ===== NEU: Annotiertes Bild OHNE Füße erstellen =====
print("\n2. Creating annotated image without feet...")
bild = estimator_2d.process_image_with_annotation(
    image_path=file, 
    output_path="image_annotated_filtered.png",
    ignore_keypoints=DEFAULT_IGNORE_KEYPOINTS  # <-- Das ist der Trick!
)
print("Saved: image_annotated_filtered.png (ohne Fuß-Linien)")

# 3. Jetzt 2D-zu-3D Konvertierung mit gefilterten Daten
print("\n3. Converting filtered 2D poses to 3D...")
results_3d = convert_2d_poses_to_3d(
    "poses_2d_filtered.json",  # <-- Verwende gefilterte JSON
    "poses_3d_filtered.json",  # <-- Neuer Dateiname
    lifting_method='geometric'  # geometric ist stabiler als hybrid
)
print("3D-Posen erfolgreich generiert!")

# 4. 3D Visualisierung
print("\n4. Creating 3D visualizations...")
plot_3d_pose_from_json(
    "poses_3d_filtered.json",  # <-- Verwende gefilterte 3D-Daten
    frame_idx=0,
    view='combined_3d',  # <-- WICHTIG: 'combined_3d' statt 'combined'
    output_path="image_3d_filtered.png",
    z_scale=5.0,
    show_plot=True
)

print("\n=== ✅ FERTIG! ===")
print("Erstellte Dateien:")
print("  📄 poses_2d_filtered.json        - Gefilterte 2D-Daten")
print("  🖼️  image_annotated_filtered.png - 2D ohne Fuß-Linien")
print("  📄 poses_3d_filtered.json        - 3D-Schätzung")
print("  🖼️  image_3d_filtered.png        - 3D-Visualisierung")
print("\n💡 Tipp: Ändere DEFAULT_IGNORE_KEYPOINTS in pose_estimator_2d.py")
print("   um andere Körperteile zu ignorieren!")