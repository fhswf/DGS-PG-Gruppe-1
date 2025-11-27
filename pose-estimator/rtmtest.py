from pose_estimator_2d import PoseEstimator2D
from pose_estimator_3d import convert_2d_poses_to_3d
from pose_3d_visualizer import plot_3d_pose_from_json, plot_multiple_views
import json

print("=== Testing 2D to 3D Pose Conversion ===")

#file = "V.png"
#file = "../hocke.jpg"
file = "../mensch.jpg"

# 1. Zuerst 2D-Posen mit dem originalen Code berechnen
print("\n1. Calculating 2D poses...")
estimator_2d = PoseEstimator2D(kpt_threshold=0.9)

# Für ein Bild verwenden wir process_image statt process_side_by_side_video
result_2d = estimator_2d.process_image(file)

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
with open("poses_2d.json", "w") as f:
    json.dump(results_2d_list, f, indent=2)

print(f"2D poses saved. Detected {result_2d.num_persons} person(s)")

bild = estimator_2d.process_image_with_annotation(image_path=file, output_path= "Nathalie-annotated.png")

# 2. Jetzt 2D-zu-3D Konvertierung
print("\n2. Converting 2D poses to 3D...")
results_3d = convert_2d_poses_to_3d(
    "poses_2d.json",
    "poses_3d.json", 
    lifting_method='hybrid'
)

print("3D-Posen erfolgreich generiert!")
print(results_3d)

# 3. 3D Visualisierung
print("\n3. Creating 3D visualizations...")

# Einzelner 3D-Plot (combined view)
print("Plotting 3D pose (combined view)...")
plot_3d_pose_from_json(
    "poses_3d.json",
    frame_idx=0,
    view='combined',
    output_path="Nathalie_3d_combined.png",
    z_scale=5.0,
    show_plot=True
)