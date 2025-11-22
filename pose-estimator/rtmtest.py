from pose_estimator_2d import PoseEstimator2D
from pose_estimator_3d import convert_2d_poses_to_3d
import json

print("=== Testing 2D to 3D Pose Conversion ===")

# 1. Zuerst 2D-Posen mit dem originalen Code berechnen
print("\n1. Calculating 2D poses...")
estimator_2d = PoseEstimator2D()

# Für ein Bild verwenden wir process_image statt process_side_by_side_video
result_2d = estimator_2d.process_image("hocke.jpg")

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

bild = estimator_2d.process_image_with_annotation(image_path="hocke.jpg", output_path= "hocke-annotated.jpg")

# 2. Jetzt 2D-zu-3D Konvertierung
print("\n2. Converting 2D poses to 3D...")
results_3d = convert_2d_poses_to_3d(
    "poses_2d.json",
    "poses_3d.json", 
    lifting_method='hybrid'
)

print("3D-Posen erfolgreich generiert!")
print(results_3d)