"""
Test script for 3D side-by-side video processing using PoseEstimator2D.
Processes the test_sbs.mp4 file in the data directory and saves annotated PNGs for the first frame.
"""
from pose_estimator_2d import PoseEstimator2D

if __name__ == "__main__":
    estimator = PoseEstimator2D(mode="balanced", device="cpu")
    results = estimator.process_side_by_side_video(
        "../data/test_sbs.mp4",
        output_json_path="poses_test_sbs.json",
        show_first_annotated=True,
        max_frames=10  # Limit for quick test, remove or increase for full video
    )
    print(f"Processed {len(results)} frames. Results saved to poses_test_sbs.json.")
    print("Annotated PNGs for the first frame saved as first_frame_left_annotated.png and first_frame_right_annotated.png.")
