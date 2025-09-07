#!/usr/bin/env python3
"""
Beispiel-Script für die Verwendung der PoseEstimator2D Klasse

Dieses Script demonstriert verschiedene Anwendungsfälle der PoseEstimator2D Klasse:
- Grundlegende Video-Verarbeitung
- Einzelframe-Verarbeitung  
- Visualisierung und Export
- Bewegungsanalyse

Verwendung:
    python example_usage.py --video ../data/test.mov --mode balanced --max-frames 10
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Import der PoseEstimator2D Klasse
from pose_estimator_2d import PoseEstimator2D, estimate_poses_from_video


def basic_video_processing_example(video_path: str, mode: str = 'balanced', max_frames: int = 5):
    """
    Grundlegendes Beispiel für Video-Verarbeitung.
    """
    print("=== Grundlegende Video-Verarbeitung ===")
    
    # PoseEstimator2D initialisieren
    estimator = PoseEstimator2D(mode=mode, kpt_threshold=0.3)
    
    # Progress callback definieren
    def progress_callback(progress, frame_idx, num_persons):
        print(f"  Fortschritt: {progress*100:.1f}% - Frame {frame_idx} - {num_persons} Personen erkannt")
    
    # Video verarbeiten
    result = estimator.process_video(
        video_path,
        max_frames=max_frames,
        progress_callback=progress_callback
    )
    
    # Ergebnisse anzeigen
    print(f"\nErgebnisse:")
    print(f"  Verarbeitete Frames: {result.total_frames}")
    print(f"  Video FPS: {result.fps:.2f}")
    print(f"  Auflösung: {result.resolution}")
    print(f"  Frames mit Erkennungen: {result.processing_stats['frames_with_detections']}")
    print(f"  Durchschnittliche Personen pro Frame: {result.processing_stats['avg_persons_per_frame']:.2f}")
    
    # Details für jeden Frame
    print(f"\nFrame-Details:")
    for i, frame_result in enumerate(result.frame_results):
        print(f"  Frame {i}: {frame_result.num_persons} Personen")
        if frame_result.num_persons > 0:
            print(f"    Keypoints Shape: {frame_result.keypoints.shape}")
            print(f"    Scores Shape: {frame_result.scores.shape}")
    
    return result


def single_frame_processing_example(video_path: str, frame_idx: int = 0):
    """
    Beispiel für Einzelframe-Verarbeitung.
    """
    print(f"\n=== Einzelframe-Verarbeitung (Frame {frame_idx}) ===")
    
    # Estimator initialisieren
    estimator = PoseEstimator2D(mode='balanced')
    
    # Frame aus Video laden
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Fehler: Konnte Frame {frame_idx} nicht laden")
        return None
    
    # Pose estimation für einzelnes Frame
    pose_result = estimator.process_frame(frame)
    
    print(f"Frame-Ergebnis:")
    print(f"  Erkannte Personen: {pose_result.num_persons}")
    
    if pose_result.num_persons > 0:
        print(f"  Keypoints Shape: {pose_result.keypoints.shape}")
        print(f"  Durchschnittliche Konfidenz: {pose_result.scores.mean():.3f}")
        
        # Beispiel: Spezifische Keypoints auslesen
        for person_idx in range(pose_result.num_persons):
            keypoints = pose_result.keypoints[person_idx]
            scores = pose_result.scores[person_idx]
            
            # Nase (Keypoint 0), rechtes Handgelenk (Keypoint 10), linkes Handgelenk (Keypoint 9)
            nose = keypoints[0]
            right_wrist = keypoints[10] 
            left_wrist = keypoints[9]
            
            print(f"  Person {person_idx}:")
            print(f"    Nase: ({nose[0]:.1f}, {nose[1]:.1f}), Konfidenz: {scores[0]:.3f}")
            print(f"    Rechtes Handgelenk: ({right_wrist[0]:.1f}, {right_wrist[1]:.1f}), Konfidenz: {scores[10]:.3f}")
            print(f"    Linkes Handgelenk: ({left_wrist[0]:.1f}, {left_wrist[1]:.1f}), Konfidenz: {scores[9]:.3f}")
    
    return frame, pose_result


def visualization_example(video_path: str, output_dir: str = "../output/pose-estimator"):
    """
    Beispiel für Visualisierung und Export.
    """
    print(f"\n=== Visualisierung und Export ===")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Estimator initialisieren
    estimator = PoseEstimator2D(mode='balanced')
    
    # Wenige Frames verarbeiten für Demo
    result = estimator.process_video(video_path, max_frames=5)
    
    # 1. JSON Export
    json_path = output_path / "poses.json"
    estimator.export_keypoints_to_json(result, json_path)
    print(f"Keypoints exportiert nach: {json_path}")
    
    # 2. Annotiertes Video speichern
    video_path_out = output_path / "annotated_video.mp4"
    estimator.save_results_to_video(result, video_path_out, video_path)
    print(f"Annotiertes Video gespeichert: {video_path_out}")
    
    # 3. Einzelne annotierte Frames speichern
    cap = cv2.VideoCapture(video_path)
    
    for i, frame_result in enumerate(result.frame_results):
        # Frame laden
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_result.frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Frame annotieren
        annotated_frame = estimator.visualize_frame(frame, frame_result)
        
        # Speichern
        frame_path = output_path / f"annotated_frame_{i:02d}.jpg"
        cv2.imwrite(str(frame_path), annotated_frame)
        print(f"Annotiertes Frame gespeichert: {frame_path}")
    
    cap.release()


def movement_analysis_example(video_path: str):
    """
    Beispiel für Bewegungsanalyse.
    """
    print(f"\n=== Bewegungsanalyse ===")
    
    # Estimator initialisieren
    estimator = PoseEstimator2D(mode='balanced')
    
    # Video verarbeiten (mehr Frames für Bewegungsanalyse)
    result = estimator.process_video(video_path, max_frames=20)
    
    if len(result.frame_results) == 0:
        print("Keine Frames für Analyse verfügbar")
        return
    
    # Trajektorie der ersten Person analysieren
    trajectory = result.get_person_trajectory(person_idx=0)
    
    # Keypoints über Zeit sammeln (nur Frames wo Person erkannt wurde)
    valid_keypoints = []
    frame_indices = []
    
    for i, keypoints in enumerate(trajectory):
        if keypoints is not None:
            valid_keypoints.append(keypoints)
            frame_indices.append(i)
    
    if len(valid_keypoints) < 2:
        print("Nicht genügend Daten für Bewegungsanalyse")
        return
        
    print(f"Person in {len(valid_keypoints)}/{len(trajectory)} Frames erkannt")
    
    # Bewegung verschiedener Körperteile analysieren
    body_parts = {
        'Nase': 0,
        'Rechte Schulter': 6,
        'Linke Schulter': 5, 
        'Rechtes Handgelenk': 10,
        'Linkes Handgelenk': 9,
        'Rechte Hüfte': 12,
        'Linke Hüfte': 11
    }
    
    print(f"\nBewegungsanalyse:")
    
    for part_name, keypoint_idx in body_parts.items():
        positions = [kpts[keypoint_idx] for kpts in valid_keypoints]
        
        # Bewegungsumfang berechnen
        positions_array = np.array(positions)
        x_range = positions_array[:, 0].max() - positions_array[:, 0].min()
        y_range = positions_array[:, 1].max() - positions_array[:, 1].min()
        
        # Durchschnittliche Geschwindigkeit
        velocities = []
        for i in range(1, len(positions)):
            pos_diff = np.array(positions[i]) - np.array(positions[i-1])
            velocity = np.linalg.norm(pos_diff)
            velocities.append(velocity)
        
        avg_velocity = np.mean(velocities) if velocities else 0
        
        print(f"  {part_name}:")
        print(f"    Bewegungsumfang X: {x_range:.1f} Pixel")
        print(f"    Bewegungsumfang Y: {y_range:.1f} Pixel") 
        print(f"    Durchschnittliche Geschwindigkeit: {avg_velocity:.2f} Pixel/Frame")


def performance_comparison_example(video_path: str):
    """
    Beispiel für Performance-Vergleich verschiedener Modi.
    """
    print(f"\n=== Performance-Vergleich ===")
    
    modes = ['lightweight', 'balanced', 'performance']
    max_frames = 5  # Wenige Frames für schnellen Vergleich
    
    import time
    
    results = {}
    
    for mode in modes:
        print(f"\nTeste Modus: {mode}")
        
        # Zeit messen
        start_time = time.time()
        
        estimator = PoseEstimator2D(mode=mode)
        result = estimator.process_video(video_path, max_frames=max_frames)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Statistiken sammeln
        total_persons = sum(fr.num_persons for fr in result.frame_results)
        avg_confidence = np.mean([
            fr.scores.mean() for fr in result.frame_results 
            if fr.num_persons > 0 and len(fr.scores) > 0
        ]) if any(fr.num_persons > 0 for fr in result.frame_results) else 0
        
        results[mode] = {
            'time': processing_time,
            'total_persons': total_persons,
            'avg_confidence': avg_confidence,
            'frames_with_detections': result.processing_stats['frames_with_detections']
        }
        
        print(f"  Verarbeitungszeit: {processing_time:.2f}s")
        print(f"  Erkannte Personen (gesamt): {total_persons}")
        print(f"  Durchschnittliche Konfidenz: {avg_confidence:.3f}")
        print(f"  Frames mit Erkennungen: {result.processing_stats['frames_with_detections']}")
    
    # Vergleich ausgeben
    print(f"\n=== Vergleichstabelle ===")
    print(f"{'Modus':<12} {'Zeit (s)':<10} {'Personen':<10} {'Konfidenz':<12} {'Erkennungen':<12}")
    print("-" * 60)
    
    for mode, stats in results.items():
        print(f"{mode:<12} {stats['time']:<10.2f} {stats['total_persons']:<10} "
              f"{stats['avg_confidence']:<12.3f} {stats['frames_with_detections']:<12}")


def convenience_function_example(video_path: str):
    """
    Beispiel für die Convenience-Funktion.
    """
    print(f"\n=== Convenience-Funktion ===")
    
    # Einfache Verwendung mit der Convenience-Funktion
    result = estimate_poses_from_video(
        video_path,
        mode='balanced',
        max_frames=3,
        device='cpu',
        kpt_threshold=0.3
    )
    
    print(f"Convenience-Funktion Ergebnis:")
    print(f"  Frames verarbeitet: {result.total_frames}")
    print(f"  Erste Erkennungen: {result.frame_results[0].num_persons if result.frame_results else 0}")


def main():
    """
    Hauptfunktion mit Kommandozeilen-Interface.
    """
    parser = argparse.ArgumentParser(description="Beispiel-Script für PoseEstimator2D")
    parser.add_argument("--video", default="../data/test.mov", 
                       help="Pfad zur Video-Datei")
    parser.add_argument("--mode", choices=['performance', 'lightweight', 'balanced'],
                       default='balanced', help="Performance-Modus")
    parser.add_argument("--max-frames", type=int, default=10,
                       help="Maximale Anzahl Frames für Beispiele")
    parser.add_argument("--output-dir", default="../output/pose-estimator",
                       help="Output-Verzeichnis")
    parser.add_argument("--examples", nargs="+", 
                       choices=['basic', 'single', 'visualization', 'movement', 'performance', 'convenience', 'all'],
                       default=['all'], help="Auszuführende Beispiele")
    
    args = parser.parse_args()
    
    # Video-Pfad prüfen
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Fehler: Video-Datei nicht gefunden: {video_path}")
        return
    
    print(f"PoseEstimator2D Beispiele")
    print(f"Video: {video_path}")
    print(f"Modus: {args.mode}")
    print(f"Max Frames: {args.max_frames}")
    
    # Beispiele ausführen
    examples_to_run = args.examples
    if 'all' in examples_to_run:
        examples_to_run = ['basic', 'single', 'visualization', 'movement', 'performance', 'convenience']
    
    try:
        # Grundlegende Video-Verarbeitung
        if 'basic' in examples_to_run:
            result = basic_video_processing_example(str(video_path), args.mode, args.max_frames)
        
        # Einzelframe-Verarbeitung
        if 'single' in examples_to_run:
            single_frame_processing_example(str(video_path), frame_idx=0)
        
        # Visualisierung und Export
        if 'visualization' in examples_to_run:
            visualization_example(str(video_path), args.output_dir)
        
        # Bewegungsanalyse
        if 'movement' in examples_to_run:
            movement_analysis_example(str(video_path))
        
        # Performance-Vergleich
        if 'performance' in examples_to_run:
            performance_comparison_example(str(video_path))
            
        # Convenience-Funktion
        if 'convenience' in examples_to_run:
            convenience_function_example(str(video_path))
    
    except Exception as e:
        print(f"Fehler beim Ausführen der Beispiele: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n=== Beispiele abgeschlossen ===")


if __name__ == "__main__":
    main()
