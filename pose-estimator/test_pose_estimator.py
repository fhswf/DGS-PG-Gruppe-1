#!/usr/bin/env python3
"""
Einfacher Test für die PoseEstimator2D Klasse.

Dieses Script testet die grundlegende Funktionalität der PoseEstimator2D Klasse
ohne externe Dependencies außer rtmlib und opencv.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Test imports
try:
    from pose_estimator_2d import PoseEstimator2D, PoseResult, VideoResult, estimate_poses_from_video
    print("✓ Import erfolgreich")
except ImportError as e:
    print(f"✗ Import-Fehler: {e}")
    sys.exit(1)


def test_initialization():
    """Test der Initialisierung mit verschiedenen Parametern."""
    print("\n=== Test: Initialisierung ===")
    
    try:
        # Standard-Initialisierung
        estimator1 = PoseEstimator2D()
        print("✓ Standard-Initialisierung erfolgreich")
        
        # Mit benutzerdefinierten Parametern
        estimator2 = PoseEstimator2D(
            mode='lightweight',
            backend='onnxruntime',
            device='cpu',
            kpt_threshold=0.5
        )
        print("✓ Initialisierung mit benutzerdefinierten Parametern erfolgreich")
        
        # Test Property Setter
        estimator2.mode = 'balanced'
        assert estimator2.mode == 'balanced'
        print("✓ Mode Property Setter funktioniert")
        
        # Test ungültiger Parameter
        try:
            estimator3 = PoseEstimator2D(mode='invalid_mode')
            print("✗ Ungültiger Mode sollte Fehler werfen")
        except ValueError:
            print("✓ Validation für ungültigen Mode funktioniert")
            
    except Exception as e:
        print(f"✗ Initialisierung fehlgeschlagen: {e}")
        return False
    
    return True


def test_synthetic_frame():
    """Test mit einem synthetischen Frame."""
    print("\n=== Test: Synthetisches Frame ===")
    
    try:
        estimator = PoseEstimator2D(mode='lightweight')
        
        # Erstelle ein synthetisches Testbild (schwarzes Bild)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Füge einfache Formen hinzu (simuliert eine Person)
        cv2.rectangle(test_frame, (250, 150), (390, 450), (100, 100, 100), -1)  # Körper
        cv2.circle(test_frame, (320, 120), 30, (150, 150, 150), -1)  # Kopf
        
        # Pose estimation auf synthetischem Frame
        result = estimator.process_frame(test_frame)
        
        print(f"✓ Frame verarbeitet - {result.num_persons} Personen erkannt")
        print(f"  Keypoints Shape: {result.keypoints.shape}")
        print(f"  Scores Shape: {result.scores.shape}")
        print(f"  Bboxes Shape: {result.bboxes.shape}")
        
        # Test Visualisierung
        annotated_frame = estimator.visualize_frame(test_frame, result)
        assert annotated_frame.shape == test_frame.shape
        print("✓ Visualisierung funktioniert")
        
    except Exception as e:
        print(f"✗ Synthetisches Frame Test fehlgeschlagen: {e}")
        return False
    
    return True


def test_data_structures():
    """Test der Datenstrukturen."""
    print("\n=== Test: Datenstrukturen ===")
    
    try:
        # Test PoseResult
        dummy_keypoints = np.zeros((1, 133, 2))
        dummy_scores = np.ones((1, 133)) * 0.5
        dummy_bboxes = np.array([[100, 100, 200, 300, 0.9]])
        
        pose_result = PoseResult(
            frame_idx=0,
            keypoints=dummy_keypoints,
            scores=dummy_scores,
            bboxes=dummy_bboxes,
            num_persons=1
        )
        
        assert pose_result.frame_idx == 0
        assert pose_result.num_persons == 1
        print("✓ PoseResult erstellt")
        
        # Test VideoResult
        video_result = VideoResult(
            video_path="test.mp4",
            total_frames=10
        )
        
        video_result.frame_results.append(pose_result)
        
        # Test Methoden
        keypoints = video_result.get_keypoints_by_frame(0)
        assert keypoints is not None
        print("✓ VideoResult.get_keypoints_by_frame funktioniert")
        
        all_keypoints = video_result.get_all_keypoints()
        assert len(all_keypoints) == 1
        print("✓ VideoResult.get_all_keypoints funktioniert")
        
        trajectory = video_result.get_person_trajectory(0)
        assert len(trajectory) == 1
        print("✓ VideoResult.get_person_trajectory funktioniert")
        
    except Exception as e:
        print(f"✗ Datenstrukturen Test fehlgeschlagen: {e}")
        return False
    
    return True


def test_convenience_function():
    """Test der Convenience-Funktion mit synthetischen Daten."""
    print("\n=== Test: Convenience-Funktion ===")
    
    try:
        # Erstelle ein temporäres Video mit synthetischen Frames
        temp_video_path = "/tmp/test_video.mp4"
        
        # Einfaches synthetisches Video erstellen
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, 1.0, (640, 480))
        
        # 3 Frames schreiben
        for i in range(3):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Bewegende "Person"
            x_offset = i * 50 + 200
            cv2.rectangle(frame, (x_offset, 150), (x_offset + 140, 450), (100, 100, 100), -1)
            cv2.circle(frame, (x_offset + 70, 120), 30, (150, 150, 150), -1)
            out.write(frame)
        
        out.release()
        
        # Test der Convenience-Funktion
        if Path(temp_video_path).exists():
            result = estimate_poses_from_video(
                temp_video_path,
                mode='lightweight',
                max_frames=3
            )
            
            print(f"✓ Convenience-Funktion erfolgreich - {result.total_frames} Frames verarbeitet")
            
            # Cleanup
            Path(temp_video_path).unlink()
        else:
            print("⚠ Konnte temporäres Video nicht erstellen, überspringe Test")
            
    except Exception as e:
        print(f"✗ Convenience-Funktion Test fehlgeschlagen: {e}")
        return False
    
    return True


def test_error_handling():
    """Test der Fehlerbehandlung."""
    print("\n=== Test: Fehlerbehandlung ===")
    
    try:
        estimator = PoseEstimator2D()
        
        # Test mit nicht-existierender Videodatei
        try:
            result = estimator.process_video("/path/that/does/not/exist.mp4")
            print("✗ Sollte FileNotFoundError werfen")
            return False
        except FileNotFoundError:
            print("✓ FileNotFoundError für nicht-existierende Datei")
        
        # Test mit ungültigen Parametern
        try:
            estimator = PoseEstimator2D(mode="invalid")
            print("✗ Sollte ValueError für ungültigen Mode werfen")
            return False
        except ValueError:
            print("✓ ValueError für ungültigen Mode")
        
        # Test mit ungültigem kpt_threshold
        try:
            estimator = PoseEstimator2D(kpt_threshold=2.0)
            print("✗ Sollte ValueError für ungültigen kpt_threshold werfen")
            return False
        except ValueError:
            print("✓ ValueError für ungültigen kpt_threshold")
            
    except Exception as e:
        print(f"✗ Fehlerbehandlung Test fehlgeschlagen: {e}")
        return False
    
    return True


def main():
    """Haupttest-Funktion."""
    print("PoseEstimator2D - Funktionstest")
    print("=" * 40)
    
    tests = [
        ("Initialisierung", test_initialization),
        ("Synthetisches Frame", test_synthetic_frame),
        ("Datenstrukturen", test_data_structures),
        ("Convenience-Funktion", test_convenience_function),
        ("Fehlerbehandlung", test_error_handling)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✓ Test '{test_name}' bestanden")
            else:
                print(f"✗ Test '{test_name}' fehlgeschlagen")
        except Exception as e:
            print(f"✗ Test '{test_name}' mit Ausnahme fehlgeschlagen: {e}")
    
    print(f"\n{'='*60}")
    print(f"Testergebnisse: {passed}/{total} Tests bestanden")
    
    if passed == total:
        print("🎉 Alle Tests bestanden! PoseEstimator2D funktioniert korrekt.")
        return 0
    else:
        print("❌ Einige Tests fehlgeschlagen. Überprüfen Sie die Implementierung.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
