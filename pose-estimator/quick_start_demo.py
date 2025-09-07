#!/usr/bin/env python3
"""
Quick Start Demo für PoseEstimator2D

Dieses Script zeigt, wie die PoseEstimator2D Klasse verwendet wird,
nachdem RTMLib installiert wurde.

Installation:
    pip install rtmlib

Verwendung:
    python quick_start_demo.py
"""

def main():
    print("PoseEstimator2D - Quick Start Demo")
    print("=" * 40)
    
    # 1. Import prüfen
    print("1. Teste Import...")
    try:
        from pose_estimator_2d import PoseEstimator2D, estimate_poses_from_video
        print("   ✓ Import erfolgreich!")
    except ImportError as e:
        print(f"   ✗ Import fehlgeschlagen: {e}")
        print("\n   Installieren Sie RTMLib mit:")
        print("   pip install rtmlib")
        return
    
    # 2. Estimator initialisieren
    print("\n2. Initialisiere PoseEstimator2D...")
    try:
        estimator = PoseEstimator2D(mode='balanced', device='cpu')
        print("   ✓ Estimator erfolgreich initialisiert!")
        print(f"   Mode: {estimator.mode}")
        print(f"   Device: {estimator.device}")
        print(f"   Backend: {estimator.backend}")
    except Exception as e:
        print(f"   ✗ Initialisierung fehlgeschlagen: {e}")
        return
    
    # 3. Video-Verarbeitung (mit Beispiel-Video falls vorhanden)
    print("\n3. Video-Verarbeitung...")
    
    import os
    from pathlib import Path
    
    # Bestimme das Projekt-Root-Verzeichnis
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # pose-estimator -> DGS-PG-Gruppe-1
    
    video_paths = [
        project_root / "data" / "test.mov",
        project_root / "data" / "1031.mp4",
        "../data/test.mov",  # Fallback für relative Pfade
        "../data/1031.mp4"
    ]
    
    video_found = None
    for video_path in video_paths:
        if Path(video_path).exists():
            video_found = str(video_path)
            break
    
    if video_found:
        print(f"   Verwende Video: {video_found}")
        try:
            # Nur wenige Frames für Demo
            result = estimator.process_video(video_found, max_frames=3)
            
            print(f"   ✓ Video verarbeitet!")
            print(f"   Frames: {result.total_frames}")
            print(f"   Auflösung: {result.resolution}")
            print(f"   FPS: {result.fps:.2f}")
            
            # Frame-Details
            for i, frame_result in enumerate(result.frame_results):
                print(f"   Frame {i}: {frame_result.num_persons} Personen")
                if frame_result.num_persons > 0:
                    avg_confidence = frame_result.scores.mean()
                    print(f"      -> Durchschnittliche Konfidenz: {avg_confidence:.3f}")
                    print(f"      -> Keypoints Shape: {frame_result.keypoints.shape}")
                    # Beispiel: Erste Person, Nase (Keypoint 0)
                    nose_pos = frame_result.keypoints[0][0]  # Erste Person, Nase
                    nose_conf = frame_result.scores[0][0]
                    print(f"      -> Nase Position: ({nose_pos[0]:.1f}, {nose_pos[1]:.1f}), Konfidenz: {nose_conf:.3f}")
                
        except Exception as e:
            print(f"   ✗ Video-Verarbeitung fehlgeschlagen: {e}")
    else:
        print("   ⚠ Kein Test-Video gefunden")
        print("   Verfügbare Video-Pfade:")
        for path in video_paths:
            print(f"     - {path}")
    
    # 4. Convenience-Funktion
    print("\n4. Teste Convenience-Funktion...")
    if video_found:
        try:
            # Verwende den gleichen Modus wie oben (balanced) um Modell-Download zu vermeiden
            result = estimate_poses_from_video(
                video_found,
                mode='balanced',  # Gleicher Modus wie bereits initialisiert
                max_frames=2
            )
            print(f"   ✓ Convenience-Funktion erfolgreich!")
            print(f"   Frames: {result.total_frames}")
        except Exception as e:
            print(f"   ✗ Convenience-Funktion fehlgeschlagen: {e}")
            print("   Hinweis: Möglicherweise SSL-Problem beim Model-Download")
    else:
        print("   ⚠ Übersprungen (kein Video)")
    
    # 5. Beispiele zeigen
    print("\n5. Weitere Beispiele:")
    print("   - Für ausführliche Beispiele: python example_usage.py")
    print("   - Für Tests: python test_pose_estimator.py")
    print("   - Dokumentation: README.md")
    
    print("\n" + "=" * 40)
    print("Demo abgeschlossen!")
    
    # Verwendungsbeispiele ausgeben
    print("\nSchnelle Verwendung:")
    print("""
# Einfaches Beispiel
from pose_estimator_2d import PoseEstimator2D

estimator = PoseEstimator2D(mode='balanced')
result = estimator.process_video('video.mp4', max_frames=10)

# Ergebnisse anzeigen
for i, frame_result in enumerate(result.frame_results):
    print(f"Frame {i}: {frame_result.num_persons} Personen")
    if frame_result.num_persons > 0:
        keypoints = frame_result.keypoints  # Shape: (N, 133, 2)
        scores = frame_result.scores        # Shape: (N, 133)
        
# Export
estimator.export_keypoints_to_json(result, 'poses.json')
estimator.save_results_to_video(result, 'annotated.mp4', 'video.mp4')
""")


if __name__ == "__main__":
    main()
