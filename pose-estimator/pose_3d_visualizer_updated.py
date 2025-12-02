"""
3D Pose Visualizer mit korrekten Verbindungen

Visualisiert 3D-Posen (Körperhaltungen) mit:
- Korrekten Schulterverbindungen (Ohren zu Schultern statt 0→5/6)
- Gefilterten Bein-/Fußverbindungen (Beine werden nicht gezeichnet)
- Verbesserte Z-Achsen-Skalierung (bessere Tiefendarstellung)

EINFACHE ERKLÄRUNG:
Dieses Programm liest Daten über Körperpositionen aus einer Datei
und zeichnet daraus eine 3D-Figur - wie einen digitalen Stabmann.
Man kann sehen, wie eine Person im Raum steht, auch in der Tiefe.
"""

# ===============================================
# 📦 IMPORTIEREN VON HILFS-MODULEN
# ===============================================
# Hier laden wir Werkzeuge, die wir benötigen:
import numpy as np  # Für Mathe und Listen mit Zahlen
import matplotlib.pyplot as plt  # Zum Erstellen von Grafiken und Bildern
from mpl_toolkits.mplot3d import Axes3D  # Speziell für 3D-Grafiken
import json  # Zum Lesen der Daten-Dateien (JSON-Format)
from pathlib import Path  # Zum Arbeiten mit Dateipfaden (Ordnern/Dateien)
from typing import Union, Optional, List, Tuple  # Für bessere Code-Lesbarkeit

# ===============================================
# 🦴 KÖRPER-VERBINDUNGEN DEFINIEREN
# ===============================================
# Hier sagen wir dem Programm, welche Punkte des Körpers 
# mit Linien verbunden werden sollen (wie ein Stabmann-Skelett)

# ██████ KÖRPER-SKELETT ██████
# Welche Punkte sollen mit Linien verbunden werden? (Startpunkt, Endpunkt)
BODY_CONNECTIONS = [
    # 🟡 KOPF-BEREICH (verbesserte Version)
    (0, 1), (0, 2),  # Linie von Nase zu linkem Auge, Nase zu rechtem Auge
    (1, 3), (2, 4),  # Linkes Auge zu linkem Ohr, rechtes Auge zu rechtem Ohr
    
    # 🔵 SCHULTERN (KORRIGIERT - richtig von Ohren zu Schultern)
    (3, 5), (4, 6),   # Linkes Ohr zu linker Schulter, rechtes Ohr zu rechter Schulter
    (5, 6),           # Linie zwischen beiden Schultern
    
    # 💪 ARME
    (5, 7), (7, 91),   # Linker Arm: Schulter → Ellbogen → Handgelenk
    (6, 8), (8, 112),  # Rechter Arm: Schulter → Ellbogen → Handgelenk
    
    # 🏋️ OBERKÖRPER/RUMPF
    (5, 11), (6, 12),  # Schultern zu Hüften
    (11, 12),          # Linie zwischen beiden Hüften
    
    # 🦵 BEINE SIND HIER ABSICHTLICH WEGGELASSEN!
    # (werden nicht gezeichnet, damit wir uns auf Oberkörper konzentrieren)
]

# 🟢 GESICHTS-KONTUR (vereinfacht)
# Zeichnet die Umrisse des Gesichts
FACE_CONNECTIONS = [
    (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
    (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37),
    (37, 38), (38, 39), (39, 40), (40, 41), (41, 42),  # Gesichtsprofil-Linien
    (0, 43), (43, 44), (44, 45), (45, 46), (46, 47),   # Nase
    (52, 53), (53, 54), (54, 55), (55, 56),            # Mund
    (56, 57), (57, 58), (58, 59), (59, 60),
]

# 🟠 LINKE HAND (Finger-Verbindungen)
LEFT_HAND_CONNECTIONS = [
    (7, 91),  # Ellbogen zu Handwurzel
    # Daumen:
    (91, 92), (92, 93), (93, 94), (94, 95),
    # Zeigefinger:
    (91, 96), (96, 97), (97, 98), (98, 99),
    # Mittelfinger:
    (91, 100), (100, 101), (101, 102), (102, 103),
    # Ringfinger:
    (91, 104), (104, 105), (105, 106), (106, 107),
    # Kleiner Finger:
    (91, 108), (108, 109), (109, 110), (110, 111),
]

# 🟡 RECHTE HAND (gleiche Struktur wie linke Hand)
RIGHT_HAND_CONNECTIONS = [
    (8, 112),  # Handgelenk zu Handwurzel
    # Daumen:
    (112, 113), (113, 114), (114, 115), (115, 116),
    # Zeigefinger:
    (112, 117), (117, 118), (118, 119), (119, 120),
    # Mittelfinger:
    (112, 121), (121, 122), (122, 123), (123, 124),
    # Ringfinger:
    (112, 125), (125, 126), (126, 127), (127, 128),
    # Kleiner Finger:
    (112, 129), (129, 130), (130, 131), (131, 132),
]

# ===============================================
# 🎯 HAUPTFUNKTION: 3D-POSE VISUALISIEREN
# ===============================================
def plot_3d_pose_from_json(
    json_path: Union[str, Path],      # 📁 Pfad zur Daten-Datei
    frame_idx: int = 0,               # 🎞️ Welches Einzelbild/Bewegungsmoment
    view: str = 'combined_3d',        # 📷 Kameraperspektive wählen
    output_path: Optional[Union[str, Path]] = None,  # 💾 Speicherort (optional)
    show_plot: bool = True,           # 👀 Sofort anzeigen oder nur speichern?
    confidence_threshold: float = 0.3, # 🎯 Nur sichere Punkte anzeigen (>30%)
    figsize: Tuple[int, int] = (14, 10), # 📐 Bildgröße in Zentimeter
    z_scale: float = 1.0,             # 🔍 Tiefenvergrößerung (z.B. 5.0 für besserer Sicht)
    show_hands: bool = True,          # ✋ Hände anzeigen?
    show_face: bool = True            # 😀 Gesicht anzeigen?
):
    """
    🎯 DIE WICHTIGSTE FUNKTION!
    Liest Körperpositions-Daten und zeichnet eine 3D-Figur.
    
    Denk dir das wie eine digitale Puppe, die du aus allen Richtungen betrachten kannst.
    
    Beispiel-Aufruf:
    plot_3d_pose_from_json("meine_daten.json", frame_idx=0, z_scale=5.0)
    """
    
    # ===============================================
    # 📥 SCHRITT 1: DATEN AUS DER DATEI LADEN
    # ===============================================
    print(f"📖 Lese Daten aus: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)  # 📄 Die ganze Datei wird geladen
    
    # 🚨 Prüfen: Gibt es diesen Frame (Bildmoment)?
    if frame_idx >= len(data):
        print(f"❌ Fehler: Frame {frame_idx} existiert nicht!")
        print(f"   Verfügbar sind nur {len(data)} Frames.")
        return
    
    frame_data = data[frame_idx]  # 🎯 Nur das gewählte Einzelbild nehmen
    view_data = frame_data.get(view)  # 📷 Die gewünschte Kameraperspektive
    
    if view_data is None:
        print(f"❌ Fehler: Ansicht '{view}' nicht gefunden!")
        return
    
    # ===============================================
    # 📊 SCHRITT 2: 3D-PUNKTE EXTRAHIEREN
    # ===============================================
    # Jeder Körperpunkt hat 3 Koordinaten: X (links/rechts), Y (oben/unten), Z (Tiefe)
    keypoints_3d = np.array(view_data['keypoints_3d'])  # 📍 Alle Punkte
    scores_3d = np.array(view_data['scores_3d'])       # 🎯 Wie sicher ist jede Position?
    
    # ===============================================
    # 🔍 SCHRITT 3: TIEFE VERGRÖSSERN FÜR BESSERE SICHT
    # ===============================================
    if z_scale != 1.0:
        keypoints_3d_scaled = keypoints_3d.copy()  # 📋 Kopie zum Bearbeiten
        keypoints_3d_scaled[:, :, 2] *= z_scale    # ✨ Alle Z-Werte (Tiefe) multiplizieren
        
        # ℹ️ Info für Entwickler: Zeige Tiefenbereich an
        print(f"🔍 Tiefenskaliert um Faktor {z_scale}")
    
    else:
        keypoints_3d_scaled = keypoints_3d  # Ohne Skalierung
    
    # ===============================================
    # 🎨 SCHRITT 4: LEERE 3D-GRAFIK VORBEREITEN
    # ===============================================
    fig = plt.figure(figsize=figsize)  # 🖼️ Neues Bild mit bestimmter Größe
    ax = fig.add_subplot(111, projection='3d')  # 📐 3D-Achsen hinzufügen
    
    # ===============================================
    # 👤 SCHRITT 5: JEDE PERSON ZEICHNEN
    # ===============================================
    # (Ein Bild kann mehrere Personen enthalten)
    num_people = len(keypoints_3d_scaled)
    print(f"👥 Zeichne {num_people} Person(en)...")
    
    for person_idx in range(num_people):
        kpts = keypoints_3d_scaled[person_idx]  # 📍 Punkte dieser Person
        scores = scores_3d[person_idx]          # 🎯 Genauigkeiten dieser Person
        
        # 5a: 🦴 KÖRPER-SKELETT ZEICHNEN (Blaue Linien)
        _plot_skeleton_3d(
            ax, kpts, scores, BODY_CONNECTIONS,
            color='blue', linewidth=2.5, 
            label='Körper' if person_idx == 0 else None,  # 📝 Beschriftung nur einmal
            threshold=confidence_threshold
        )
        
        # 5b: 😀 OPTIONAL: GESICHTSKONTUR ZEICHNEN (Grüne Linien)
        if show_face:
            _plot_skeleton_3d(
                ax, kpts, scores, FACE_CONNECTIONS,
                color='green', linewidth=1, alpha=0.5,  # alpha = Durchsichtigkeit
                threshold=confidence_threshold
            )
        
        # 5c: ✋ OPTIONAL: HÄNDE ZEICHNEN
        if show_hands:
            # Linke Hand (Rote Linien)
            _plot_skeleton_3d(
                ax, kpts, scores, LEFT_HAND_CONNECTIONS,
                color='red', linewidth=1.5, alpha=0.7,
                label='Linke Hand' if person_idx == 0 else None,
                threshold=confidence_threshold
            )
            # Rechte Hand (Orange Linien)
            _plot_skeleton_3d(
                ax, kpts, scores, RIGHT_HAND_CONNECTIONS,
                color='orange', linewidth=1.5, alpha=0.7,
                label='Rechte Hand' if person_idx == 0 else None,
                threshold=confidence_threshold
            )
        
        # 5d: ⚫ KÖRPERPUNKTE ALS PUNKTE ZEICHNEN
        valid_mask = scores > confidence_threshold  # ✅ Nur sichere Punkte
        valid_kpts = kpts[valid_mask]               # 📍 Gefilterte Punkte
        
        if len(valid_kpts) > 0:
            # 🚫 Filtere Punkte mit Null-Koordinaten (fehlende Daten)
            non_zero_mask = ~np.all(valid_kpts == 0, axis=1)
            valid_kpts = valid_kpts[non_zero_mask]
            
            if len(valid_kpts) > 0:
                # ⚫ Zeichne schwarze Punkte mit weißem Rand
                ax.scatter(
                    valid_kpts[:, 0], valid_kpts[:, 1], valid_kpts[:, 2],
                    c='black', marker='o', s=30, alpha=0.7,
                    edgecolors='white', linewidths=0.5
                )
    
    # ===============================================
    # 📝 SCHRITT 6: GRAFIK BESCHRIFTEN
    # ===============================================
    ax.set_xlabel('X (links ↔ rechts)', fontsize=11)
    ax.set_ylabel('Y (oben ↔ unten)', fontsize=11)
    
    # Z-Achse mit Skalierungs-Info
    if z_scale != 1.0:
        ax.set_zlabel(f'Z (Tiefe, {z_scale}x vergrößert)', fontsize=11)
    else:
        ax.set_zlabel('Z (Tiefe)', fontsize=11)
    
    # 🏷️ Titel der Grafik
    method = view_data.get('method', 'unbekannt')
    confidence = view_data.get('confidence', 0)
    ax.set_title(
        f'3D Körperhaltung - Moment {frame_idx}\n'
        f'Kamera: {view} | Methode: {method} | Sicherheit: {confidence:.1%}',
        fontsize=13, pad=15
    )
    
    # 🔄 Y-Achse umdrehen (in Bildern zeigt Y nach unten, in 3D nach oben)
    ax.invert_yaxis()
    
    # 📖 Legende hinzufügen (erklärt die Farben)
    ax.legend(loc='upper right', fontsize=10)
    
    # 🔲 Gitter im Hintergrund für bessere Orientierung
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 🎨 Transparenter Hintergrund
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # ===============================================
    # 💾 SCHRITT 7: BILD SPEICHERN (OPTIONAL)
    # ===============================================
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"💾 Bild gespeichert als: {output_path}")
    
    # ===============================================
    # 👀 SCHRITT 8: BILD ANZEIGEN
    # ===============================================
    if show_plot:
        print("👀 Zeige 3D-Grafik... (Fenster schließen um fortzufahren)")
        plt.show()
    else:
        plt.close()
    
    print("✅ Fertig!")
    return fig, ax


# ===============================================
# 🛠️ HILFSFUNKTION: SKELETT-LINIEN ZEICHNEN
# ===============================================
def _plot_skeleton_3d(
    ax,           # 📊 Das 3D-Zeichenfeld
    keypoints,    # 📍 Alle Punkte einer Person
    scores,       # 🎯 Wie sicher sind die Punkte?
    connections,  # ↔️ Welche Punkte sollen verbunden werden?
    color='blue', linewidth=2, alpha=1.0, label=None, threshold=0.3
):
    """
    🛠️ INTERNE HILFSFUNKTION
    Zeichnet Linien zwischen Körperpunkten.
    
    WICHTIG: Filtert automatisch:
    1. Unsichere Punkte (Genauigkeit zu niedrig)
    2. Fehlende Punkte (Null-Koordinaten)
    """
    for i, (start_idx, end_idx) in enumerate(connections):
        # 🚫 Prüfen ob Punkt-Indizes existieren
        if start_idx >= len(keypoints) or end_idx >= len(keypoints):
            continue  # ⏭️ Überspringen
        
        # 🎯 Prüfen ob beide Punkte sicher genug sind
        if scores[start_idx] <= threshold or scores[end_idx] <= threshold:
            continue  # ⏭️ Überspringen wenn unsicher
        
        # 📍 Koordinaten der beiden Punkte holen
        start = keypoints[start_idx]  # [X, Y, Z] vom Startpunkt
        end = keypoints[end_idx]      # [X, Y, Z] vom Endpunkt
        
        # 🚫 Prüfen auf Null-Koordinaten (fehlende Daten)
        if np.all(start == 0) or np.all(end == 0):
            continue  # ⏭️ Überspringen
        
        # 🎨 Linie zwischen den Punkten zeichnen
        ax.plot(
            [start[0], end[0]],  # X-Koordinaten
            [start[1], end[1]],  # Y-Koordinaten
            [start[2], end[2]],  # Z-Koordinaten (Tiefe)
            color=color, linewidth=linewidth, alpha=alpha,
            label=label if i == 0 else None  # Beschriftung nur für erste Linie
        )


# ===============================================
# 📊 FUNKTION FÜR MEHRERE KAMERAANSICHTEN
# ===============================================
def plot_multiple_views(
    json_path: Union[str, Path],
    frame_idx: int = 0,
    output_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
    z_scale: float = 1.0
):
    """
    📽️ Zeigt 3 verschiedene Blickwinkel nebeneinander:
    
    1. 📷 Linke Kamera
    2. 📷 Rechte Kamera  
    3. 🎯 Kombinierte (beste) Ansicht
    
    Perfekt zum Vergleichen verschiedener Perspektiven!
    """
    print("📽️ Erstelle Multi-View Vergleich...")
    
    # 🖼️ Neue Grafik mit 3 Bildern nebeneinander
    fig = plt.figure(figsize=(20, 6))
    
    # 🎬 Definition der drei Ansichten
    views = ['left_3d', 'right_3d', 'combined_3d']
    titles = ['👈 Linke Ansicht', '👉 Rechte Ansicht', '🎯 Beste Kombination']
    
    # 📥 Daten laden
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frame_data = data[frame_idx]
    
    # 🔄 Für jede der drei Ansichten...
    for idx, (view, title) in enumerate(zip(views, titles)):
        # 📐 3D-Diagramm erstellen (Position 1, 2 oder 3)
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        
        view_data = frame_data.get(view)
        
        if view_data is None:
            # 📝 Falls keine Daten: Fehlermeldung anzeigen
            ax.text(0.5, 0.5, 0.5, f"Keine Daten für\n{view}", 
                   ha='center', va='center', fontsize=14)
            continue
        
        # 📊 3D-Punkte extrahieren
        keypoints_3d = np.array(view_data['keypoints_3d'])
        scores_3d = np.array(view_data['scores_3d'])
        
        if z_scale != 1.0:
            keypoints_3d[:, :, 2] *= z_scale
        
        # 👤 Für jede Person...
        for person_idx in range(len(keypoints_3d)):
            kpts = keypoints_3d[person_idx]
            scores = scores_3d[person_idx]
            
            # 🦴 Körper-Skelett zeichnen
            _plot_skeleton_3d(ax, kpts, scores, BODY_CONNECTIONS, 
                             color='blue', linewidth=2)
            
            # ⚫ Punkte zeichnen
            valid_mask = scores > 0.3
            valid_kpts = kpts[valid_mask]
            if len(valid_kpts) > 0:
                non_zero = ~np.all(valid_kpts == 0, axis=1)
                valid_kpts = valid_kpts[non_zero]
                if len(valid_kpts) > 0:
                    ax.scatter(valid_kpts[:, 0], valid_kpts[:, 1], valid_kpts[:, 2],
                              c='black', marker='o', s=25, alpha=0.6)
        
        # 📝 Beschriftungen
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title, fontsize=12, pad=10)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
    
    # 🏷️ Gesamttitel
    plt.suptitle(f'3D Körperhaltung - Moment {frame_idx}', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # 💾 Speichern
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 Multi-View gespeichert: {output_path}")
    
    # 👀 Anzeigen
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


# ===============================================
# 🎬 FUNKTION FÜR ANIMATIONS-ERSTELLUNG
# ===============================================
def create_3d_animation_frames(
    json_path: Union[str, Path],
    output_dir: Union[str, Path],
    view: str = 'combined_3d',
    max_frames: Optional[int] = None,
    z_scale: float = 1.0,
    show_hands: bool = False,
    show_face: bool = False
):
    """
    🎞️ Erstellt viele Einzelbilder für eine Animation
    
    Denke an ein Daumenkino: Viele Bilder hintereinander ergeben Bewegung!
    
    Verwendung:
    create_3d_animation_frames("daten.json", "meine_animation")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # 📁 Ordner erstellen
    
    # 📥 Daten laden
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 🔢 Anzahl der Frames bestimmen
    num_frames = min(len(data), max_frames) if max_frames else len(data)
    
    print(f"🎞️ Erstelle {num_frames} Animations-Frames...")
    print(f"📁 Speichere in: {output_dir}")
    
    # 🔄 Für jeden Frame...
    for frame_idx in range(num_frames):
        # 📝 Dateinamen erstellen (z.B. frame_00001.png, frame_00002.png, ...)
        output_path = output_dir / f"frame_{frame_idx:05d}.png"
        
        # 🖼️ Bild für diesen Frame erstellen
        plot_3d_pose_from_json(
            json_path,
            frame_idx=frame_idx,
            view=view,
            output_path=output_path,
            show_plot=False,  # ❌ Nicht anzeigen, nur speichern
            z_scale=z_scale,
            show_hands=show_hands,
            show_face=show_face
        )
        
        # 📊 Fortschrittsanzeige
        if frame_idx % 10 == 0:
            print(f"  📊 {frame_idx}/{num_frames} Frames fertig")
    
    print(f"✅ Alle {num_frames} Frames gespeichert!")
    print(f"💡 Tipp: Verwende diese Bilder zum Erstellen eines Videos.")


# ===============================================
# 🚀 START: WENN DAS PROGRAMM DIREKT GESTARTET WIRD
# ===============================================
if __name__ == "__main__":
    print("=" * 70)
    print("🎯 3D Pose Visualizer - Aktualisierte Version")
    print("=" * 70)
    print("📝 Visualisiert Körperhaltungen in 3D mit korrekten Verbindungen")
    print("")
    
    # 🔍 Test-Datei suchen
    test_json = Path("poses_3d_filtered.json")
    
    if test_json.exists():
        print(f"✅ Test-Datei gefunden: {test_json}")
        print("")
        
        # ===============================================
        # 🎯 BEISPIEL 1: EINZELNES BILD
        # ===============================================
        print("1️⃣  Beispiel 1: Einzelnes Bild mit korrekten Verbindungen")
        print("   (Schau dir die korrigierten Schulter-Linien an!)")
        print("")
        
        plot_3d_pose_from_json(
            test_json,
            frame_idx=0,           # 🎞️ Erstes Bild
            view='combined_3d',    # 📷 Beste Kameraperspektive
            output_path="pose_3d_corrected.png",  # 💾 Speichern
            show_plot=True,        # 👀 Anzeigen
            z_scale=5.0,           # 🔍 Tiefe 5x vergrößern
            show_hands=True,       # ✋ Hände zeigen
            show_face=True         # 😀 Gesicht zeigen
        )
        
        print("")
        print("=" * 50)
        print("")
        
        # ===============================================
        # 📽️ BEISPIEL 2: 3 ANSICHTEN NEBENEINANDER
        # ===============================================
        print("2️⃣  Beispiel 2: Drei Kameraperspektiven vergleichen")
        print("   (Linke Kamera, Rechte Kamera, Kombinierte Ansicht)")
        print("")
        
        plot_multiple_views(
            test_json,
            frame_idx=0,
            output_path="pose_3d_multiview.png",
            show_plot=True,
            z_scale=5.0
        )
        
        print("")
        print("=" * 50)
        print("")
        print("🎉 Alles fertig! Du kannst jetzt:")
        print("   1. Die Bilder im Ordner finden")
        print("   2. Andere Frames ausprobieren (frame_idx=1, 2, ...)")
        print("   3. Die Tiefenskala anpassen (z_scale=3.0, 10.0, ...)")
        
    else:
        # ❌ Falls keine Test-Datei gefunden wurde
        print(f"⚠️  Keine Test-Daten gefunden: {test_json}")
        print("")
        print("ℹ️  So bekommst du Test-Daten:")
        print("   1. Stelle sicher, dass 'poses_3d_filtered.json' im gleichen Ordner liegt")
        print("   2. Oder ändere 'test_json' auf deine Datei")
        print("")
        print("💡 Du kannst trotzdem die Funktionen nutzen:")
        print("   plot_3d_pose_from_json('deine_datei.json', frame_idx=0)")