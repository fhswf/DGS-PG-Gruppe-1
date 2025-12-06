"""
🎓 Training-Skript für das 3D Pose Estimation Modell
ANGEPASST FÜR IHRE DATEN

Dieses Skript trainiert ein neuronales Netz, das aus 2D-Posen (z.B. von OpenPose)
3D-Posen rekonstruiert. Es verwendet einen zweistufigen Trainingsansatz.
"""

from pose_estimator_3d import train_new_model
from pathlib import Path
import sys

# ===============================================
# VISUELLE TRENNUNG IN DER KONSOLE
# ===============================================
print("=" * 80)
print("🎓 TRAINING: 3D Pose Estimation Modell")
print("=" * 80)
print()

# ===============================================
# ⚙️ KONFIGURATION - ANGEPASST FÜR IHRE DATEIEN
# ===============================================

# 📁 OPTION B: Mehrere Dateien (inkrementelles Training)
# Diese Variable steuert, ob eine einzelne Datei oder mehrere Dateien verwendet werden
SINGLE_FILE_MODE = False  # Wichtig: Auf False setzen für inkrementelles Training!

# Ihre Trainingsdateien - Liste der JSON-Dateien, die die Trainingsdaten enthalten
# Jede Datei enthält 2D→3D Pose-Paare für das Training
TRAIN_DATA = [
    '2Dto3D_train_part1.json',  # Teil 1 des Datasets
    '2Dto3D_train_part2.json',  # Teil 2 des Datasets
    '2Dto3D_train_part3.json',  # Teil 3 des Datasets
    '2Dto3D_train_part4.json',  # Teil 4 des Datasets
    '2Dto3D_train_part5.json'   # Teil 5 des Datasets
]

# Hauptdatei für Final-Training - Enthält alle Daten kombiniert
MAIN_TRAIN_DATA = '2Dto3D_train.json'

# Testdaten - Wird auf None gesetzt, da automatisch gesplittet wird
TEST_DATA = None  # Das Skript teilt automatisch 20% der Daten für Tests ab

# 🎯 Training-Parameter
OUTPUT_MODEL = 'lifting2DTo3D.pth'  # Dateiname des trainierten Modells
EPOCHS = 75                          # Anzahl der Trainingsdurchläufe pro Phase
LEARNING_RATE = 0.002               # Schrittweite der Gradientenabstiegsoptimierung
BATCH_SIZE = 128                    # Anzahl der Samples pro Optimierungsschritt

# ===============================================
# 🔍 PRE-FLIGHT CHECK - DATEIEXISTENZ PRÜFEN
# ===============================================
print("🔍 Überprüfe Dateien...")

def check_file(path):
    """
    Prüft, ob eine Datei oder Liste von Dateien existiert
    
    Args:
        path: String mit Dateipfad oder Liste von Dateipfaden
    
    Returns:
        bool: True wenn alle Dateien existieren, sonst False
    """
    if isinstance(path, list):
        # Prüfe jede Datei in der Liste
        for p in path:
            if not Path(p).exists():
                print(f"❌ Datei nicht gefunden: {p}")
                return False
        return True
    else:
        # Prüfe einzelne Datei (nur wenn nicht None)
        if path and not Path(path).exists():
            print(f"❌ Datei nicht gefunden: {path}")
            return False
        return True

# Kopiere die Trainingsdatenliste und füge die Hauptdatei hinzu
all_files = TRAIN_DATA.copy()
all_files.append(MAIN_TRAIN_DATA)

# Prüfe ob alle benötigten Dateien existieren
if not check_file(all_files):
    print("\n⚠️  FEHLER: Training-Daten nicht gefunden!")
    print("   Stelle sicher, dass alle JSON-Dateien im aktuellen Verzeichnis sind")
    print("   Aktuelles Verzeichnis:", Path.cwd())
    exit(1)  # Beende das Skript mit Fehlercode 1

print("✅ Alle Dateien gefunden!")

# ===============================================
# 📊 KONFIGURATION ANZEIGEN
# ===============================================
print("\n" + "="*80)
print("📋 TRAINING-KONFIGURATION")
print("="*80)

# Erkläre den zweistufigen Trainingsansatz
print("📁 Training in zwei Phasen:")
print("   1. Inkrementelles Training auf 5 Teilen")
print("   2. Final-Training auf vollständigem Dataset")

print(f"\n📁 Inkrementelle Dateien ({len(TRAIN_DATA)} Teile):")
for i, f in enumerate(TRAIN_DATA, 1):
    print(f"   {i}. {f}")

print(f"\n📁 Final-Training Datei: {MAIN_TRAIN_DATA}")
print(f"📊 Split: 80% Training / 20% Test (automatisch)")

print(f"\n⚙️  Parameter:")
print(f"   🎯 Epochen pro Teil: {EPOCHS}")
print(f"   📦 Batch-Größe: {BATCH_SIZE}")
print(f"   📈 Lernrate: {LEARNING_RATE}")
print(f"   💾 Output: {OUTPUT_MODEL}")

# ===============================================
# 🚀 TRAINING-PROZESS STARTEN
# ===============================================
print("\n" + "="*80)
print("🚀 STARTE TRAINING")
print("="*80)
print("\n💡 HINWEIS: Das Training erfolgt in zwei Phasen...")
print()

# Warte auf Benutzerbestätigung bevor das Training startet
input("Drücke ENTER zum Starten oder STRG+C zum Abbrechen...")

try:
    # ===============================================
    # 📦 PHASE 1: INKREMENTELLES TRAINING AUF TEILEN
    # ===============================================
    print("\n" + "="*60)
    print("📦 PHASE 1: Inkrementelles Training auf 5 Teilen")
    print("="*60)
    
    print("🤖 Initialisiere Modell...")
    
    # Importiere die Funktion für inkrementelles Training
    from pose_estimator_3d import train_on_h3wb_incremental
    
    # Starte das inkrementelle Training
    model = train_on_h3wb_incremental(
        train_json_files=TRAIN_DATA,  # Liste der Teil-Datasets
        test_json=None,               # Automatischer Split aus Trainingsdaten
        epochs=EPOCHS,                # Epochen pro Teil-Dataset
        batch_size=BATCH_SIZE,        # Batch-Größe
        learning_rate=LEARNING_RATE,  # Anfängliche Lernrate
        output_model=OUTPUT_MODEL,    # Wo das Modell gespeichert wird
        train_split=0.8,              # 80% Training, 20% Test
        checkpoint_interval=10        # Speichert Modell alle 10 Epochen
    )
    
    print("✅ Phase 1 abgeschlossen!")
    
    # ===============================================
    # 🏆 PHASE 2: FINAL-TRAINING AUF VOLLEM DATASET
    # ===============================================
    print("\n" + "="*60)
    print("🏆 PHASE 2: Final-Training auf vollständigem Dataset")
    print("="*60)
    
    # Importiere die Funktion für Final-Training
    from pose_estimator_3d import train_on_h3wb
    
    print(f"📂 Verwende vollständiges Dataset: {MAIN_TRAIN_DATA}")
    
    # Finales Training mit reduzierter Lernrate für Feintuning
    model = train_on_h3wb(
        train_json=MAIN_TRAIN_DATA,       # Vollständiges Dataset
        test_json=None,                   # Automatischer Split
        epochs=EPOCHS,                    # Weitere Epochen
        batch_size=BATCH_SIZE,            # Gleiche Batch-Größe
        learning_rate=LEARNING_RATE * 0.5,# Reduzierte Lernrate für Feintuning
        output_model=OUTPUT_MODEL,        # Überschreibt das vorherige Modell
        train_split=0.8                   # 80% Training, 20% Test
    )
    
    # ===============================================
    # 🎉 TRAINING ERFOLGREICH ABGESCHLOSSEN
    # ===============================================
    print("\n" + "="*80)
    print("🎉 TRAINING ERFOLGREICH ABGESCHLOSSEN!")
    print("="*80)
    print(f"\n✅ Trainiertes Modell gespeichert: {OUTPUT_MODEL}")
    print(f"\n📊 Zusammenfassung:")
    print(f"   📁 {len(TRAIN_DATA)} Teil-Datasets verarbeitet")
    print(f"   📊 Vollständiges Dataset: {MAIN_TRAIN_DATA}")
    print(f"   ⏱️  Gesamt-Epochen: {EPOCHS * (len(TRAIN_DATA) + 1)}")
    print(f"\n🚀 Nächste Schritte:")
    print(f"   1. Teste das Modell: python rtmtest.py")
    print(f"   2. Überprüfe die Ergebnisse in poses_3d_mlp.json")
    print(f"   3. Vergleiche mit geometric Methode")
    
# ===============================================
# 🛑 FEHLERBEHANDLUNG
# ===============================================
except KeyboardInterrupt:
    # Wird aufgerufen wenn der Benutzer STRG+C drückt
    print("\n\n⚠️  Training abgebrochen!")
    print("   Checkpoints wurden gespeichert und können fortgesetzt werden")
    
except Exception as e:
    # Allgemeine Fehlerbehandlung für unerwartete Fehler
    print(f"\n\n❌ FEHLER beim Training:")
    print(f"   {str(e)}")
    
    # Zeige detaillierten Stack-Trace für Debugging
    import traceback
    traceback.print_exc()
    
    # Gebe dem Benutzer hilfreiche Lösungsvorschläge
    print("\n🔧 Mögliche Lösungen:")
    print("   - Reduziere BATCH_SIZE weiter (z.B. auf 64)")
    print("   - Stelle sicher, dass JSON-Dateien korrektes Format haben")
    print("   - Prüfe ob genug RAM/VRAM vorhanden ist")

# ===============================================
# ENDE DES SKRIPTS
# ===============================================
print("\n" + "="*80)