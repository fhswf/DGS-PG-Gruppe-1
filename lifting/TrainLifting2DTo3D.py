import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Modelldefinition (wie vorgegeben)
class model_A_simple_yet_effective_baseline_for_3d_human_pose_estimation(nn.Module):
    def __init__(self):
        super(model_A_simple_yet_effective_baseline_for_3d_human_pose_estimation, self).__init__()
        self.upscale = nn.Linear(133*2, 1024)
        self.fc1 = nn.Linear(1024, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.outputlayer = nn.Linear(1024, 133*3)

    def forward(self, x):
        x = self.upscale(x)
        x1 = nn.Dropout(p=0.5)(nn.ReLU()(self.bn1(self.fc1(x))))
        x1 = nn.Dropout(p=0.5)(nn.ReLU()(self.bn2(self.fc2(x1))))
        x = x + x1
        x1 = nn.Dropout(p=0.5)(nn.ReLU()(self.bn3(self.fc3(x))))
        x1 = nn.Dropout(p=0.5)(nn.ReLU()(self.bn4(self.fc4(x1))))
        x = x + x1
        x = self.outputlayer(x)
        return x

# Dataset Klasse für Ihre Daten
class KeypointsDataset(Dataset):
    def __init__(self, data_dict):
        self.data = []
        
        # Daten aus dem JSON-Format extrahieren
        for frame_id, frame_data in data_dict.items():
            keypoints_2d = frame_data['keypoints_2d']
            keypoints_3d = frame_data['keypoints_3d']
            
            # 2D Keypoints extrahieren und in einen Vektor umwandeln
            kp2d_list = []
            for i in range(133):  # 133 Keypoints
                if str(i) in keypoints_2d:
                    kp2d_list.append(keypoints_2d[str(i)]['x'])
                    kp2d_list.append(keypoints_2d[str(i)]['y'])
                else:
                    # Falls Keypoints fehlen, mit 0 füllen
                    kp2d_list.extend([0.0, 0.0])
            
            # 3D Keypoints extrahieren und in einen Vektor umwandeln
            kp3d_list = []
            for i in range(133):  # 133 Keypoints
                if str(i) in keypoints_3d:
                    kp3d_list.append(keypoints_3d[str(i)]['x'])
                    kp3d_list.append(keypoints_3d[str(i)]['y'])
                    kp3d_list.append(keypoints_3d[str(i)]['z'])
                else:
                    # Falls Keypoints fehlen, mit 0 füllen
                    kp3d_list.extend([0.0, 0.0, 0.0])
            
            self.data.append({
                'kp2d': np.array(kp2d_list, dtype=np.float32),
                'kp3d': np.array(kp3d_list, dtype=np.float32)
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['kp2d'], self.data[idx]['kp3d']

# Haupttrainingsfunktion
def train_model(data_dict, epochs=100, batch_size=2, learning_rate=0.001):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset und DataLoader erstellen
    dataset = KeypointsDataset(data_dict)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Modell initialisieren
    model = model_A_simple_yet_effective_baseline_for_3d_human_pose_estimation().to(device)
    
    # Falls vortrainiertes Modell existiert, laden
    if os.path.exists('net.pth'):
        model.load_state_dict(torch.load('net.pth', map_location=device))
        print('Pretrained weights loaded successfully')
    
    # Loss-Funktion und Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training Loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (inputs_2d, targets_3d) in enumerate(dataloader):
            inputs_2d = inputs_2d.to(device)
            targets_3d = targets_3d.to(device)
            
            # Forward pass
            outputs_3d = model(inputs_2d)
            
            # Loss berechnen
            loss = criterion(outputs_3d, targets_3d)
            
            # Backward pass und Optimierung
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Alle 10 Batches Fortschritt anzeigen
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.6f}')
        
        # Durchschnittlicher Loss für die Epoche
        avg_loss = total_loss / num_batches
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.6f}')
        
        # Learning Rate anpassen basierend auf dem Loss
        scheduler.step(avg_loss)
        
        # Modell speichern (jede 10. Epoche)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'net_epoch_{epoch+1}.pth')
            print(f'Model saved: net_epoch_{epoch+1}.pth')
    
    # Finales Modell speichern
    torch.save(model.state_dict(), 'net_final.pth')
    print('Training completed! Final model saved as net_final.pth')
    
    return model

# Evaluierungsfunktion
def evaluate_model(model, data_dict, device):
    model.eval()
    dataset = KeypointsDataset(data_dict)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    criterion = nn.MSELoss()
    total_loss = 0
    num_samples = 0
    
    with torch.no_grad():
        for inputs_2d, targets_3d in dataloader:
            inputs_2d = inputs_2d.to(device)
            targets_3d = targets_3d.to(device)
            
            outputs_3d = model(inputs_2d)
            loss = criterion(outputs_3d, targets_3d)
            
            total_loss += loss.item()
            num_samples += 1
    
    avg_loss = total_loss / num_samples
    print(f'Evaluation Loss: {avg_loss:.6f}')
    
    return avg_loss


if __name__ == "__main__":
    # Daten laden
    with open('2Dto3D_train.json', 'r') as f:
        train_data=json.load(f)

    with open('2Dto3D_test_2d.json', 'r') as f:
        test_data=json.load(f)

    # Trainingsparameter
    EPOCHS=50
    BATCH_SIZE=32
    LEARNING_RATE=0.001

    # Training starten
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=train_model(train_data, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)

    # Modell auf Trainingsdaten evaluieren
    print("\nEvaluating model on training data...")
    evaluate_model(model, train_data, device)

    # Modell auf Testdaten evaluieren
    print("\nEvaluating model on test data...")
    evaluate_model(model, test_data, device)
