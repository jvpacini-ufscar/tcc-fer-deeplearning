import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import pandas as pd
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configurações
BASE_DIR = "/mnt/c/Users/jvpac/Videos/tcc-fer-deeplearning"
DATA_DIR = os.path.join(BASE_DIR, "data/raw/fer2013")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "test")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_CLASSES = 7
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 1. Data Augmentation & Loading
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 2. Class Weights
targets = train_dataset.targets
weights = class_weight.compute_class_weight('balanced', classes=np.unique(targets), y=targets)
class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

# 3. Model Building (EfficientNetB0)
def get_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    # Freeze base
    for param in model.parameters():
        param.requires_grad = False
    
    # Custom head
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(128, NUM_CLASSES)
    )
    return model.to(DEVICE)

# 4. Training Function
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total

# 5. Main Execution
def main():
    print(f"[INFO] Treinando na GPU: {torch.cuda.get_device_name(0)}")
    model = get_model()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # FASE 1: Warmup (5 épocas)
    print("\n--- FASE 1: Warmup (Top Classifier) ---")
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    history = []
    
    for epoch in range(5):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/5 | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}")
        history.append({'epoch': epoch, 'phase': 1, 't_loss': t_loss, 't_acc': t_acc, 'v_loss': v_loss, 'v_acc': v_acc})

    # FASE 2: Fine-tuning (15 épocas)
    print("\n--- FASE 2: Fine-tuning (Descongelando últimas camadas) ---")
    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True
    
    # Optimizer com LR menor para Fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    best_acc = 0
    for epoch in range(15):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/15 | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}")
        history.append({'epoch': epoch+5, 'phase': 2, 't_loss': t_loss, 't_acc': t_acc, 'v_loss': v_loss, 'v_acc': v_acc})
        
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "efficientnetb0_pytorch_best.pth"))

    # 6. Evaluation Final
    print("\n[INFO] Avaliação Final...")
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "efficientnetb0_pytorch_best.pth")))
    model.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.cpu().tolist())

    # Métricas
    report = classification_report(y_true, y_pred, target_names=EMOTION_CLASSES, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(REPORTS_DIR, "metrics_efficientnetb0_pytorch.csv"))
    
    # CM
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Greens', xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES)
    plt.savefig(os.path.join(FIGURES_DIR, "cm_efficientnetb0_pytorch.png"), dpi=300)
    
    # Histórico
    pd.DataFrame(history).to_csv(os.path.join(REPORTS_DIR, "historico_efficientnetb0_pytorch.csv"), index=False)
    
    print(f"\n[SUCESSO] Treinamento concluído. Melhor Acurácia Val: {best_acc:.4f}")

if __name__ == "__main__":
    main()
