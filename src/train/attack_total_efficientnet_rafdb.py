import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import os
import pandas as pd
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configurações
BASE_DIR = "."
DATA_DIR = os.path.join(BASE_DIR, "data/raw/raf-db")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "test")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32 # EfficientNet consome mais memória de ativação que a ResNet50
NUM_CLASSES = 7
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 1. Pipeline de Dados (Otimizado para EfficientNet)
# EfficientNetB0 usa imagens 224x224 por padrão
transform = {
    'train': transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalização ImageNet
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform['train'])
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform['val'])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# 2. Pesos de Classe
targets = train_dataset.targets
weights = class_weight.compute_class_weight('balanced', classes=np.unique(targets), y=targets)
class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

# 3. Model Building
def get_efficientnet():
    # Usando efficientnet_b0
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES)
    return model.to(DEVICE)

def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    t_loss, t_correct, t_total = 0, 0, 0
    pbar = tqdm(loader, leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        t_loss += loss.item() * inputs.size(0)
        _, pred = outputs.max(1)
        t_total += labels.size(0)
        t_correct += pred.eq(labels).sum().item()
        pbar.set_postfix({'acc': t_correct/t_total})
    return t_loss/t_total, t_correct/t_total

def validate(model, loader, criterion):
    model.eval()
    v_loss, v_correct, v_total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            v_loss += loss.item() * inputs.size(0)
            _, pred = outputs.max(1)
            v_total += labels.size(0)
            v_correct += pred.eq(labels).sum().item()
    return v_loss/v_total, v_correct/v_total

def main():
    print(f"\n[ATAQUE TOTAL] Treinando EfficientNetB0 no RAF-DB")
    model = get_efficientnet()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.cuda.amp.GradScaler()
    
    # FASE 1: Warmup Classifier
    print("\n--- FASE 1: WARMUP (5 epochs) ---")
    # Congela tudo exceto a última camada
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    for epoch in range(5):
        _, t_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        _, v_acc = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}: Train Acc {t_acc:.4f} | Val Acc {v_acc:.4f}")
        
    # FASE 2: Fine-tuning Total
    print("\n--- FASE 2: FINE-TUNING TOTAL (30 epochs) ---")
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # LR um pouco maior para FT de EfficientNet
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_acc = 0
    for epoch in range(30):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        v_loss, v_acc = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}: Train Acc {t_acc:.4f} | Val Acc {v_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.7f}")
        
        scheduler.step(v_loss)
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "efficientnetb0_rafdb_best.pth"))
            
    print(f"\n[FINALIZADO] Melhor Acurácia EfficientNetB0: {best_acc:.4f}")
    
    # Métricas Finais
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "efficientnetb0_rafdb_best.pth")))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.cpu().tolist())
            
    report = classification_report(y_true, y_pred, target_names=EMOTION_CLASSES, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(REPORTS_DIR, "metrics_efficientnetb0_rafdb.csv"))
    
    # Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Greens', xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES)
    plt.title('Matriz de Confusão EfficientNetB0 - RAF-DB')
    plt.savefig(os.path.join(FIGURES_DIR, "cm_efficientnetb0_rafdb.png"))
    plt.close()

if __name__ == "__main__":
    main()
