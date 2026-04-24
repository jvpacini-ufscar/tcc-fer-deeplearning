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

# 1. Pipeline de Dados Ajustado (Mais gentil com imagens 48x48)
transform = {
    'train': transforms.Compose([
        transforms.Resize((240, 240)), # Um pouco maior para o crop
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform['train'])
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform['val'])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 2. Pesos de Classe
targets = train_dataset.targets
weights = class_weight.compute_class_weight('balanced', classes=np.unique(targets), y=targets)
class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

# 3. Model Building
def get_resnet50v2():
    # Usando a resnetv2_50 (pre-activation)
    base_model = timm.create_model('resnetv2_50', pretrained=True, num_classes=0)
    
    # Cabeça customizada um pouco mais robusta
    model = nn.Sequential(
        base_model,
        nn.Flatten(),
        nn.Linear(2048, 512), # Aumentando para 512 neurônios
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(512, NUM_CLASSES)
    )
    return model.to(DEVICE)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, phase_name):
    best_acc = 0
    history = []
    for epoch in range(num_epochs):
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"{phase_name} - Epoch {epoch+1}/{num_epochs}", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item() * inputs.size(0)
            _, pred = outputs.max(1)
            t_total += labels.size(0)
            t_correct += pred.eq(labels).sum().item()
            pbar.set_postfix({'acc': t_correct/t_total})
        
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * inputs.size(0)
                _, pred = outputs.max(1)
                v_total += labels.size(0)
                v_correct += pred.eq(labels).sum().item()
        
        train_acc = t_correct / t_total
        val_acc = v_correct / v_total
        print(f"[{phase_name}] Epoch {epoch+1}: Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.7f}")
        
        if scheduler:
            scheduler.step(v_loss / v_total)
            
        history.append({'epoch': epoch, 'phase': phase_name, 't_loss': t_loss/t_total, 't_acc': train_acc, 'v_loss': v_loss/v_total, 'v_acc': val_acc})
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "resnet50v2_pytorch_best.pth"))
            
    return history, best_acc

def main():
    print(f"[INFO] Treinando ResNet50V2 (Ataque Total) na GPU: {torch.cuda.get_device_name(0)}")
    model = get_resnet50v2()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # FASE 1: Warmup (Congela base)
    print("\n--- FASE 1: WARMUP ---")
    for param in model[0].parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 10, "Warmup")
    
    # FASE 2: Fine-tuning (Libera TUDO)
    print("\n--- FASE 2: FINE-TUNING TOTAL ---")
    for param in model.parameters():
        param.requires_grad = True
                
    optimizer = optim.Adam(model.parameters(), lr=5e-5) # LR um pouco maior para Fine-tuning total
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    history, best_v_acc = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 30, "Fine-tuning")
    
    # Salvar Relatórios
    pd.DataFrame(history).to_csv(os.path.join(REPORTS_DIR, "historico_resnet50v2_pytorch.csv"), index=False)
    
    # Matriz de Confusão Final
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "resnet50v2_pytorch_best.pth")))
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
    pd.DataFrame(report).transpose().to_csv(os.path.join(REPORTS_DIR, "metrics_resnet50v2_pytorch.csv"))
    
    print(f"\n[FINALIZADO] ResNet50V2 PyTorch Best Acc: {best_v_acc:.4f}")

if __name__ == "__main__":
    main()
