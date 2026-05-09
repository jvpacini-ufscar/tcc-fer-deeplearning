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
DATA_DIR = os.path.join(BASE_DIR, "data/raw/fer2013")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "test")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_CLASSES = 7
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 1. Pipeline de Dados
transform = {
    'train': transforms.Compose([
        transforms.Resize((232, 232)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform['train'])
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform['val'])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# 2. Pesos de Classe (FER2013 é muito desbalanceado, especialmente 'disgust')
targets = train_dataset.targets
weights = class_weight.compute_class_weight('balanced', classes=np.unique(targets), y=targets)
class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

# 3. Model Building
def get_convnext():
    model = timm.create_model('convnext_tiny', pretrained=True, num_classes=NUM_CLASSES)
    return model.to(DEVICE)

def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    t_loss, t_correct, t_total = 0, 0, 0
    pbar = tqdm(loader, desc="Training", leave=False)
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
    print(f"\n[SISTEMÁTICO] Treinando ConvNeXt Tiny no FER2013")
    model = get_convnext()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.cuda.amp.GradScaler()
    
    # FASE 1: Warmup Classifier (5 epochs)
    print("\n--- FASE 1: WARMUP CLASSIFIER ---")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
        
    optimizer = optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=0.05)
    for epoch in range(5):
        _, t_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        _, v_acc = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}: Train Acc {t_acc:.4f} | Val Acc {v_acc:.4f}")
        
    # FASE 2: Fine-tuning Total (30 epochs)
    print("\n--- FASE 2: FINE-TUNING TOTAL ---")
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    best_acc = 0
    for epoch in range(30):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        v_loss, v_acc = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}: Train Acc {t_acc:.4f} | Val Acc {v_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.7f}")
        
        scheduler.step()
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "convnext_fer2013_best.pth"))
            
    print(f"\n[FINALIZADO] Melhor Acurácia ConvNeXt no FER2013: {best_acc:.4f}")
    
    # Métricas Finais
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "convnext_fer2013_best.pth")))
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
    pd.DataFrame(report).transpose().to_csv(os.path.join(REPORTS_DIR, "metrics_convnext_fer2013.csv"))
    
    # Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Purples', xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES)
    plt.title('Matriz de Confusão ConvNeXt Tiny - FER2013')
    plt.savefig(os.path.join(FIGURES_DIR, "cm_convnext_fer2013.png"))
    plt.close()

if __name__ == "__main__":
    main()
