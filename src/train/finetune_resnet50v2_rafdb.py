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
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configurações de Hardware e Caminhos
BASE_DIR = "."
DATA_DIR = os.path.join(BASE_DIR, "data/raw/raf-db")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "test")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
PRETRAINED_PATH = os.path.join(MODELS_DIR, "resnet50v2_pytorch_best.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64 # Ajustado para ataque total
NUM_CLASSES = 7
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 1. Pipeline de Dados (Ataque Total com Augmentation)
transform = {
    'train': transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

# 2. Carregamento do Dataset
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform['train'])
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform['val'])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# 3. Pesos de Classe para o RAF-DB
targets = train_dataset.targets
weights = class_weight.compute_class_weight('balanced', classes=np.unique(targets), y=targets)
class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

# 4. Model Building (Carregando do FER2013)
def get_model_finetune():
    base_model = timm.create_model('resnetv2_50', pretrained=False, num_classes=0)
    model = nn.Sequential(
        base_model,
        nn.Flatten(),
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(512, NUM_CLASSES)
    )
    
    # Carregar pesos pré-treinados no FER2013
    if os.path.exists(PRETRAINED_PATH):
        print(f"[INFO] Carregando pesos pré-treinados do FER2013: {PRETRAINED_PATH}")
        model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=DEVICE))
    else:
        print("[AVISO] Pesos do FER2013 não encontrados. Começando do ImageNet.")
        # Se não achar o nosso, usa o do timm
        model[0] = timm.create_model('resnetv2_50', pretrained=True, num_classes=0)
        
    return model.to(DEVICE)

def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Mixed Precision para extrair performance do hardware
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * inputs.size(0)
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
        
    return total_loss/total, correct/total

def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return total_loss/total, correct/total

def main():
    print(f"\n[ATAQUE TOTAL] Fine-tuning ResNet50V2 no RAF-DB")
    model = get_model_finetune()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.cuda.amp.GradScaler()
    
    # FASE 1: Congela BackBone (Warmup)
    print("\n--- FASE 1: Warmup Classifier (5 epochs) ---")
    for param in model[0].parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(5):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        v_loss, v_acc = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}: Train Acc {t_acc:.4f} | Val Acc {v_acc:.4f}")

    # FASE 2: Unfreeze All (Ataque Total)
    print("\n--- FASE 2: Fine-tuning Total (30 epochs) ---")
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_acc = 0
    history = []
    
    for epoch in range(30):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        v_loss, v_acc = validate(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1}: Train Acc {t_acc:.4f} | Val Acc {v_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.7f}")
        
        scheduler.step(v_loss)
        history.append({'epoch': epoch, 't_acc': t_acc, 'v_acc': v_acc, 't_loss': t_loss, 'v_loss': v_loss})
        
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "resnet50v2_rafdb_best.pth"))
            
    # Relatório Final
    print(f"\n[FINALIZADO] Melhor Acurácia no RAF-DB: {best_acc:.4f}")
    
    # Gerar Métricas Detalhadas
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "resnet50v2_rafdb_best.pth")))
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
    pd.DataFrame(report).transpose().to_csv(os.path.join(REPORTS_DIR, "metrics_resnet50v2_rafdb.csv"))
    
    # Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES)
    plt.title('Matriz de Confusão Fine-tuned: RAF-DB (ResNet50V2)')
    plt.savefig(os.path.join(FIGURES_DIR, "cm_resnet50v2_rafdb.png"))
    plt.close()

if __name__ == "__main__":
    main()
