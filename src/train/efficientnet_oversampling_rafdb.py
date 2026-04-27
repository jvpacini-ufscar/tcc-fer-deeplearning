import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
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
BATCH_SIZE = 32
NUM_CLASSES = 7
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 1. Pipeline de Dados (Oversampling exige Augmentation forte para não dar Overfit)
transform = {
    'train': transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
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

# --- LÓGICA DE OVERSAMPLING (WeightedRandomSampler) ---
targets = np.array(train_dataset.targets)
class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in targets])
samples_weight = torch.from_numpy(samples_weight)
sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# 3. Model Building
def get_efficientnet():
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES)
    return model.to(DEVICE)

def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    t_loss, t_correct, t_total = 0, 0, 0
    for inputs, labels in loader:
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
    print(f"\n[EXP 1] Oversampling + Augmentation na EfficientNetB0")
    model = get_efficientnet()
    # Como o Sampler já balanceia os lotes, não precisamos de class_weights no Criterion!
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_acc = 0
    for epoch in range(30):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        v_loss, v_acc = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}: Train Acc {t_acc:.4f} | Val Acc {v_acc:.4f}")
        
        scheduler.step(v_loss)
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "efficientnet_oversampling_best.pth"))
            
    # Métricas Finais
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "efficientnet_oversampling_best.pth")))
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
    pd.DataFrame(report).transpose().to_csv(os.path.join(REPORTS_DIR, "metrics_efficientnet_oversampling.csv"))
    
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Oranges', xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES)
    plt.title('Matriz de Confusão: EfficientNet + Oversampling')
    plt.savefig(os.path.join(FIGURES_DIR, "cm_efficientnet_oversampling.png"))
    plt.close()
    print(f"[FINALIZADO EXP 1] Best Val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    main()
