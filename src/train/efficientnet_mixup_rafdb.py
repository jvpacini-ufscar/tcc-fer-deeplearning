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
BATCH_SIZE = 32
NUM_CLASSES = 7
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 1. Pipeline de Dados
transform = {
    'train': transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# --- FUNÇÃO MIXUP ---
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def main():
    print(f"\n[EXP 3] MixUp Regularization na EfficientNetB0")
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=7).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    best_acc = 0
    for epoch in range(30):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Aplicar MixUp
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.4)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        # Validação
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()
        
        acc = correct / total
        print(f"Epoch {epoch+1}: Val Acc {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "efficientnet_mixup_best.pth"))

    # Relatório Final
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "efficientnet_mixup_best.pth")))
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
    pd.DataFrame(report).transpose().to_csv(os.path.join(REPORTS_DIR, "metrics_efficientnet_mixup.csv"))
    
    print(f"[FINALIZADO EXP 3] Best Val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    main()
