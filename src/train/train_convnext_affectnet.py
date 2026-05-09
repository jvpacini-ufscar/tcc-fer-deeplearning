import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import os
import pandas as pd
from tqdm import tqdm

# Configurações
BASE_DIR = "."
DATA_DIR = os.path.join(BASE_DIR, "data/raw/affectnet_data")
TRAIN_DIR = os.path.join(DATA_DIR, "Train")
VAL_DIR = os.path.join(DATA_DIR, "Test")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_CLASSES = 8 # AffectNet tem Contempt
EPOCHS_WARMUP = 3
EPOCHS_FULL = 20

def train():
    print(f"\n[INÍCIO] Treinando ConvNeXt no AffectNet")
    
    # 1. Dados
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

    # 2. Modelo
    model = timm.create_model('convnext_tiny', pretrained=True, num_classes=NUM_CLASSES)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # FASE 1: Warmup
    print("--- FASE 1: WARMUP CLASSIFIER ---")
    for param in model.parameters(): param.requires_grad = False
    for param in model.head.parameters(): param.requires_grad = True
    
    optimizer = optim.AdamW(model.head.parameters(), lr=1e-3)
    for epoch in range(EPOCHS_WARMUP):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Warmup {epoch+1}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    # FASE 2: Fine-tuning
    print("--- FASE 2: FINE-TUNING TOTAL ---")
    for param in model.parameters(): param.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FULL)

    best_acc = 0
    for epoch in range(EPOCHS_FULL):
        model.train()
        t_correct, t_total = 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            _, pred = outputs.max(1)
            t_total += labels.size(0)
            t_correct += pred.eq(labels).sum().item()

        # Validação
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, pred = outputs.max(1)
                v_total += labels.size(0)
                v_correct += pred.eq(labels).sum().item()
        
        v_acc = v_correct / v_total
        print(f"Epoch {epoch+1}: Train Acc {t_correct/t_total:.4f} | Val Acc {v_acc:.4f}")
        
        scheduler.step()
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "convnext_affectnet_best.pth"))
            print(f"Novo recorde salvo: {best_acc:.4f}")

if __name__ == "__main__":
    train()
