import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import os
import pandas as pd
from sklearn.metrics import accuracy_score

# Mapeamento de classes (AffectNet -> Common 7)
# AffectNet: 0:anger, 1:contempt, 2:disgust, 3:fear, 4:happy, 5:neutral, 6:sad, 7:surprise
# Common: 0:angry, 1:disgust, 2:fear, 3:happy, 4:neutral, 5:sad, 6:surprise
MAP_AFFECT_TO_COMMON = {0:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = "models"
REPORTS_DIR = "reports"

def eval_on_dataset(model, data_dir, name):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Converte predição AffectNet para escala comum
            for p, t in zip(predicted.cpu().tolist(), labels.tolist()):
                if p in MAP_AFFECT_TO_COMMON:
                    y_pred.append(MAP_AFFECT_TO_COMMON[p])
                    y_true.append(t)
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Acurácia Cross-Dataset no {name}: {acc:.4f}")
    return acc

def main():
    model = timm.create_model('convnext_tiny', pretrained=False, num_classes=8)
    model_path = os.path.join(MODELS_DIR, "convnext_affectnet_best.pth")
    if not os.path.exists(model_path):
        print("Modelo AffectNet não encontrado.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    
    results = {}
    results['FER2013'] = eval_on_dataset(model, "data/raw/fer2013/test", "FER2013")
    results['RAF-DB'] = eval_on_dataset(model, "data/raw/raf-db/test", "RAF-DB")
    
    pd.DataFrame([results]).to_csv(os.path.join(REPORTS_DIR, "metrics_cross_dataset_affectnet.csv"), index=False)

if __name__ == "__main__":
    main()
