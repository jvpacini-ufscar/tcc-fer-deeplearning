import os
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.evaluate.metrics import evaluate_model

def get_data(val_dir):
    val_datagen = ImageDataGenerator()
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=128,
        class_mode='categorical',
        shuffle=False
    )
    
    x_val = []
    y_val = []
    for i in range(len(val_generator)):
        x, y = next(val_generator)
        x_val.append(x)
        y_val.append(y)
    return np.concatenate(x_val), np.concatenate(y_val), list(val_generator.class_indices.keys())

def main():
    val_dir = 'data/raw/fer2013/test'
    if not os.path.exists(val_dir):
        print("Data dir not found")
        return
        
    x_val, y_val, classes = get_data(val_dir)
    print("Dados de validação carregados:", x_val.shape)
    
    models_to_evaluate = {
        'melhor_modelo_cnn_customizada.keras': 'cnn_customizada',
        'melhor_modelo_resnet50.keras': 'resnet50_baseline'
    }
    
    for filename, model_name in models_to_evaluate.items():
        filepath = os.path.join('models', filename)
        if os.path.exists(filepath):
            try:
                print(f"Carregando {filepath}...")
                model = load_model(filepath)
                evaluate_model(model, x_val, y_val, classes, model_name)
            except Exception as e:
                print(f"Erro ao avaliar {filename}: {e}")
                
if __name__ == "__main__":
    main()
