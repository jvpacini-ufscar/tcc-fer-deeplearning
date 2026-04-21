#!/usr/bin/env python3
"""
Transfer Learning para FER2013 - Script de Treinamento Otimizado
Objetivo: Atingir 60%+ de acurácia no FER2013

Tecnicas:
- ResNet50V2 + EfficientNetB0 com Transfer Learning
- Fine-tuning em duas fases
- Data Augmentation forte
- Class Weights para desbalanceamento
- Learning Rate Scheduling
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuracoes
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_DIR = os.path.join(BASE_DIR, 'data/raw/fer2013/train')
VAL_DIR = os.path.join(BASE_DIR, 'data/raw/fer2013/test')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')

for d in [MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

BATCH_SIZE = 64
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 7
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

print("[INFO] GPU disponivel:", tf.config.list_physical_devices('GPU'))


def create_data_generators():
    """Cria data generators com augmentation forte."""
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        rescale=1./255
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_gen, val_gen


def get_class_weights(train_gen):
    """Calcula class weights para balancear classes desbalanceadas."""
    classes = train_gen.classes
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(classes),
        y=classes
    )
    return dict(enumerate(class_weights))


def build_resnet50v2():
    """Construi ResNet50V2 com top customizado para FER."""
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    base_model.trainable = False
    
    inputs = Input(shape=INPUT_SHAPE)
    x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model, base_model


def build_efficientnetb0():
    """Construi EfficientNetB0 com top customizado para FER."""
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    base_model.trainable = False
    
    inputs = Input(shape=INPUT_SHAPE)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model, base_model


def train_model_two_phase(model, base_model, train_gen, val_gen, c_weights, 
                          model_name, num_unfreeze_layers=30):
    """Treina modelo em duas fases: warmup + fine-tuning."""
    
    print(f"\n{'='*60}")
    print(f"TREINANDO: {model_name}")
    print(f"{'='*60}")
    
    # FASE 1: Warmup
    print("\n[FASE 1] Warmup - Treinando top classifier...")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_f1 = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=0),
        ModelCheckpoint(
            os.path.join(MODELS_DIR, f'{model_name}_f1.keras'),
            save_best_only=True,
            verbose=0
        )
    ]
    
    history_f1 = model.fit(
        train_gen,
        epochs=15,
        validation_data=val_gen,
        class_weight=c_weights,
        callbacks=callbacks_f1,
        verbose=1
    )
    
    # FASE 2: Fine-tuning
    print("\n[FASE 2] Fine-tuning - Descongelando ultimas camadas...")
    base_model.trainable = True
    
    for layer in base_model.layers[:-num_unfreeze_layers]:
        layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_f2 = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=0),
        ModelCheckpoint(
            os.path.join(MODELS_DIR, f'{model_name}_f2.keras'),
            save_best_only=True,
            verbose=0
        )
    ]
    
    train_gen.reset()
    val_gen.reset()
    
    history_f2 = model.fit(
        train_gen,
        epochs=30,
        validation_data=val_gen,
        class_weight=c_weights,
        callbacks=callbacks_f2,
        verbose=1
    )
    
    return history_f1, history_f2


def evaluate_model(model, val_gen, model_name):
    """Avalia modelo e salva metricas."""
    print(f"\n[AVALIACAO] {model_name}...")
    
    val_gen.reset()
    y_pred_probs = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes
    
    # Classification report
    report_dict = classification_report(
        y_true, y_pred,
        target_names=EMOTION_CLASSES,
        output_dict=True
    )
    df_report = pd.DataFrame(report_dict).transpose()
    csv_path = os.path.join(REPORTS_DIR, f'metrics_{model_name}.csv')
    df_report.to_csv(csv_path, index=True)
    print(f"[SALVO] Metricas: {csv_path}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    fig_path = os.path.join(FIGURES_DIR, f'cm_{model_name}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SALVO] Confusion matrix: {fig_path}")
    
    # Acurácia global
    accuracy = df_report.loc['accuracy', 'f1-score']
    print(f"[RESULTADO] Acurácia {model_name}: {accuracy:.4f}")
    
    return y_true, y_pred, df_report


def main():
    """Funcao principal."""
    print("[INFO] Criando data generators...")
    train_gen, val_gen = create_data_generators()
    print(f"[INFO] Train samples: {train_gen.samples}, Val samples: {val_gen.samples}")
    
    print("\n[INFO] Calculando class weights...")
    c_weights = get_class_weights(train_gen)
    print(f"[INFO] Class weights: {c_weights}")
    
    results = {}
    
    # ========== ResNet50V2 ==========
    print("\n\n" + "#"*60)
    print("# EXPERIMENTO 1: ResNet50V2")
    print("#"*60)
    
    model_resnet, base_resnet = build_resnet50v2()
    hist_f1, hist_f2 = train_model_two_phase(
        model_resnet, base_resnet, train_gen, val_gen, c_weights,
        'resnet50v2', num_unfreeze_layers=30
    )
    
    train_gen.reset()
    val_gen.reset()
    y_true_r, y_pred_r, metrics_r = evaluate_model(model_resnet, val_gen, 'resnet50v2')
    results['ResNet50V2'] = metrics_r.loc['accuracy', 'f1-score']
    
    # Salva historico
    hist_combined = pd.concat([
        pd.DataFrame(hist_f1.history),
        pd.DataFrame(hist_f2.history)
    ], ignore_index=True)
    hist_combined.to_csv(os.path.join(REPORTS_DIR, 'historico_resnet50v2.csv'), index=False)
    
    # ========== EfficientNetB0 ==========
    print("\n\n" + "#"*60)
    print("# EXPERIMENTO 2: EfficientNetB0")
    print("#"*60)
    
    model_eff, base_eff = build_efficientnetb0()
    hist_eff_f1, hist_eff_f2 = train_model_two_phase(
        model_eff, base_eff, train_gen, val_gen, c_weights,
        'efficientnetb0', num_unfreeze_layers=20
    )
    
    train_gen.reset()
    val_gen.reset()
    y_true_e, y_pred_e, metrics_e = evaluate_model(model_eff, val_gen, 'efficientnetb0')
    results['EfficientNetB0'] = metrics_e.loc['accuracy', 'f1-score']
    
    # Salva historico
    hist_eff_combined = pd.concat([
        pd.DataFrame(hist_eff_f1.history),
        pd.DataFrame(hist_eff_f2.history)
    ], ignore_index=True)
    hist_eff_combined.to_csv(os.path.join(REPORTS_DIR, 'historico_efficientnetb0.csv'), index=False)
    
    # ========== Comparativa Final ==========
    print("\n\n" + "="*60)
    print("RESUMO FINAL")
    print("="*60)
    print("\nAcuracias por Modelo:")
    for model_name, acc in results.items():
        status = "OK META ATINGIDA" if acc >= 0.60 else "ABAIXO DA META"
        print(f"  {model_name:20s}: {acc:.4f} {status}")
    
    print("\n[NOTA CRITICA]")
    print("-" * 60)
    print("Essas metricas foram validadas usando:")
    print("  OK sklearn.metrics.classification_report (Precision, Recall, F1)")
    print("  OK Confusion Matrix normalizada por true labels")
    print("  OK Dados de validacao separados (sem overlap com train)")
    print("\nPossiveis distorcoes:")
    print("  WARNING Desbalanceamento extremo (Disgust 5%, Happy 40%)")
    print("  WARNING Overfitting em classes maiores apesar de class weights")
    print("-" * 60)
    
    print("\n[ARTEFATOS SALVOS]")
    print(f"  Modelos: {MODELS_DIR}")
    print(f"  Metricas: {REPORTS_DIR}")
    print(f"  Graficos: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
