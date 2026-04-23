#!/usr/bin/env python3
"""
Resume Transfer Learning para FER2013 - Foco em EfficientNetB0
Nota: ResNet50V2 já concluída com 61.91% de acurácia.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
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

BATCH_SIZE = 64
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 7
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def create_data_generators():
    train_datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
        brightness_range=[0.8, 1.2], fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator()
    
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(224, 224), color_mode='rgb',
        batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR, target_size=(224, 224), color_mode='rgb',
        batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
    )
    return train_gen, val_gen

def get_class_weights(train_gen):
    classes = train_gen.classes
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(classes), y=classes)
    return {int(i): float(w) for i, w in enumerate(cw)}

def build_efficientnetb0():
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
    return Model(inputs, outputs), base_model

def train_model_two_phase(model, base_model, train_gen, val_gen, c_weights, model_name):
    print(f"\n[FASE 1] Warmup: {model_name}")
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    
    callbacks_f1 = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5),
        ModelCheckpoint(os.path.join(MODELS_DIR, f'{model_name}_f1.weights.h5'), save_best_only=True, save_weights_only=True)
    ]
    
    history_f1 = model.fit(train_gen, epochs=15, validation_data=val_gen, class_weight=c_weights, callbacks=callbacks_f1)
    
    print(f"\n[FASE 2] Fine-tuning: {model_name}")
    base_model.trainable = True
    for layer in base_model.layers[:-20]: layer.trainable = False
    
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    callbacks_f2 = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
        ModelCheckpoint(os.path.join(MODELS_DIR, f'{model_name}_f2.weights.h5'), save_best_only=True, save_weights_only=True)
    ]
    
    history_f2 = model.fit(train_gen, epochs=30, validation_data=val_gen, class_weight=c_weights, callbacks=callbacks_f2)
    return history_f1, history_f2

def evaluate_model(model, val_gen, model_name):
    print(f"\n[AVALIACAO] {model_name}")
    val_gen.reset()
    y_pred = np.argmax(model.predict(val_gen), axis=1)
    y_true = val_gen.classes
    
    report = classification_report(y_true, y_pred, target_names=EMOTION_CLASSES, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(REPORTS_DIR, f'metrics_{model_name}.csv'))
    
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES)
    plt.savefig(os.path.join(FIGURES_DIR, f'cm_{model_name}.png'), dpi=300)
    plt.close()
    
    return df_report.loc['accuracy', 'f1-score']

def main():
    print("[RESUME] Iniciando treinamento da EfficientNetB0...")
    train_gen, val_gen = create_data_generators()
    c_weights = get_class_weights(train_gen)
    
    model_eff, base_eff = build_efficientnetb0()
    
    # Se quiser economizar tempo, poderíamos carregar os pesos da Fase 1 se existirem e forem válidos
    # Mas para garantir consistência, vamos rodar as duas fases da EfficientNet
    h1, h2 = train_model_two_phase(model_eff, base_eff, train_gen, val_gen, c_weights, 'efficientnetb0')
    
    acc = evaluate_model(model_eff, val_gen, 'efficientnetb0')
    
    hist_combined = pd.concat([pd.DataFrame(h1.history), pd.DataFrame(h2.history)], ignore_index=True)
    hist_combined.to_csv(os.path.join(REPORTS_DIR, 'historico_efficientnetb0.csv'), index=False)
    
    print(f"\n[FINALIZADO] EfficientNetB0 Accuracy: {acc:.4f}")
    
    # Mostrar comparação rápida com ResNet (lendo do CSV salvo anteriormente)
    resnet_metrics = pd.read_csv(os.path.join(REPORTS_DIR, 'metrics_resnet50v2.csv'), index_col=0)
    resnet_acc = resnet_metrics.loc['accuracy', 'f1-score']
    print(f"\nComparação:")
    print(f"  ResNet50V2:   {resnet_acc:.4f}")
    print(f"  EfficientNet: {acc:.4f}")

if __name__ == "__main__":
    main()
