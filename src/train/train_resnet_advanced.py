import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils import class_weight
import sys

# Garante que o diretório atual reconheça o src para import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.evaluate.metrics import evaluate_model

def get_class_weights(train_generator):
    classes = train_generator.classes
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(classes),
        y=classes
    )
    return dict(enumerate(class_weights))

def build_model(input_shape=(224, 224, 3), num_classes=7):
    # Base model com weights do ImageNet
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Congela a base
    base_model.trainable = False
    
    # Top classifier (Customizado para FER)
    inputs = Input(shape=(48, 48, 1)) # FER2013 original
    
    # Expande grayscale para RGB e faz resize
    x = tf.image.grayscale_to_rgb(inputs)
    x = tf.image.resize(x, (224, 224))
    
    # Passa pelo preprocessamento do ResNet50V2 (-1 a 1)
    x = tf.keras.applications.resnet_v2.preprocess_input(x)
    
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model, base_model

def main():
    # Caminhos
    train_dir = '../../data/raw/fer2013/train'
    val_dir = '../../data/raw/fer2013/test'
    models_dir = '../../models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Augmentation rigoroso (Meta-learning literature insights)
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validação não tem augmentation, apenas rescaling é feito via Lambda no modelo
    val_datagen = ImageDataGenerator()
    
    batch_size = 64
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48), # Lemos no tamanho original
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Pesos das classes para balancear "Disgust" (Nojo) e outros
    c_weights = get_class_weights(train_generator)
    print("[INFO] Pesos das classes calculados:", c_weights)
    
    model, base_model = build_model()
    
    # FASE 1: Treinar apenas o Top Classifier
    print("\n[INFO] --- FASE 1: Treinando o Top Classifier (Warmup) ---")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    callbacks_f1 = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5),
        ModelCheckpoint(os.path.join(models_dir, 'resnet50v2_fase1_best.h5'), save_best_only=True)
    ]
    
    history1 = model.fit(
        train_generator,
        epochs=15,
        validation_data=val_generator,
        class_weight=c_weights,
        callbacks=callbacks_f1
    )
    
    # FASE 2: Fine-Tuning do último bloco
    print("\n[INFO] --- FASE 2: Fine-Tuning do último bloco convolucional ---")
    base_model.trainable = True
    
    # Congela as primeiras camadas, deixa o último bloco livre
    for layer in base_model.layers[:-30]:
        layer.trainable = False
        
    # Usa learning rate MUITO menor para não destruir os pesos
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    callbacks_f2 = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
        ModelCheckpoint(os.path.join(models_dir, 'resnet50v2_fase2_best.h5'), save_best_only=True)
    ]
    
    history2 = model.fit(
        train_generator,
        epochs=30,
        validation_data=val_generator,
        class_weight=c_weights,
        callbacks=callbacks_f2
    )
    
    print("\n[INFO] --- Avaliação Final Padronizada ---")
    # Coleta todas as imagens de validação para o classification report
    x_val = []
    y_val = []
    val_generator.reset()
    for i in range(len(val_generator)):
        x, y = next(val_generator)
        x_val.append(x)
        y_val.append(y)
    x_val = np.concatenate(x_val)
    y_val = np.concatenate(y_val)
    
    classes = list(train_generator.class_indices.keys())
    evaluate_model(model, x_val, y_val, classes, 'resnet50v2_advanced')
    
    # Salva histórico unificado
    hist_df1 = pd.DataFrame(history1.history)
    hist_df2 = pd.DataFrame(history2.history)
    hist_df = pd.concat([hist_df1, hist_df2], ignore_index=True)
    hist_df.to_csv('../../reports/historico_resnet50v2_advanced.csv', index=False)
    print("[INFO] Treinamento concluído. Todos os artefatos salvos.")

if __name__ == "__main__":
    main()
