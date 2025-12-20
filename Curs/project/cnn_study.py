import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Настройка для CPU: ограничиваем использование потоков, если нужно
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_PATH = "C:\\Users\\danak\\Downloads\\1_Intro_Color_models\\Curs\\EuroSAT"

class EuroSatNet:
    """Оптимизированный CNN классификатор для работы на CPU."""
    
    def __init__(self, input_shape=(64, 64, 3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.label_map = None

    def _build_architecture(self):
        """Создание легкой архитектуры на основе Separable Convolutions."""
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            
            # Блок 1
            layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Блок 2
            layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Блок 3
            layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Вместо Flatten используем GlobalAveragePooling для экономии памяти и весов
            layers.GlobalAveragePooling2D(),
            
            # Компактный полносвязный слой
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def get_data_generators(self, train_df, val_df, batch_size=32):
        """Эффективная загрузка данных прямо из DataFrame."""
        # Аугментация только для тренировки
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            horizontal_flip=True,
            zoom_range=0.1
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)

        # Создаем генераторы
        train_gen = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            directory=DATA_PATH,
            x_col='Filename',
            y_col='Label',
            target_size=self.input_shape[:2],
            class_mode='raw',
            batch_size=batch_size,
            shuffle=True
        )

        val_gen = val_datagen.flow_from_dataframe(
            dataframe=val_df,
            directory=DATA_PATH,
            x_col='Filename',
            y_col='Label',
            target_size=self.input_shape[:2],
            class_mode='raw',
            batch_size=batch_size,
            shuffle=False
        )

        return train_gen, val_gen

    def train_model(self, epochs=25, batch_size=32):
        print("🚀 Загрузка данных и подготовка пайплайна...")
        
        # Загрузка CSV
        train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
        val_df = pd.read_csv(os.path.join(DATA_PATH, "validation.csv"))
        
        with open(os.path.join(DATA_PATH, "label_map.json"), 'r') as f:
            self.label_map = json.load(f)
        
        train_gen, val_gen = self.get_data_generators(train_df, val_df, batch_size)
        
        self.model = self._build_architecture()
        self.model.summary()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
            ModelCheckpoint('best_cpu_model.h5', save_best_only=True)
        ]

        print(f"\n🧠 Начинаю обучение на CPU (потоков: {os.cpu_count()})...")
        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self._evaluate(val_gen)

    def _evaluate(self, val_gen):
        """Оценка модели и вывод метрик."""
        val_gen.reset()
        predictions = self.model.predict(val_gen)
        y_pred = np.argmax(predictions, axis=1)
        y_true = val_gen.labels

        class_names = list(self.label_map.keys())
        print("\n📊 ОТЧЕТ ПО КЛАССИФИКАЦИИ:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        self._plot_results(y_true, y_pred, class_names)

    def _plot_results(self, y_true, y_pred, class_names):
        """Визуализация матрицы ошибок."""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='rocket_r', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Матрица ошибок (Optimized CNN)')
        plt.ylabel('Истина')
        plt.xlabel('Предсказание')
        plt.tight_layout()
        plt.show()

    def save(self, filename='euro_sat_cpu_model'):
        self.model.save(f'{filename}.h5')
        meta = {'label_map': self.label_map, 'input_shape': self.input_shape}
        joblib.dump(meta, f'{filename}_meta.pkl')
        print(f"✅ Модель сохранена: {filename}.h5")

def main():
    # Инициализируем классификатор
    clf = EuroSatNet(input_shape=(64, 64, 3))
    
    # Обучаем (для CPU 20-30 эпох обычно достаточно с такой архитектурой)
    clf.train_model(epochs=25, batch_size=32)
    
    # Сохраняем результат
    clf.save()

if __name__ == "__main__":
    main()