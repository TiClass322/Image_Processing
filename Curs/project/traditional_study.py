import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from skimage import feature, transform
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "C:\\Users\\danak\\Downloads\\1_Intro_Color_models\\Curs\\EuroSAT"

class TraditionalClassifier:
    
    def __init__(self, target_size=(64, 64)):
        self.svm = SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            class_weight='balanced',
            C=1.0,
            gamma='scale',
            cache_size=500,
            verbose=False
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.label_map = None
        self.reverse_label_map = None
        self.target_size = target_size
        
        # Оптимизированные параметры HOG
        self.hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': 'L2-Hys',
            'channel_axis': None
        }
        
        self.feature_cache = {}
    
    def load_data(self):
        """Загрузка данных с проверкой файлов"""
        try:
            train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
            val_df = pd.read_csv(os.path.join(DATA_PATH, "validation.csv"))
            
            with open(os.path.join(DATA_PATH, "label_map.json"), 'r') as f:
                self.label_map = json.load(f)
            self.reverse_label_map = {v: k for k, v in self.label_map.items()}
            
            print(f"✅ Загружено: {len(train_df)} train, {len(val_df)} val")
            return train_df, val_df
            
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            return None, None
    
    def load_image(self, img_path):
        """Загрузка и предобработка изображения"""
        try:
            img = Image.open(img_path)
            img_array = np.array(img)
            
            if img_array.size == 0 or len(img_array.shape) != 3:
                return None
            
            # Ресайз с сохранением пропорций
            if img_array.shape[:2] != self.target_size:
                img_pil = Image.fromarray(img_array)
                img_pil = img_pil.resize(self.target_size, Image.Resampling.LANCZOS)
                img_array = np.array(img_pil)
            
            return img_array
            
        except Exception as e:
            return None
    
    def prepare_data(self, df, sample_size=None, use_cache=True):
        """Подготовка данных с кэшированием"""
        print(f"\n📊 Подготовка данных...")
        
        if sample_size is None:
            sample_size = len(df)
        
        sample_df = df.sample(min(sample_size, len(df)), random_state=42)
        cache_key = f"data_{sample_size}_{hash(str(sample_df.index.tolist()))}"
        
        if use_cache and cache_key in self.feature_cache:
            print(f"✅ Используем кэшированные данные")
            return self.feature_cache[cache_key]
        
        images = []
        labels = []
        failed = 0
        
        print(f"Загрузка {len(sample_df)} изображений...")
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Изображения"):
            img_path = os.path.join(DATA_PATH, row['Filename'])
            img_array = self.load_image(img_path)
            
            if img_array is not None:
                images.append(img_array)
                labels.append(row['Label'])
            else:
                failed += 1
        
        print(f"✅ Успешно: {len(images)}, Ошибок: {failed}")
        
        if use_cache:
            self.feature_cache[cache_key] = (images, labels)
        
        return images, labels
    
    def extract_hog_features(self, images, progress=True):
        """Извлечение HOG-признаков с прогресс-баром"""
        print(f"\n🔍 Извлечение HOG-признаков...")
        
        features = []
        cache_key = f"hog_{len(images)}_{hash(str([id(img) for img in images]))}"
        
        if cache_key in self.feature_cache:
            print(f"✅ Используем кэшированные признаки")
            return self.feature_cache[cache_key]
        
        iterator = tqdm(images, desc="HOG обработка") if progress else images
        
        for img_array in iterator:
            # Конвертация в grayscale
            if len(img_array.shape) == 3:
                gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
            else:
                gray = img_array
            
            # Нормализация
            gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
            
            # HOG
            hog_feat = feature.hog(
                gray,
                orientations=self.hog_params['orientations'],
                pixels_per_cell=self.hog_params['pixels_per_cell'],
                cells_per_block=self.hog_params['cells_per_block'],
                block_norm=self.hog_params['block_norm'],
                visualize=False,
                channel_axis=self.hog_params['channel_axis']
            )
            
            features.append(hog_feat)
        
        features = np.array(features)
        self.feature_cache[cache_key] = features
        
        print(f"✅ Размерность признаков: {features.shape}")
        return features
    
    def train(self, train_sample=4000, val_sample=1000):
        """Обучение модели"""
        print("\n" + "="*60)
        print("🚀 ОБУЧЕНИЕ HOG + SVM КЛАССИФИКАТОРА")
        print("="*60)
        
        train_df, val_df = self.load_data()
        if train_df is None:
            return 0.0
        
        X_train, y_train = self.prepare_data(train_df, train_sample)
        X_val, y_val = self.prepare_data(val_df, val_sample)
        
        # Извлечение признаков
        print("\n📈 Извлечение признаков...")
        X_train_features = self.extract_hog_features(X_train)
        X_val_features = self.extract_hog_features(X_val)
        
        # Масштабирование
        print("\n⚖️  Масштабирование признаков...")
        X_train_scaled = self.scaler.fit_transform(X_train_features)
        X_val_scaled = self.scaler.transform(X_val_features)
        
        # Обучение
        print("\n🎯 Обучение SVM...")
        self.svm.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Оценка
        return self.evaluate(X_train_scaled, y_train, X_val_scaled, y_val)
    
    def evaluate(self, X_train, y_train, X_val, y_val):
        """Оценка модели"""
        print("\n📊 ОЦЕНКА МОДЕЛИ")
        print("-" * 40)
        
        y_train_pred = self.svm.predict(X_train)
        y_val_pred = self.svm.predict(X_val)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        print(f"🎯 Точность обучения: {train_acc:.4f}")
        print(f"🎯 Точность валидации: {val_acc:.4f}")
        print(f"📏 Переобучение: {train_acc - val_acc:.4f}")
        
        # Детальный отчет
        class_names = [self.reverse_label_map[i] for i in range(len(self.label_map))]
        print(f"\n📋 Детальный отчет:")
        print(classification_report(y_val, y_val_pred, 
                                  target_names=class_names,
                                  zero_division=0))
        
        # Матрица ошибок
        self.plot_confusion_matrix(y_val, y_val_pred, class_names)
        
        return val_acc
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Визуализация матрицы ошибок"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Количество'})
        
        plt.title('Матрица ошибок - HOG+SVM', fontsize=16, pad=20)
        plt.xlabel('Предсказанные классы', fontsize=14)
        plt.ylabel('Истинные классы', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        self.analyze_confusion_matrix(cm, class_names)
    
    def analyze_confusion_matrix(self, cm, class_names):
        """Анализ матрицы ошибок"""
        print("\n🔍 АНАЛИЗ ОШИБОК:")
        print("-" * 40)
        
        for i, class_name in enumerate(class_names):
            correct = cm[i, i]
            total = cm[i].sum()
            accuracy = correct / total if total > 0 else 0
            
            if accuracy < 0.6:
                print(f"\n❌ Проблемный класс: {class_name}")
                print(f"   Точность: {accuracy:.2%} ({correct}/{total})")
                
                errors = [(class_names[j], cm[i, j]) 
                         for j in range(len(class_names)) 
                         if j != i and cm[i, j] > 0]
                errors.sort(key=lambda x: x[1], reverse=True)
                
                if errors:
                    print(f"   Частые ошибки: {errors[:3]}")
    
    def save_model(self, filename='traditional_model_hog.pkl'):
        """Сохранение модели"""
        if not self.is_trained:
            print("❌ Модель не обучена!")
            return
        
        model_data = {
            'svm': self.svm,
            'scaler': self.scaler,
            'label_map': self.label_map,
            'hog_params': self.hog_params,
            'target_size': self.target_size
        }
        
        joblib.dump(model_data, filename)
        print(f"✅ Модель сохранена: {filename}")
    
    def load_model(self, filename='traditional_model_hog.pkl'):
        """Загрузка модели"""
        try:
            model_data = joblib.load(filename)
            self.svm = model_data['svm']
            self.scaler = model_data['scaler']
            self.label_map = model_data['label_map']
            self.hog_params = model_data.get('hog_params', self.hog_params)
            self.target_size = model_data.get('target_size', (64, 64))
            self.reverse_label_map = {v: k for k, v in self.label_map.items()}
            self.is_trained = True
            print(f"✅ Модель загружена: {filename}")
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")

def main():
    """Основная функция"""
    classifier = TraditionalClassifier(target_size=(64, 64))
    
    print("\n" + "="*60)
    print("🎯 ЗАПУСК HOG+SVM КЛАССИФИКАТОРА")
    print("="*60)
    
    accuracy = classifier.train(train_sample=4000, val_sample=1000)
    classifier.save_model()
    
    print("\n" + "="*60)
    print(f"🏁 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"📈 Итоговая точность: {accuracy:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()