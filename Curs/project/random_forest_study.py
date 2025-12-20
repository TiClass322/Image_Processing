import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
from skimage import feature, transform
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "C:\\Users\\danak\\Downloads\\1_Intro_Color_models\\Curs\\EuroSAT"

class RandomForestClassifierModel:
    
    def __init__(self, target_size=(64, 64)):
        self.rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.label_map = None
        self.reverse_label_map = None
        self.target_size = target_size
        
        # Параметры признаков
        self.hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': 'L2-Hys'
        }
        
        self.color_bins = 16
        self.feature_cache = {}
    
    def load_data(self):
        """Загрузка данных"""
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
        """Загрузка изображения"""
        try:
            img = Image.open(img_path)
            img_array = np.array(img)
            
            if img_array.size == 0 or len(img_array.shape) != 3:
                return None
            
            # Ресайз
            if img_array.shape[:2] != self.target_size:
                img_pil = Image.fromarray(img_array)
                img_pil = img_pil.resize(self.target_size, Image.Resampling.LANCZOS)
                img_array = np.array(img_pil)
            
            # Нормализация
            img_array = img_array.astype('float32') / 255.0
            
            return img_array
            
        except Exception as e:
            return None
    
    def prepare_data(self, df, sample_size=None, balance=False):
        """Подготовка данных"""
        print(f"\n📊 Подготовка данных...")
        
        if sample_size is None:
            sample_size = len(df)
        
        if balance:
            # Балансировка классов
            class_counts = df['Label'].value_counts()
            min_samples = min(class_counts.min(), sample_size // len(class_counts))
            
            balanced_samples = []
            for label in df['Label'].unique():
                class_df = df[df['Label'] == label]
                if len(class_df) > min_samples:
                    class_df = class_df.sample(min_samples, random_state=42)
                balanced_samples.append(class_df)
            
            sample_df = pd.concat(balanced_samples, ignore_index=True).sample(frac=1, random_state=42)
        else:
            sample_df = df.sample(min(sample_size, len(df)), random_state=42)
        
        images = []
        labels = []
        
        print(f"Загрузка {len(sample_df)} изображений...")
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Изображения"):
            img_path = os.path.join(DATA_PATH, row['Filename'])
            img_array = self.load_image(img_path)
            
            if img_array is not None:
                images.append(img_array)
                labels.append(row['Label'])
        
        print(f"✅ Успешно загружено: {len(images)}")
        return images, np.array(labels)
    
    def extract_hog_features(self, images):
        """Извлечение HOG-признаков"""
        print(f"\n🔍 Извлечение HOG-признаков...")
        
        features = []
        cache_key = f"hog_{len(images)}"
        
        if cache_key in self.feature_cache:
            print(f"✅ Используем кэшированные HOG признаки")
            return self.feature_cache[cache_key]
        
        for img_array in tqdm(images, desc="HOG обработка"):
            # Grayscale
            if len(img_array.shape) == 3:
                gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
            else:
                gray = img_array
            
            # HOG
            hog_feat = feature.hog(
                gray,
                orientations=self.hog_params['orientations'],
                pixels_per_cell=self.hog_params['pixels_per_cell'],
                cells_per_block=self.hog_params['cells_per_block'],
                block_norm=self.hog_params['block_norm'],
                visualize=False,
                channel_axis=None
            )
            
            features.append(hog_feat)
        
        features = np.array(features)
        self.feature_cache[cache_key] = features
        return features
    
    def extract_color_features(self, images):
        """Извлечение цветовых признаков"""
        print(f"\n🎨 Извлечение цветовых признаков...")
        
        features = []
        
        for img_array in tqdm(images, desc="Цветовые признаки"):
            # Гистограммы
            hist_r = np.histogram(img_array[:,:,0], bins=self.color_bins, range=(0, 1))[0]
            hist_g = np.histogram(img_array[:,:,1], bins=self.color_bins, range=(0, 1))[0]
            hist_b = np.histogram(img_array[:,:,2], bins=self.color_bins, range=(0, 1))[0]
            
            # Статистики
            stats = [
                np.mean(img_array[:,:,0]), np.std(img_array[:,:,0]),
                np.mean(img_array[:,:,1]), np.std(img_array[:,:,1]),
                np.mean(img_array[:,:,2]), np.std(img_array[:,:,2]),
                np.mean(img_array), np.std(img_array),
                np.max(img_array) - np.min(img_array)  # Контраст
            ]
            
            # Объединение
            color_feat = np.concatenate([hist_r, hist_g, hist_b, stats])
            features.append(color_feat)
        
        return np.array(features)
    
    def extract_combined_features(self, images):
        """Извлечение комбинированных признаков"""
        print(f"\n🔄 Извлечение комбинированных признаков...")
        
        hog_features = self.extract_hog_features(images)
        color_features = self.extract_color_features(images)
        
        combined = np.hstack([hog_features, color_features])
        
        print(f"✅ Размерность признаков: {combined.shape}")
        print(f"   - HOG: {hog_features.shape[1]}")
        print(f"   - Цвет: {color_features.shape[1]}")
        print(f"   - Всего: {combined.shape[1]}")
        
        return combined
    
    def train(self, train_sample=3000, val_sample=800):
        """Обучение модели"""
        print("\n" + "="*60)
        print("🚀 ОБУЧЕНИЕ RANDOM FOREST КЛАССИФИКАТОРА")
        print("="*60)
        
        train_df, val_df = self.load_data()
        if train_df is None:
            return 0.0
        
        X_train, y_train = self.prepare_data(train_df, train_sample, balance=True)
        X_val, y_val = self.prepare_data(val_df, val_sample, balance=False)
        
        # Признаки
        print("\n📈 Извлечение признаков...")
        X_train_features = self.extract_combined_features(X_train)
        X_val_features = self.extract_combined_features(X_val)
        
        # Масштабирование
        print("\n⚖️  Масштабирование...")
        X_train_scaled = self.scaler.fit_transform(X_train_features)
        X_val_scaled = self.scaler.transform(X_val_features)
        
        # Обучение
        print("\n🎯 Обучение Random Forest...")
        self.rf.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Анализ важности признаков
        self.plot_feature_importance(X_train_features.shape[1])
        
        # Оценка
        return self.evaluate(X_train_scaled, y_train, X_val_scaled, y_val)
    
    def evaluate(self, X_train, y_train, X_val, y_val):
        """Оценка модели"""
        print("\n📊 ОЦЕНКА МОДЕЛИ")
        print("-" * 40)
        
        y_train_pred = self.rf.predict(X_train)
        y_val_pred = self.rf.predict(X_val)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        print(f"🎯 Точность обучения: {train_acc:.4f}")
        print(f"🎯 Точность валидации: {val_acc:.4f}")
        print(f"📏 Переобучение: {train_acc - val_acc:.4f}")
        
        # Отчет
        class_names = [self.reverse_label_map[i] for i in range(len(self.label_map))]
        print(f"\n📋 Детальный отчет:")
        print(classification_report(y_val, y_val_pred,
                                  target_names=class_names,
                                  zero_division=0))
        
        # Матрица ошибок
        self.plot_confusion_matrix(y_val, y_val_pred, class_names)
        
        return val_acc
    
    def plot_feature_importance(self, num_features):
        """Визуализация важности признаков"""
        if not self.is_trained:
            return
        
        importance = self.rf.feature_importances_
        
        # Группировка по типам признаков
        hog_dim = 324  # Размерность HOG
        color_hist_dim = self.color_bins * 3  # 3 канала × bins
        
        hog_importance = np.sum(importance[:hog_dim])
        color_hist_importance = np.sum(importance[hog_dim:hog_dim+color_hist_dim])
        color_stat_importance = np.sum(importance[hog_dim+color_hist_dim:])
        
        # Визуализация
        labels = ['HOG (форма)', 'Цвет. гистограммы', 'Цвет. статистики']
        values = [hog_importance, color_hist_importance, color_stat_importance]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values, color=['#3498db', '#e74c3c', '#2ecc71'])
        plt.title('Важность типов признаков в Random Forest', fontsize=14, pad=20)
        plt.ylabel('Суммарная важность', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n📊 АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ:")
        print(f"HOG (форма): {hog_importance:.3f}")
        print(f"Цветовые гистограммы: {color_hist_importance:.3f}")
        print(f"Цветовые статистики: {color_stat_importance:.3f}")
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Матрица ошибок"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Количество'})
        
        plt.title('Матрица ошибок - Random Forest', fontsize=16, pad=20)
        plt.xlabel('Предсказанные классы', fontsize=14)
        plt.ylabel('Истинные классы', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        self.analyze_confusion_matrix(cm, class_names)
    
    def analyze_confusion_matrix(self, cm, class_names):
        """Анализ ошибок"""
        print("\n🔍 АНАЛИЗ ОШИБОК:")
        print("-" * 40)
        
        for i, class_name in enumerate(class_names):
            correct = cm[i, i]
            total = cm[i].sum()
            accuracy = correct / total if total > 0 else 0
            
            if accuracy < 0.65:
                print(f"\n❌ Проблемный класс: {class_name}")
                print(f"   Точность: {accuracy:.2%} ({correct}/{total})")
                
                errors = [(class_names[j], cm[i, j], cm[i, j]/total*100)
                         for j in range(len(class_names))
                         if j != i and cm[i, j] > 0]
                errors.sort(key=lambda x: x[1], reverse=True)
                
                if errors:
                    print(f"   Основные ошибки:")
                    for err_class, count, percent in errors[:3]:
                        print(f"     → {err_class}: {count} ({percent:.1f}%)")
    
    def save_model(self, filename='random_forest_model.pkl'):
        """Сохранение модели"""
        if not self.is_trained:
            print("❌ Модель не обучена!")
            return
        
        model_data = {
            'rf': self.rf,
            'scaler': self.scaler,
            'label_map': self.label_map,
            'hog_params': self.hog_params,
            'color_bins': self.color_bins,
            'target_size': self.target_size
        }
        
        joblib.dump(model_data, filename)
        print(f"✅ Модель сохранена: {filename}")
    
    def load_model(self, filename='random_forest_model.pkl'):
        """Загрузка модели"""
        try:
            model_data = joblib.load(filename)
            self.rf = model_data['rf']
            self.scaler = model_data['scaler']
            self.label_map = model_data['label_map']
            self.hog_params = model_data.get('hog_params', self.hog_params)
            self.color_bins = model_data.get('color_bins', 16)
            self.target_size = model_data.get('target_size', (64, 64))
            self.reverse_label_map = {v: k for k, v in self.label_map.items()}
            self.is_trained = True
            print(f"✅ Модель загружена: {filename}")
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")

def main():
    """Основная функция"""
    rf_classifier = RandomForestClassifierModel(target_size=(64, 64))
    
    print("\n" + "="*60)
    print("🎯 ЗАПУСК RANDOM FOREST КЛАССИФИКАТОРА")
    print("="*60)
    
    accuracy = rf_classifier.train(train_sample=3000, val_sample=800)
    rf_classifier.save_model()
    
    print("\n" + "="*60)
    print(f"🏁 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"📈 Итоговая точность: {accuracy:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()