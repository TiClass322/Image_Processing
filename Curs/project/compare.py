import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import joblib
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Настройки отображения
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

class ModelComparator:
    def __init__(self, data_path="C:\\Users\\danak\\Downloads\\1_Intro_Color_models\\Curs\\EuroSAT"):
        self.data_path = data_path
        self.label_map = None
        self.reverse_label_map = None
        self.test_data = None
        self.test_labels = None
        self.results = {}
        self.models_loaded = {}
        
    def load_test_data(self, sample_size=1000):
        """Загрузка тестовых данных с оптимизацией"""
        try:
            print("\n📥 Загрузка тестовых данных...")
            
            test_df = pd.read_csv(os.path.join(self.data_path, "test.csv"))
            
            with open(os.path.join(self.data_path, "label_map.json"), 'r') as f:
                self.label_map = json.load(f)
            self.reverse_label_map = {v: k for k, v in self.label_map.items()}
            
            # Балансировка классов в тестовой выборке
            if sample_size < len(test_df):
                samples_per_class = sample_size // len(self.label_map)
                balanced_samples = []
                
                for label in test_df['Label'].unique():
                    class_samples = test_df[test_df['Label'] == label]
                    if len(class_samples) > samples_per_class:
                        class_samples = class_samples.sample(samples_per_class, random_state=42)
                    balanced_samples.append(class_samples)
                
                test_df = pd.concat(balanced_samples, ignore_index=True)
            else:
                test_df = test_df.sample(min(sample_size, len(test_df)), random_state=42)
            
            images = []
            labels = []
            
            print(f"Загрузка {len(test_df)} изображений...")
            for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Тестовые изображения"):
                img_filename = row['Filename']
                img_path = os.path.join(self.data_path, img_filename)
                
                try:
                    from PIL import Image
                    img = Image.open(img_path)
                    img_array = np.array(img)
                    
                    if img_array.size > 0 and len(img_array.shape) == 3:
                        # Приведение к 64x64 для CNN
                        if img_array.shape[:2] != (64, 64):
                            img_pil = Image.fromarray(img_array)
                            img_pil = img_pil.resize((64, 64), Image.Resampling.LANCZOS)
                            img_array = np.array(img_pil)
                        
                        images.append(img_array)
                        labels.append(row['Label'])
                        
                except Exception as e:
                    continue
            
            self.test_data = images
            self.test_labels = np.array(labels)
            
            print(f"✅ Успешно загружено {len(self.test_data)} тестовых изображений")
            print(f"📊 Распределение классов:")
            unique, counts = np.unique(self.test_labels, return_counts=True)
            for cls, count in zip(unique, counts):
                cls_name = self.reverse_label_map.get(cls, f"Class_{cls}")
                print(f"   {cls_name}: {count} изображений")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка загрузки тестовых данных: {e}")
            return False
    
    def load_model_if_exists(self, model_type):
        """Загрузка модели с проверкой"""
        model_files = {
            'HOG+SVM': 'models/traditional_model_hog.pkl',
            'Random Forest': 'models/random_forest_model.pkl',
            'CNN': ('models/euro_sat_cpu_model.h5', 'models/euro_sat_cpu_model_meta.pkl')
        }
        
        if model_type not in model_files:
            return False
        
        if model_type == 'CNN':
            model_file, meta_file = model_files[model_type]
            if not (os.path.exists(model_file) and os.path.exists(meta_file)):
                return False
        else:
            model_file = model_files[model_type]
            if not os.path.exists(model_file):
                return False
        
        return True
    
    def extract_features_for_model(self, model_type):
        """Извлечение признаков для разных моделей"""
        from traditional_study import TraditionalClassifier
        from random_forest_study import RandomForestClassifierModel
        
        if model_type == 'HOG+SVM':
            classifier = TraditionalClassifier()
            return classifier.extract_hog_features(self.test_data, progress=False)
        
        elif model_type == 'Random Forest':
            classifier = RandomForestClassifierModel()
            return classifier.extract_combined_features(self.test_data)
        
        return None
    
    def evaluate_model(self, model_type):
        """Оценка конкретной модели"""
        print(f"\n" + "="*60)
        print(f"🔍 ОЦЕНКА {model_type} МОДЕЛИ")
        print("="*60)
        
        try:
            start_load = time.time()
            
            if model_type == 'HOG+SVM':
                model_data = joblib.load('models/traditional_model_hog.pkl')
                svm = model_data['svm']
                scaler = model_data['scaler']
                
                # Извлечение признаков
                print("📊 Извлечение HOG-признаков...")
                test_features = self.extract_features_for_model(model_type)
                test_scaled = scaler.transform(test_features)
                
                # Предсказание
                start_pred = time.time()
                predictions = svm.predict(test_scaled)
                inference_time = time.time() - start_pred
                
                model_size = os.path.getsize('models/traditional_model_hog.pkl') / 1024 / 1024
                
            elif model_type == 'Random Forest':
                model_data = joblib.load('models/random_forest_model.pkl')
                rf = model_data['rf']
                scaler = model_data['scaler']
                
                # Извлечение признаков
                print("📊 Извлечение комбинированных признаков...")
                test_features = self.extract_features_for_model(model_type)
                test_scaled = scaler.transform(test_features)
                
                # Предсказание
                start_pred = time.time()
                predictions = rf.predict(test_scaled)
                inference_time = time.time() - start_pred
                
                model_size = os.path.getsize('models/random_forest_model.pkl') / 1024 / 1024
                
            elif model_type == 'CNN':
                import tensorflow as tf
                
                # Загрузка модели
                cnn_model = tf.keras.models.load_model('models/euro_sat_cpu_model.h5', compile=False)
                model_metadata = joblib.load('models/euro_sat_cpu_model_meta.pkl')
                
                # Подготовка изображений для CNN
                print("📊 Подготовка изображений для CNN...")
                test_images = []
                for img_array in tqdm(self.test_data, desc="Обработка изображений"):
                    # Нормализация
                    img_normalized = img_array.astype('float32') / 255.0
                    # Проверка каналов
                    if img_normalized.shape[2] == 4:
                        img_normalized = img_normalized[:, :, :3]
                    elif img_normalized.shape[2] == 1:
                        img_normalized = np.repeat(img_normalized, 3, axis=2)
                    test_images.append(img_normalized)
                
                test_images = np.array(test_images)
                
                # Предсказание
                start_pred = time.time()
                predictions_proba = cnn_model.predict(test_images, verbose=0, batch_size=32)
                predictions = np.argmax(predictions_proba, axis=1)
                inference_time = time.time() - start_pred
                
                model_size = os.path.getsize('models/euro_sat_cpu_model.h5') / 1024 / 1024
            
            else:
                return False
            
            load_time = time.time() - start_load
            
            # Расчет метрик
            accuracy = accuracy_score(self.test_labels, predictions)
            f1 = f1_score(self.test_labels, predictions, average='weighted')
            
            # Сохранение результатов
            self.results[model_type] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'inference_time': inference_time,
                'load_time': load_time,
                'total_time': load_time + inference_time,
                'predictions': predictions,
                'model_size': model_size,
                'samples_per_second': len(self.test_labels) / inference_time if inference_time > 0 else 0
            }
            
            # Вывод результатов
            print(f"🎯 Точность (Accuracy): {accuracy:.4f}")
            print(f"📊 F1-Score: {f1:.4f}")
            print(f"⏱️  Время загрузки модели: {load_time:.2f} сек")
            print(f"⏱️  Время предсказания: {inference_time:.2f} сек")
            print(f"⏱️  Общее время: {load_time + inference_time:.2f} сек")
            print(f"🚀 Скорость: {self.results[model_type]['samples_per_second']:.1f} изобр./сек")
            print(f"💾 Размер модели: {model_size:.2f} MB")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка оценки {model_type}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def plot_comparison_results(self):
        """Визуализация результатов сравнения"""
        if not self.results:
            print("❌ Нет данных для визуализации")
            return
        
        models = list(self.results.keys())
        
        # Создание фигуры с несколькими графиками
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('СРАВНЕНИЕ МОДЕЛЕЙ КЛАССИФИКАЦИИ', fontsize=16, y=1.02)
        
        # 1. Точность (Accuracy)
        accuracies = [self.results[m]['accuracy'] for m in models]
        axes[0, 0].bar(models, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
        axes[0, 0].set_title('Точность (Accuracy)', fontsize=14)
        axes[0, 0].set_ylabel('Точность')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 2. F1-Score
        f1_scores = [self.results[m]['f1_score'] for m in models]
        axes[0, 1].bar(models, f1_scores, color=['#3498db', '#2ecc71', '#e74c3c'])
        axes[0, 1].set_title('F1-Score', fontsize=14)
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_ylim(0, 1)
        for i, v in enumerate(f1_scores):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 3. Время предсказания
        inference_times = [self.results[m]['inference_time'] for m in models]
        axes[0, 2].bar(models, inference_times, color=['#3498db', '#2ecc71', '#e74c3c'])
        axes[0, 2].set_title('Время предсказания', fontsize=14)
        axes[0, 2].set_ylabel('Секунды')
        for i, v in enumerate(inference_times):
            axes[0, 2].text(i, v + max(inference_times)*0.05, f'{v:.2f}с', ha='center')
        
        # 4. Скорость обработки
        speeds = [self.results[m]['samples_per_second'] for m in models]
        axes[1, 0].bar(models, speeds, color=['#3498db', '#2ecc71', '#e74c3c'])
        axes[1, 0].set_title('Скорость обработки', fontsize=14)
        axes[1, 0].set_ylabel('Изображений/сек')
        for i, v in enumerate(speeds):
            axes[1, 0].text(i, v + max(speeds)*0.05, f'{v:.0f}', ha='center')
        
        # 5. Размер модели
        sizes = [self.results[m]['model_size'] for m in models]
        axes[1, 1].bar(models, sizes, color=['#3498db', '#2ecc71', '#e74c3c'])
        axes[1, 1].set_title('Размер модели', fontsize=14)
        axes[1, 1].set_ylabel('МБ')
        for i, v in enumerate(sizes):
            axes[1, 1].text(i, v + max(sizes)*0.05, f'{v:.1f}МБ', ha='center')
        
        # 6. Композитный график (нормализованные значения)
        normalized_acc = [a/max(accuracies) for a in accuracies]
        normalized_speed = [s/max(speeds) if max(speeds) > 0 else 0 for s in speeds]
        normalized_size = [1 - (s/max(sizes)) for s in sizes]  # инвертируем (меньше = лучше)
        
        x = np.arange(len(models))
        width = 0.25
        
        axes[1, 2].bar(x - width, normalized_acc, width, label='Точность (норм.)', color='#3498db')
        axes[1, 2].bar(x, normalized_speed, width, label='Скорость (норм.)', color='#2ecc71')
        axes[1, 2].bar(x + width, normalized_size, width, label='Размер⁻¹ (норм.)', color='#e74c3c')
        
        axes[1, 2].set_title('Нормализованное сравнение', fontsize=14)
        axes[1, 2].set_ylabel('Нормализованное значение')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(models)
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Вывод рекомендации
        self.print_recommendation()
    
    def print_recommendation(self):
        """Вывод рекомендации по выбору модели"""
        if not self.results:
            return
        
        print("\n" + "="*80)
        print("💡 РЕКОМЕНДАЦИИ ПО ВЫБОРУ МОДЕЛИ")
        print("="*80)
        
        # Находим лучшую модель по каждому критерию
        best_accuracy = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        best_speed = max(self.results.items(), key=lambda x: x[1]['samples_per_second'])
        best_size = min(self.results.items(), key=lambda x: x[1]['model_size'])
        best_f1 = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        
        print(f"\n🏆 Лучшая по точности: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.3f})")
        print(f"⚡ Лучшая по скорости: {best_speed[0]} ({best_speed[1]['samples_per_second']:.0f} изобр./сек)")
        print(f"📦 Самая компактная: {best_size[0]} ({best_size[1]['model_size']:.1f} MB)")
        print(f"🎯 Лучшая F1-Score: {best_f1[0]} ({best_f1[1]['f1_score']:.3f})")
        
        # Композитная оценка (взвешенная)
        composite_scores = {}
        for model_name, result in self.results.items():
            score = (
                result['accuracy'] * 0.4 +  # Точность 40%
                result['f1_score'] * 0.3 +  # F1 30%
                (result['samples_per_second'] / max([r['samples_per_second'] for r in self.results.values()])) * 0.2 +  # Скорость 20%
                (min([r['model_size'] for r in self.results.values()]) / result['model_size']) * 0.1  # Размер 10%
            )
            composite_scores[model_name] = score
        
        best_composite = max(composite_scores.items(), key=lambda x: x[1])
        print(f"\n🌟 РЕКОМЕНДУЕМАЯ МОДЕЛЬ (композитная оценка):")
        print(f"   {best_composite[0]} с оценкой {best_composite[1]:.3f}")
        
        # Объяснение рекомендации
        print(f"\n📝 Обоснование:")
        best_result = self.results[best_composite[0]]
        print(f"   • Точность: {best_result['accuracy']:.3f}")
        print(f"   • F1-Score: {best_result['f1_score']:.3f}")
        print(f"   • Скорость: {best_result['samples_per_second']:.0f} изобр./сек")
        print(f"   • Размер: {best_result['model_size']:.1f} MB")
        
        print("\n" + "="*80)
    
    def print_detailed_report(self):
        """Печать детального отчета"""
        print("\n" + "="*80)
        print("📊 ДЕТАЛЬНЫЙ ОТЧЕТ СРАВНЕНИЯ МОДЕЛЕЙ")
        print("="*80)
        
        if not self.results:
            print("❌ Нет результатов для отчета")
            return
        
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        print("\n🏆 РЕЙТИНГ МОДЕЛЕЙ ПО ТОЧНОСТИ:")
        print("-" * 50)
        for i, (model_name, result) in enumerate(sorted_results, 1):
            print(f"{i}. {model_name}: {result['accuracy']:.4f} (F1: {result['f1_score']:.4f})")
        
        print("\n📈 СВОДНАЯ ИНФОРМАЦИЯ:")
        print("-" * 50)
        for model_name, result in sorted_results:
            print(f"\n🔹 {model_name}:")
            print(f"   Точность (Accuracy): {result['accuracy']:.4f}")
            print(f"   F1-Score: {result['f1_score']:.4f}")
            print(f"   Время загрузки: {result['load_time']:.2f} сек")
            print(f"   Время предсказания: {result['inference_time']:.2f} сек")
            print(f"   Общее время: {result['total_time']:.2f} сек")
            print(f"   Скорость: {result['samples_per_second']:.1f} изобр./сек")
            print(f"   Размер модели: {result['model_size']:.2f} MB")
        
        # Детальные отчеты по классификации для каждой модели
        print("\n📋 ДЕТАЛЬНЫЕ ОТЧЕТЫ ПО КЛАССИФИКАЦИИ:")
        print("-" * 50)
        
        class_names = [self.reverse_label_map[i] for i in range(len(self.label_map))]
        
        for model_name, result in sorted_results:
            print(f"\n🎯 {model_name}:")
            print(classification_report(self.test_labels, result['predictions'], 
                                      target_names=class_names,
                                      zero_division=0))
    
    def run_comparison(self, test_sample_size=1000):
        """Запуск полного сравнения моделей"""
        print("\n" + "="*80)
        print("🚀 ЗАПУСК ПОЛНОГО СРАВНЕНИЯ МОДЕЛЕЙ")
        print("="*80)
        
        # Загрузка тестовых данных
        if not self.load_test_data(test_sample_size):
            return
        
        # Список моделей для сравнения
        models_to_evaluate = ['HOG+SVM', 'Random Forest', 'CNN']
        
        models_evaluated = 0
        
        # Оценка каждой модели
        for model_type in models_to_evaluate:
            if self.load_model_if_exists(model_type):
                if self.evaluate_model(model_type):
                    models_evaluated += 1
            else:
                print(f"❌ Модель {model_type} не найдена (пропускаем)")
        
        if models_evaluated == 0:
            print("❌ Не удалось оценить ни одну модель")
            return
        
        # Вывод результатов
        self.print_detailed_report()
        
        # Визуализация
        self.plot_comparison_results()
        
        print(f"\n" + "="*80)
        print(f"🎉 СРАВНЕНИЕ ЗАВЕРШЕНО!")
        print(f"✅ Оценено {models_evaluated} из {len(models_to_evaluate)} моделей")
        print("="*80)

def main():
    """Основная функция"""
    comparator = ModelComparator(data_path="C:\\Users\\danak\\Downloads\\1_Intro_Color_models\\Curs\\EuroSAT")
    comparator.run_comparison(test_sample_size=1000)

if __name__ == "__main__":
    main()