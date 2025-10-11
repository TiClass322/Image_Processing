
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import os

# --- Настройка путей ---
# Получаем директорию, в которой находится скрипт
script_dir = os.path.dirname(os.path.abspath(__file__))

# Папка для результатов будет создана в той же директории
output_dir = os.path.join(script_dir, "results")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- Задача 1: Загрузка изображения ---
image_path = os.path.join(script_dir, "sar_1_gray.jpg")
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if gray_image is None:
    print(f"Не удалось загрузить изображение: {image_path}")
    exit()

print("--- Задача 1: Изображение sar_1_gray.jpg загружено ---")
print(f"Размер: {gray_image.shape}")
# Сохраняем исходное изображение для сравнения
original_image_path = os.path.join(output_dir, "original_image.png")
cv2.imwrite(original_image_path, gray_image)


# --- Задача 2: Построение гистограммы ---
histogram_path = os.path.join(output_dir, "histogram.png")
plt.hist(gray_image.ravel(), 256, [0, 256])
plt.title("Histogram for sar_1_gray.jpg")
plt.xlabel("Intensity")
plt.ylabel("Pixel Count")
plt.savefig(histogram_path)
plt.close()
print(f"\n--- Задача 2: Гистограмма сохранена в {histogram_path} ---")


# --- Задача 3: Гамма-коррекция ---
def gamma_correction(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

gamma_low = 0.5
gamma_high = 1.5
corrected_low = gamma_correction(gray_image, gamma_low)
corrected_high = gamma_correction(gray_image, gamma_high)

gamma_low_path = os.path.join(output_dir, f"gamma_{gamma_low}.png")
gamma_high_path = os.path.join(output_dir, f"gamma_{gamma_high}.png")
cv2.imwrite(gamma_low_path, corrected_low)
cv2.imwrite(gamma_high_path, corrected_high)
print(f"\n--- Задача 3: Изображения с гамма-коррекцией (gamma={gamma_low}, gamma={gamma_high}) сохранены ---")


# --- Задача 4: Сравнение изображений (MSE, SSIM) ---
mse_low = mse(gray_image, corrected_low)
ssim_low = ssim(gray_image, corrected_low)
mse_high = mse(gray_image, corrected_high)
ssim_high = ssim(gray_image, corrected_high)

print("\n--- Задача 4: Сравнение с гамма-коррекцией ---")
print(f"Gamma < 1 (gamma={gamma_low}):")
print(f"  MSE: {mse_low:.2f}")
print(f"  SSIM: {ssim_low:.2f}")
print(f"Gamma > 1 (gamma={gamma_high}):")
print(f"  MSE: {mse_high:.2f}")
print(f"  SSIM: {ssim_high:.2f}")


# --- Задача 5: Статистическая цветокоррекция ---
image_for_stats_path = os.path.join(script_dir, "sar_2_color.jpg")
image_for_stats = cv2.imread(image_for_stats_path, cv2.IMREAD_GRAYSCALE)
if image_for_stats is not None:
    eq_gray = cv2.equalizeHist(image_for_stats)
    
    (target_mean, target_std) = cv2.meanStdDev(gray_image)
    (source_mean, source_std) = cv2.meanStdDev(eq_gray)

    stat_corrected_image = (gray_image - target_mean) * (source_std / target_std) + source_mean
    stat_corrected_image = np.clip(stat_corrected_image, 0, 255).astype("uint8")
    
    stat_corrected_path = os.path.join(output_dir, "stat_corrected.png")
    cv2.imwrite(stat_corrected_path, stat_corrected_image)
    print("\n--- Задача 5: Изображение со статистической коррекцией сохранено ---")
else:
    print("\n--- Задача 5: Не удалось загрузить sar_2_color.jpg для статистической коррекции ---")


# --- Задача 6: Пороговая фильтрация ---
ret, thresh_binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
ret, thresh_binary_inv = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh_trunc = cv2.threshold(gray_image, 127, 255, cv2.THRESH_TRUNC)
ret, thresh_tozero = cv2.threshold(gray_image, 127, 255, cv2.THRESH_TOZERO)
ret, thresh_tozero_inv = cv2.threshold(gray_image, 127, 255, cv2.THRESH_TOZERO_INV)

thresh_binary_path = os.path.join(output_dir, "thresh_binary.png")
thresh_binary_inv_path = os.path.join(output_dir, "thresh_binary_inv.png")
thresh_trunc_path = os.path.join(output_dir, "thresh_trunc.png")
thresh_tozero_path = os.path.join(output_dir, "thresh_tozero.png")
thresh_tozero_inv_path = os.path.join(output_dir, "thresh_tozero_inv.png")

cv2.imwrite(thresh_binary_path, thresh_binary)
cv2.imwrite(thresh_binary_inv_path, thresh_binary_inv)
cv2.imwrite(thresh_trunc_path, thresh_trunc)
cv2.imwrite(thresh_tozero_path, thresh_tozero)
cv2.imwrite(thresh_tozero_inv_path, thresh_tozero_inv)
print("\n--- Задача 6: Результаты пороговой фильтрации сохранены ---")

print(f"\nСкрипт завершен. Все результаты в папке {output_dir}.")
