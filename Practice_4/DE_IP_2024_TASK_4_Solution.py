import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


# 1. Подберите парамтеры алгоритма разрастания регионов так, чтобы был выделен весь участок газона.
# 2. Реализуйте вычисление критерия однородности, отличного от представленного. Сравните результаты.
# 3. Применить алгоритм сегментации watershed+distance transform для задачи подсчета пальмовых деревьев.


# 1
image = cv2.imread('sar_1.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(image_gray, cmap="gray")
plt.show()

def homo_average(img, mask, point, T):
    av_val = img[mask > 0].sum() / np.count_nonzero(img[mask > 0])

    if abs(av_val - img[point]) <= T:
        return True

    return False


def region_growing(image, seed_point, homo_fun, r, T):
    mask = np.zeros(image_gray.shape, np.uint8)
    mask[seed_point] = 1
    count = 1
    while count > 0:
        count = 0
        local_mask = np.zeros(image_gray.shape, np.uint8)
        for i in range(r, image.shape[0] - r):
            for j in range(r, image.shape[1] - r):
                if mask[i, j] == 0 and mask[i - r:i + r, j - r: j + r].sum() > 0:
                    if homo_fun(image, mask, (i, j), T):
                        local_mask[i, j] = 1
        count = np.count_nonzero(local_mask)
        print(count)
        mask += local_mask

    return mask * 255

seed_point = (157,130)
mask = region_growing(image_gray,seed_point,homo_average,1, 25)
print(mask)
plt.imshow(mask, cmap='gray')
plt.show()


# 2
def homo_median(image, mask, point, T):  # медиана вместо среднего (устойчивее к шуму, в итоге меньше черных точек на участке)
    region_vals = image[mask > 0]
    med_val = np.median(region_vals)
    return abs(med_val - image[point]) <= T


seed_point = (157,130)
mask = region_growing(image_gray,seed_point,homo_median,1, 25)
print(mask)
plt.imshow(mask, cmap='gray')
plt.show()


# 3
image = cv2.imread('palm_1.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(image_gray, cmap="gray")
plt.show()

ret, thresh = cv2.threshold(image_gray,0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(thresh, cmap='gray')
plt.show()

dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
plt.imshow(dist, cmap="gray")
plt.show()

dist_blur = cv2.GaussianBlur(dist, (7,7), 0)
ret, sure_fg = cv2.threshold(dist_blur, 0.3 * dist.max(), 255, cv2.THRESH_BINARY)
plt.imshow(sure_fg, cmap="gray")
plt.show()

sure_fg = sure_fg.astype(np.uint8)
ret, markers = cv2.connectedComponents(sure_fg)
plt.imshow(markers, cmap="gray")
plt.show()

markers = cv2.watershed(image, markers)
print(len(np.unique(markers)))  # примерное кол-во пальм
plt.imshow(markers, cmap="gray")
plt.show()
