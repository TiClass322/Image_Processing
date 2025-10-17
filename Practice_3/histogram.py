import cv2
import matplotlib.pyplot as plt

# Load the image in grayscale
image_sar3 = cv2.imread('sar_3.jpg', 0)

# Calculate histogram
hist = cv2.calcHist([image_sar3], [0], None, [256], [0, 256])

# Plot histogram
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.savefig('histogram.png')

print("Histogram saved to histogram.png")
