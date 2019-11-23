import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread("1.jpg")
# img = cv2.resize(img, (32, 32))
img = img/255
for i in range(4):
    plt.subplot(3, 3, i+1)
    plt.imshow((img*255).astype(np.uint8))
    plt.axis('off')

    plt.savefig('test.png', cmap='bgr')