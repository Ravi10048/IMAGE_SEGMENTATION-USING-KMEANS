import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image

img=cv2.imread("cell.jpg")
print(img.shape)
img2=img.reshape((-1,3))
print(img2.shape)
img2=np.float32(img2)
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

k=4

attempts=10

ret,label,center=cv2.kmeans(img2,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
centers = np.uint8(center)
segmented_data = centers[label.flatten()]
  
# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((img.shape))
plt.imshow(segmented_image)
cv2.imwrite('segmented.jpg',segmented_image)  
plt.show()

