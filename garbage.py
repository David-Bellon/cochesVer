import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
import cv2

def yolobbox2bbox(x,y,w,h):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return x1, y1, x2, y2

image_cv2 = cv2.imread("train/images/" + "000006.jpg")
image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image_cv2)
image = ToTensor()(image_pil)
width = image.shape[2]
height = image.shape[1]
tlc, tlr, brc, brr = yolobbox2bbox(0.3699, 0.4840, 0.0164, 0.0402)
tlc = int(tlc * (width - 1))
brc = int(brc * (width - 1))
tlr = int(tlr * (height - 1))
brr = int(brr * (height - 1))
image = cv2.rectangle(image_cv2, (tlc, tlr), (brc, brr), (255, 0, 0), 2)
copr = image_pil.crop((tlc, tlr, brc, brr))
plt.imshow(image)
plt.imshow(copr)
plt.show()