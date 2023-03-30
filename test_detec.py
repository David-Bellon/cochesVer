import torch
from torch.utils.data import DataLoader, Dataset
import os
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

detection = torch.load("detection.pt")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


test = os.listdir("test/images")

detection.eval()
with torch.no_grad():
    for i, image in enumerate(test):
        cv2_image = cv2.imread("test/images/" + image)
        pil_image = Image.open("test/images/" + image)
        tensor = ToTensor()(Resize((32, 32))(pil_image)).to(device)
        tensor = tensor[None, :, :, :]
        width = cv2_image.shape[1]
        height = cv2_image.shape[0]
        out = detection(tensor)
        tlc = int(out[0][0] * (width - 1))
        tlr = int(out[0][1] * (width - 1))
        brc = int(out[0][2] * (height - 1))
        brr = int(out[0][3] * (height - 1))
        image = cv2.rectangle(cv2_image, (tlc, tlr), (brc, brr), (255, 0, 0), 2)
        plt.imshow(image)
        plt.show()