import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from arch import Model
from tqdm import tqdm
import cv2


def yolobbox2bbox(x,y,w,h):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return x1, y1, x2, y2


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class CustomData(Dataset):
    def __init__(self):
        self.images = os.listdir("train/images")
        self.labels = os.listdir("train/labels")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        pil_image = Image.open("train/images/" + self.images[index])
        cv2_image = cv2.imread("train/images/" + self.images[index])
        real_image = ToTensor()(Resize((32, 32))(pil_image))
        width = pil_image.width
        height = pil_image.height
        with open("train/labels/" + self.labels[index], "r") as f:
            data = f.readline().split(" ")
            tensor = torch.zeros(7)
            tensor[int(data[0])] = 1
            x, y, w, h = float(data[1]), float(data[2]), float(data[3]), float(data[4].split("\n")[0])
            tlc, tlr, brc, brr = yolobbox2bbox(x, y, w, h)
        rect = torch.tensor([tlc, tlr, brc, brr])
        tlc = int(tlc * (width - 1))
        brc = int(brc * (width - 1))
        tlr = int(tlr * (height - 1))
        brr = int(brr * (height - 1))
        image = pil_image.crop((tlc, tlr, brc, brr))
        image = Resize((32, 32))(image)
        image = ToTensor()(image)
        aux = cv2.rectangle(cv2_image, (tlc, tlr), (brc, brr), (255, 0, 0), 2)
        plt.imshow(aux)
        plt.show()
        plt.imshow(ToPILImage()(image))
        plt.show()
        return image.to(device), tensor.to(device), real_image.to(device), rect.to(device)


train_data = CustomData()

data = DataLoader(
    train_data,
    batch_size=150,
    shuffle=True,
)

classifier = Model(7).to(device)
loss_classifier = nn.CrossEntropyLoss()
detection = Model(4).to(device)
loss_detection = nn.MSELoss()

optim_c = torch.optim.Adam(classifier.parameters(), lr=0.001)
optim_d = torch.optim.Adam(detection.parameters(), lr=0.001)


def train_all(image, label, full_image, rect):
    optim_c.zero_grad()
    optim_d.zero_grad()

    out_c = classifier(image)
    out_d = detection(full_image)

    loss_c = loss_classifier(out_c, label)
    loss_d = loss_detection(out_d, rect)

    loss_c.backward()
    loss_d.backward()

    optim_c.step()
    optim_d.step()

    return loss_c, loss_d


epochs = 50
for epoch in range(epochs):
    epoch_loss_c = 0.0
    epoch_loss_d = 0.0
    for i, (image, label, real_img, rect) in tqdm(enumerate(data)):
        out_c, out_d = train_all(image, label, real_img, rect)
        epoch_loss_c += out_c
        epoch_loss_d += out_d

    print(f"Epoch: {epoch} ----- Loss Classifier: {epoch_loss_c.item()/i} ----- Loss Detection: {epoch_loss_d.item()/i}")


torch.save(classifier, "classifier.pt")
torch.save(detection, "detection.pt")