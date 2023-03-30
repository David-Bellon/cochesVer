import torch
from torch.utils.data import DataLoader, Dataset
import os
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

classifier = torch.load("classifier.pt")
detection = torch.load("detection.pt")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

def yolobbox2bbox(x,y,w,h):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return x1, y1, x2, y2


class CustomData(Dataset):
    def __init__(self):
        self.images = os.listdir("test/images")
        self.labels = os.listdir("test/labels")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        pil_image = Image.open("test/images/" + self.images[index])
        real_image = ToTensor()(Resize((32, 32))(pil_image))
        width = pil_image.width
        height = pil_image.height
        with open("test/labels/" + self.labels[index], "r") as f:
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
        return image.to(device), tensor.to(device), real_image.to(device), rect.to(device)


data = CustomData()
test = DataLoader(
    data,
    batch_size=120,
    shuffle=True,
)


number_samples = len(data)
print(number_samples)
rights = 0
wrongs = 0
with torch.no_grad():
    for i, (image, label, real_img, rect) in tqdm(enumerate(test)):
        outs = classifier(image)
        detec = detection(real_img)
        for j, value in enumerate(image):
            print(f"Real Label: {torch.argmax(label[j])} ----- Predicted Label: {torch.argmax(outs[j])}")
            if torch.argmax(label[j]) == torch.argmax(outs[j]):
                rights += 1
            else:
                wrongs += 1


print(f"Rights: {rights} ----- Wrongs: {wrongs}")
print(f"Percentage right: {(rights/number_samples) * 100}%")