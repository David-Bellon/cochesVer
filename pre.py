import os

labels = os.listdir("train/labels")
for i, j in enumerate(labels):
    with open("train/labels/" + j, "r") as f:
        data = f.readline().split(" ")
    try:
        int(data[0])
    except:
        print(data)
        print(i)

#os.remove("train/images/" + os.listdir("train/images")[4910])
#os.remove("train/labels/" + os.listdir("train/labels")[4910])