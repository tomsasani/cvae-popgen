from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFilter

import os

H, W = 64, 64

mnist_transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize(size=(H, W)),
        transforms.ToTensor(),
    ]
)


mnist_train = MNIST(
        "data/",
        transform=mnist_transform,
        train=True,
        download=True,
    )
# get 0,1,2 idxs
digit_idxs = [i for i,v in enumerate(mnist_train.targets) if v in (0, 1, 2)]
mnist_train = torch.utils.data.Subset(mnist_train, digit_idxs)


mnist_train = DataLoader(
    mnist_train,
    batch_size=1,
    shuffle=True,
)

mnist_test = MNIST(
    "data/",
    transform=mnist_transform,
    train=False,
    download=True,
)

# get 0,1,2 idxs
digit_idxs = [i for i,v in enumerate(mnist_test.targets) if v in (0, 1, 2)]
mnist_test = torch.utils.data.Subset(mnist_test, digit_idxs)

mnist_test = DataLoader(
    mnist_test,
    batch_size=1,
    shuffle=True,
)


cifar_data = CIFAR10(
    "data/",
    transform=mnist_transform,
    train=True,
    download=True,
)

cifar_frog_idxs = [i for i,v in enumerate(cifar_data.targets) if v == 6]
# choose half to be superimposed and half to be background
fg_idxs = np.random.choice(cifar_frog_idxs, size=int(0.5 * len(cifar_frog_idxs)), replace=False)
bg_idxs = [i for i in cifar_frog_idxs if i not in fg_idxs]

cifar_fg_data = torch.utils.data.Subset(cifar_data, fg_idxs)
cifar_bg_data = torch.utils.data.Subset(cifar_data, bg_idxs)

print (len(fg_idxs))
print (len(bg_idxs))

cifar_fg = DataLoader(dataset=cifar_fg_data, batch_size=1, shuffle=True)
cifar_fg_iter = iter(cifar_fg)
for i, (img, label) in enumerate(mnist_train):

    # if i > N_TRAIN: break
    if i % 100 == 0: print (i)
    # if i > 100: break

    label = label[0]
    
    digit_img = img.numpy()[0, 0, :, :]
    
    # random frog from fg
    try:
        frog_img, _ = next(cifar_fg_iter)
    except StopIteration:
        cifar_fg_iter = iter(cifar_fg)
        frog_img, _ = next(cifar_fg_iter)

    frog_img = frog_img.numpy()[0, 0, :, :]

    new_image = 0.5 * digit_img + frog_img
    new_image /= np.max(new_image)
    new_image *= 255
    new_image = np.uint8(new_image)

    if not os.path.isdir(f"data/corrupted/train/{label}"):
        os.mkdir(f"data/corrupted/train/{label}")
    
    new_img = Image.fromarray(new_image, mode="L")
    new_img.save(f"data/corrupted/train/{label}/{i}.png")

print (i)

cifar_bg = DataLoader(dataset=cifar_bg_data, batch_size=1, shuffle=True)

for i, (img, label) in enumerate(cifar_bg):
    # try:
    #     img, _ = next(cifar_bg_iter)
    # except StopIteration:
    #     cifar_bg_iter = iter(cifar_bg)
    #     img, _ = next(cifar_bg_iter)

    frog_img = img.numpy()[0, 0, :, :]
    frog_img /= np.max(frog_img)
    frog_img *= 255
    frog_img = np.uint8(frog_img)
    new_img = Image.fromarray(frog_img, mode="L")

    new_img.save(f"data/background_frog/0/{i}.png")
