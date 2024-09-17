from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFilter
import pathlib
import os

H, W = 64, 64
rng = np.random.default_rng(42)

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


cifar_train = CIFAR10(
    "data/",
    transform=mnist_transform,
    train=True,
    download=True,
)
cifar_test = CIFAR10(
    "data/",
    transform=mnist_transform,
    train=False,
    download=True,
)

cifar_frog_idxs = [i for i,v in enumerate(cifar_train.targets) if v == 6]
# choose half to be superimposed and half to be background
fg_idxs = np.random.choice(cifar_frog_idxs, size=int(0.5 * len(cifar_frog_idxs)), replace=False)
bg_idxs = [i for i in cifar_frog_idxs if i not in fg_idxs]

cifar_fg_data = torch.utils.data.Subset(cifar_train, fg_idxs)
cifar_bg_data = torch.utils.data.Subset(cifar_train, bg_idxs)

# randomly augment with probability p
prob = 0.5

def output_corrupted(
    tg_dataloader,
    bg_dataloader,
    outpref: str = "test",
    prob: float = 0.5,
):
    
    bg_iter = iter(bg_dataloader)
    for i, (tg_x, tg_y) in enumerate(tg_dataloader):

        if i % 100 == 0: print (i)

        label = tg_y[0]
        tg_img = tg_x.numpy()[0, 0, :, :]

        # random image from background
        try:
            bg_img, _ = next(bg_iter)
        except StopIteration:
            bg_iter = iter(bg_dataloader)
            bg_img, _ = next(bg_iter)

        bg_img = bg_img.numpy()[0, 0, :, :]

        # augment if probability is <= p
        if rng.uniform() <= prob:
            new_image = bg_img
        # otherwise just keep the image as the background
        else:
            new_image = 0.5 * tg_img + bg_img
        
        new_image /= np.max(new_image)
        new_image *= 255
        new_image = np.uint8(new_image)

        outpath = f"data/corrupted/target/{outpref}/{label}"
       
        p = pathlib.Path(outpath)
        if not p.is_dir():
            p.mkdir(parents=True)

        new_img = Image.fromarray(new_image, mode="L")
        new_img.save(f"{outpath}/{i}.png")

cifar_fg = DataLoader(dataset=cifar_fg_data, batch_size=1, shuffle=True)
output_corrupted(mnist_train, cifar_fg, outpref="train", prob=prob)
cifar_fg = DataLoader(dataset=cifar_fg_data, batch_size=1, shuffle=True)
output_corrupted(mnist_test, cifar_fg, outpref="test", prob=prob)

cifar_bg = DataLoader(dataset=cifar_bg_data, batch_size=1, shuffle=True)

for i, (img, label) in enumerate(cifar_bg):

    frog_img = img.numpy()[0, 0, :, :]
    frog_img /= np.max(frog_img)
    frog_img *= 255
    frog_img = np.uint8(frog_img)
    new_img = Image.fromarray(frog_img, mode="L")

    new_img.save(f"data/corrupted/background/0/{i}.png")
