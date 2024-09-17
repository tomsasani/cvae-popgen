from torchvision.datasets import MNIST, CIFAR10, CelebA
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import torch
import torchvision

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
import pandas as pd
import itertools
from typing import List

import models
import generator_fake
import losses
import demographies
import params
import util

BATCH_SIZE = 128
LATENT_DIM = 3
INTERMEDIATE_DIM = 128
H, W = 128, 128
CHANNELS = 1
INPUT_DIM = H * W
LR = 1e-3
EPOCHS = 10
HIDDEN_DIMS = [32, 64, 128, 256, 512]
# HIDDEN_DIMS = [h * 2 for h in HIDDEN_DIMS]
DEVICE = torch.device("cuda")


mnist_transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.RandomCrop(size=(H, W)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

# train = CIFAR10(
#     "data/",
#     transform=mnist_transform,
#     train=True,
#     download=True,

# )
# test = CIFAR10(
#     "data/",
#     transform=mnist_transform,
#     train=False,
#     download=True,
# )


tg_train = torchvision.datasets.ImageFolder(
    "data/simulated/target/train/",
    transform=mnist_transform,
)
tg_test = torchvision.datasets.ImageFolder(
    "data/simulated/target/test/",
    transform=mnist_transform,
)

train_loader = DataLoader(
    dataset=tg_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    dataset=tg_test,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

PARAM_NAMES = ["N1", "N2", "T1", "T2", "growth"]

# initialize basic engine
ENGINE = generator_fake.Generator(
    demographies.simulate_exp,
    PARAM_NAMES,
    42,
    convert_to_rgb=True,
    n_snps=W,
    convert_to_diploid=True,
    seqlen=50_000,
    sort=True,
    filter_singletons=False,
)

sim_params = params.ParamSet()
RNG = np.random.default_rng(42)

# define parameter values we'll use for target dataset,
# in which we have a population size change in the admixed
# population
PARAM_VALUES = [
    [23_231, 29_962, 4_870, 581, 0.00531],
    [22_552, 3_313, 3_589, 1_050, 0.00535],
    [9_000, 5_000, 2_000, 350, 0.001],
]


def data_iterator():

    n_models = len(PARAM_VALUES)

    X, y = [], []

    counted = 0
    while counted < BATCH_SIZE:
        # pick a random model to simulate
        model_i = RNG.choice(n_models)
        model_params = PARAM_VALUES[model_i]

        region = ENGINE.sample_fake_region(
            [H],
            param_values=model_params,
        )

        n_batches_zero_padded = util.check_for_missing_data(region) > 0
        if n_batches_zero_padded:
            continue

        region = region[:, :2, :, :].astype(np.float32)

        X.append(region)
        y.append(model_i)

        counted += 1

    X = np.concatenate(X)
    y = np.array(y)

    X = torch.from_numpy(X).to(DEVICE)
    y = torch.from_numpy(y).to(DEVICE)

    return X, y


model = models.CNN(
    in_channels=CHANNELS,
    latent_dim=LATENT_DIM,
    kernel_size=3,
    stride=2,
    padding=1,
    hidden_dims=HIDDEN_DIMS,
    intermediate_dim=INTERMEDIATE_DIM,
    in_H=H,
)

model = model.to(DEVICE)
print (model)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print (pytorch_total_params)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

loss_fn = torch.nn.CrossEntropyLoss()

def train_loop_onthefly(model, loss_fn, optimizer):
    
    model.train()

    X, y = data_iterator()
       
    optimizer.zero_grad()

    preds = model(X)
    loss = loss_fn(preds, y)
    accuracy = torch.sum(torch.argmax(preds, dim=1) == y) / X.shape[0]
    loss.backward()
    optimizer.step()

    return loss.item(), accuracy.item()


def train_loop(model, dataloader, loss_fn, optimizer):
    model.train()

    n_batches = len(dataloader)
    total_loss, total_acc = 0, 0
    n = 0
    for batch_idx, (x, y) in tqdm.tqdm(enumerate(dataloader)):

        optimizer.zero_grad()

        n += x.shape[0]

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        preds = model(x)
        loss = loss_fn(preds, y)

        loss.backward()
        optimizer.step()

        accuracy = torch.sum(torch.argmax(preds, dim=1) == y) / x.shape[0]
        total_acc += accuracy.item()
        total_loss += loss.item()

    return (
        total_loss / n_batches,
        total_acc / n_batches,
    )

def test_loop(model, dataloader, loss_fn):

    model.eval()

    n_batches = len(dataloader)
    total_loss, total_acc = 0, 0
    n = 0
    with torch.no_grad():
        for batch_idx, (x, y) in tqdm.tqdm(enumerate(dataloader)):
            n += x.shape[0]

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            preds = model(x)
            loss = loss_fn(preds, y)
            accuracy = torch.sum(torch.argmax(preds, dim=1) == y) / x.shape[0]
            total_acc += accuracy.item()
            total_loss += loss.item()

    return (
        total_loss / n_batches,
        total_acc / n_batches,
    )


print("Start training CNN...")

res = []

for epoch in range(EPOCHS):

    train_loss, train_acc = train_loop(model, train_loader, loss_fn, optimizer)
    test_loss, test_acc = test_loop(model, test_loader, loss_fn)
    for loss, loss_name in zip((train_loss, test_loss), ("train", "test")):
        res.append({"batch": epoch, "loss_kind": loss_name, "loss_val": loss})
    for acc, acc_name in zip((train_acc, test_acc), ("train", "test")):
        res.append(({"epoch": epoch, "acc_kind": acc_name, "acc_val": acc,}) )
    print(
            "Epoch",
            epoch + 1,
            "complete!",
            "\tTrain Loss: ",
            train_loss,
            "\tTest Loss: ",
            test_loss,
            "\tTrain Acc: ",
            train_acc,
            "\tTest Acc: ",
            test_acc,
        )

res_df = pd.DataFrame(res)

f, ax = plt.subplots()
# sns.lineplot(data=res_df, x="epoch", y="loss_val", hue="loss_kind", ax=ax)
sns.lineplot(data=res_df, x="batch", y="loss", ax=ax)
sns.lineplot(data=res_df, x="batch", y="acc", ax=ax)
f.tight_layout()
sns.despine(ax=ax)
f.savefig("loss.png", dpi=200)

print("Finish!!")

model.eval()
