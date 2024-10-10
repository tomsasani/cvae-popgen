from torchvision.datasets import MNIST, CIFAR10, CelebA
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import torch
from datasets import MyDataset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
import pandas as pd
from typing import List
import models
import losses
import itertools
from datasets import MyDataset
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
import argparse
from make_training_data_cancer import poisson_resample

def train_loop(model, dataloader, loss_fn, optimizer):

    model.train()

    n_batches = len(dataloader)
    total_loss = 0
    batch_size = None
    for batch_idx, (x, _) in tqdm.tqdm(enumerate(dataloader)):
        if batch_size is None:
            batch_size = x.shape[0]

        if x.shape[0] == 1:
            continue

        x = x.to(DEVICE)
        optimizer.zero_grad()

        x_hat, mean, log_var, z = model(x)
        loss = loss_fn(x, x_hat, mean, log_var)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / (n_batches * batch_size)


def test_loop(model, dataloader, loss_fn):

    model.eval()

    n_batches = len(dataloader)
    total_loss = 0

    with torch.no_grad():
        batch_size = None
        for batch_idx, (x, _) in tqdm.tqdm(enumerate(dataloader)):
            if batch_size is None:
                batch_size = x.shape[0]

            if x.shape[0] == 1: continue

            x = x.to(DEVICE)

            x_hat, mean, log_var, z = model(x)
            # x = x.reshape(x.shape[0], 28 * 28)

            loss = loss_fn(x, x_hat, mean, log_var)
            total_loss += loss.item()

    return total_loss / (n_batches * batch_size)


def plot_example(model, dataloader, plot_name: str):

    f, axarr = plt.subplots(5, 2, figsize=(8, 12))

    dataloader_iter = iter(dataloader)

    model.eval()
    with torch.no_grad():

        for i in range(5):
            # grab first example
            xs, ys = next(dataloader_iter)
            # add batch dimension to single example
            x = torch.unsqueeze(xs[0].to(DEVICE), dim=0)
            x_hat, mu, log_var, z = model(x)

            x = x.cpu().numpy()[0]
            x_hat = x_hat.cpu().numpy()[0]

            x_hat = np.exp(x_hat)

            ind = np.arange(x.shape[0])

            axarr[i, 0].bar(ind, x, 1, color=CMAP)
            axarr[i, 1].bar(ind, x_hat, 1, color=CMAP)

    f.tight_layout()
    # plt.subplots_adjust(wspace=0, hspace=0)
    f.savefig(plot_name, dpi=200)
    plt.close()

def plot_reconstructed(model, r0=(-4, 4), r1=(-4, 4), n=4):

    f, axarr = plt.subplots(n, n, figsize=(8, 8))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            # sample a salient vector
            z = torch.Tensor([[x, y]]).to(DEVICE)
            # sample 0s for the irrelevant vector
            x_hat = model.decoder(z)
            x_hat = x_hat.to("cpu").detach().numpy()[0]
            ind = np.arange(x_hat.shape[0])

            axarr[i, j].bar(ind, x_hat, 1, color=CMAP)
            axarr[i, j].set_xticks([])
            axarr[i, j].set_yticks([])
    f.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    f.savefig("vae.recons.png")


LR = 1e-3
EPOCHS = 25

BATCH_SIZE = 128
LATENT_DIM = 2
HIDDEN_DIMS = [128, 256]

CMAP = np.repeat(["blue", "black", "red", "grey", "green", "pink"], 16)

DEVICE = torch.device("mps")

rng = np.random.default_rng(42)

# bg = np.load("data/background_spectra.npz")
# tg = np.load("data/target_spectra.npz")

def main(args):

    bg = np.load(args.bg)
    tg = np.load(args.tg)

    X_bg, y_bg = bg["X"].astype(np.float32), bg["y"].astype(np.float32)
    X_tg, y_tg = tg["X"].astype(np.float32), tg["y"].astype(np.float32)

    if args.resample:
        # get minimum count across both
        min_count = min(
            [
                np.min(np.sum(X_bg, axis=1)),
                np.min(np.sum(X_tg, axis=1)),
            ]
        )

        X_bg = poisson_resample(rng, X_bg, min_count)
        X_tg = poisson_resample(rng, X_tg, min_count)

        X_bg, y_bg = torch.from_numpy(X_bg), torch.from_numpy(y_bg)
        X_tg, y_tg = torch.from_numpy(X_tg), torch.from_numpy(y_tg)

    tg = MyDataset(X_tg, y_tg, transform=transforms.ToTensor())

    tg_train, tg_test = torch.utils.data.random_split(tg, [0.8, 0.2])

    # NOTE: these must be shuffled!!!
    tg_train = DataLoader(
        dataset=tg_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    tg_test = DataLoader(
        dataset=tg_test,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    encoder = models.EncoderFC(
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
        in_W=96,
    )

    decoder = models.DecoderFC(
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
        in_W=96,
    )

    model = models.VAE(
        encoder=encoder,
        decoder=decoder,
    )
    model = model.to(DEVICE)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    loss_fn = losses.VAELoss(losses.PoissonMultinomial())

    print("Start training VAE...")

    res = []

    for epoch in range(args.epochs):

        plot_example(model, tg_test, f"fig/reconstructions/vae/{epoch}.png")

        train_loss = train_loop(model, tg_train, loss_fn, optimizer)
        test_loss = test_loop(model, tg_test, loss_fn)
        for loss, loss_name in zip((train_loss, test_loss), ("train", "test")):
            res.append(
                {
                    "epoch": epoch,
                    "loss_kind": loss_name,
                    "loss_val": loss,
                }
            )

        print(
            "\tEpoch",
            epoch + 1,
            "complete!",
            "\tAverage Train Loss: ",
            train_loss,
            "\tAverage Test Loss: ",
            test_loss,
        )

    res_df = pd.DataFrame(res)

    print("Finish!!")

    model.eval()

    f, ax = plt.subplots()

    reps = []
    labels = []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm.tqdm(tg_test)):
            x = x.to(DEVICE)

            x_hat, mu, log_var, z = model(x)
            z = z.cpu().numpy()
            reps.append(z)
            labels.append(y)

    reps = np.concatenate(reps)
    labels = np.concatenate(labels)

    clf = LogisticRegressionCV(cv=5)
    clf.fit(reps, labels)
    preds = clf.predict(reps)

    f, ax = plt.subplots()
    if LATENT_DIM > 2:
        clf = PCA(n_components=2)
        reps = clf.fit_transform(reps)
    ax.scatter(reps[:, 0], reps[:, 1], c=labels, alpha=0.5)
    silhouette = round(silhouette_score(reps, labels), 3)
    accuracy = round(accuracy_score(labels, preds), 3)

    res_df["salient_silhouette"] = silhouette
    res_df["salient_accuracy"] = accuracy

    ax.set_title(silhouette)
    f.savefig("vae.coords.png", dpi=200)

    res_df.to_csv(args.out, index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bg")
    p.add_argument("--tg")
    p.add_argument("--out")
    p.add_argument("-epochs", type=int, default=10)
    p.add_argument("-resample", action="store_true")
    args = p.parse_args()
    main(args)
