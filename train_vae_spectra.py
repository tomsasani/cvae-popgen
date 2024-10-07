from torchvision.datasets import MNIST, CIFAR10, CelebA
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import torch
import torchvision
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

LR = 5e-4
EPOCHS = 25

BATCH_SIZE = 64
LATENT_DIM = 2
INTERMEDIATE_DIM = 64

DEVICE = torch.device("cuda")

bg = np.load("data/background_spectra.npz")
tg = np.load("data/target_spectra.npz")

X_bg, y_bg = torch.from_numpy(bg["X"].astype(np.float32)), torch.from_numpy(bg["y"].astype(np.float32))
X_tg, y_tg = torch.from_numpy(tg["X"].astype(np.float32)), torch.from_numpy(tg["y"].astype(np.float32))

tg = MyDataset(X_tg, y_tg, transform=None)#transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]))#transforms.Normalize([X_mean, ], [X_std, ]))

tg_train, tg_test = torch.utils.data.random_split(tg, [0.8, 0.2])

# tg_train = MNIST("data/", train=True, transform=transforms.ToTensor())
# tg_test = MNIST("data/", train=False, transform=transforms.ToTensor())

# NOTE: these must be shuffled!!!
tg_train = DataLoader(dataset=tg_train, batch_size=BATCH_SIZE, shuffle=True)
tg_test = DataLoader(dataset=tg_test, batch_size=BATCH_SIZE, shuffle=True)


encoder = models.EncoderFC(
    latent_dim=LATENT_DIM,
    intermediate_dim=INTERMEDIATE_DIM,
    in_W=96,
    
)

decoder = models.DecoderFC(
    latent_dim=LATENT_DIM,
    intermediate_dim=INTERMEDIATE_DIM,
    in_W=96
)

model = models.VAE(
    encoder=encoder,
    decoder=decoder,
)
model = model.to(DEVICE)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print (pytorch_total_params)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, )

loss_fn = losses.VAELoss()

def train_loop(model, dataloader, loss_fn, optimizer):
    
    model.train()

    n_batches = len(dataloader)
    total_loss = 0
    batch_size = None
    for batch_idx, (x, _) in tqdm.tqdm(enumerate(dataloader)):
        if batch_size is None:
            batch_size = x.shape[0]

        x = x.to(DEVICE)


        optimizer.zero_grad()

        x_hat, mean, log_var, z = model(x)

        # x = x.reshape(x.shape[0], 28 * 28)

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

            # x = x.reshape(28 * 28)

            ind = np.arange(x.shape[0])

            axarr[i, 0].bar(ind, x, 1)
            axarr[i, 1].bar(ind, x_hat, 1)

            axarr[i, 1].set_title(cosine_similarity(x.reshape(1, -1), x_hat.reshape(1, -1))[0][0])

    f.tight_layout()
    # plt.subplots_adjust(wspace=0, hspace=0)
    f.savefig(plot_name, dpi=200)
    plt.close()

print("Start training VAE...")

res = []

for epoch in range(EPOCHS):

    plot_example(model, tg_test, f"fig/reconstructions/vae/{epoch}.png")
   
    train_loss = train_loop(model, tg_train, loss_fn, optimizer)
    test_loss = test_loop(model, tg_test, loss_fn)
    for loss, loss_name in zip((train_loss, test_loss), ("train", "test")):
        res.append({"epoch": epoch, "loss_kind": loss_name, "loss_val": loss,})

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

f, ax = plt.subplots()
sns.lineplot(data=res_df, x="epoch", y="loss_val", hue="loss_kind", ax=ax)
f.tight_layout()
sns.despine(ax=ax)
f.savefig("vae.loss.png", dpi=200)

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
labels = np.concatenate(labels)# [:, glasses_attr_idx]
print (reps.shape, labels.shape)
clf = LogisticRegressionCV(cv=5)
clf.fit(reps, labels)
preds = clf.predict(reps)

f, ax = plt.subplots()
if LATENT_DIM > 2:
    clf = PCA(n_components=2)
    reps = clf.fit_transform(reps)
ax.scatter(reps[:, 0], reps[:, 1], c=labels, alpha=0.5)
ax.set_title(round(silhouette_score(reps, labels), 3))
f.savefig("vae.coords.png", dpi=200)


def plot_reconstructed(model, r0=(-4, 4), r1=(-4, 4), n=12):

    f, axarr = plt.subplots(n, n, figsize=(8, 8))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            # sample a salient vector
            z = torch.Tensor([[x, y]]).to(DEVICE)
            # sample 0s for the irrelevant vector
            x_hat = model.decoder(z)
            x_hat = x_hat.to('cpu').detach().numpy()[0]
            ind = np.arange(x_hat.shape[0])

            axarr[i, j].bar(ind, x_hat, 1)
            axarr[i, j].set_xticks([])
            axarr[i, j].set_yticks([])
    f.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    f.savefig("vae.recons.png")

# if LATENT_DIM == 2:
#     plot_reconstructed(model)
