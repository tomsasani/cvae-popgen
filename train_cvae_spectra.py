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
from losses import PoissonMultinomial

LR = 5e-4
EPOCHS = 10

BATCH_SIZE = 128
LATENT_DIM_S = 2
LATENT_DIM_Z = 2
HIDDEN_DIMS = [128, 256]

CMAP = np.repeat(["blue", "black", "red", "grey", "green", "pink"], 16)


DEVICE = torch.device("cuda")

bg = np.load("data/background_spectra.npz")
tg = np.load("data/target_spectra.npz")

X_bg, y_bg = torch.from_numpy(bg["X"].astype(np.float32)), torch.from_numpy(bg["y"].astype(np.float32))
X_tg, y_tg = torch.from_numpy(tg["X"].astype(np.float32)), torch.from_numpy(tg["y"].astype(np.float32))

tg = MyDataset(X_tg, y_tg, transform=transforms.ToTensor())
bg = MyDataset(X_bg, y_bg, transform=transforms.ToTensor())

tg_train, tg_test = torch.utils.data.random_split(tg, [0.8, 0.2])
bg_train, bg_test = torch.utils.data.random_split(bg, [0.8, 0.2])


# NOTE: these must be shuffled!!!
tg_train = DataLoader(dataset=tg_train, batch_size=BATCH_SIZE, shuffle=True)
tg_test = DataLoader(dataset=tg_test, batch_size=BATCH_SIZE, shuffle=True)
bg_train = DataLoader(dataset=bg_train, batch_size=BATCH_SIZE, shuffle=True)
bg_test = DataLoader(dataset=bg_test, batch_size=BATCH_SIZE, shuffle=True)


qs_encoder = models.EncoderFC(
    latent_dim=LATENT_DIM_S,
    hidden_dims=HIDDEN_DIMS,
    in_W=96,
)

qz_encoder = models.EncoderFC(
    latent_dim=LATENT_DIM_Z,
    hidden_dims=HIDDEN_DIMS,
    in_W=96,
)

decoder = models.DecoderFC(
    latent_dim=LATENT_DIM_S + LATENT_DIM_Z,
    hidden_dims=HIDDEN_DIMS,
    in_W=96
)

discriminator = models.Discriminator(
    latent_dim_s=LATENT_DIM_S,
    latent_dim_z=LATENT_DIM_Z,
)

model = models.CVAE(
    s_encoder=qs_encoder,
    z_encoder=qz_encoder,
    decoder=decoder,
    discriminator=discriminator,
)
model = model.to(DEVICE)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print (pytorch_total_params)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

loss_fn = losses.CVAELoss(PoissonMultinomial())

def train_loop(model, tg_dataloader, bg_dataloader, loss_fn, optimizer):
    
    model.train()

    n_batches = len(tg_dataloader)
    total_loss = 0
    batch_size = None

    # get iterator over backgrounds
    bg_iterator = iter(bg_dataloader)

    # NOTE: this only works when batch sizes are identical in the
    # two iterators!
    for i, (tg_x, tg_y) in tqdm.tqdm(enumerate(tg_dataloader)):

        # for every target batch, grab a random batch from the background
        # iterator. 
        try:
            bg_x, bg_y = next(bg_iterator)
        # if we've reached the end of the background data, just start
        # at the beginning and grab another random batch.
        except StopIteration:
            # print ("Can't iterate over background.")
            # start over iterating over the backgrounds
            bg_iterator = iter(bg_dataloader)
            bg_x, bg_y = next(bg_iterator)

        if batch_size is None:
            batch_size = tg_x.shape[0]

        # if batch sizes mismatch, subset
        # NOTE: should do this randomly
        bg_bsz, tg_bsz = bg_x.shape[0], tg_x.shape[0]
        if bg_bsz > tg_bsz:
            idxs = np.random.choice(bg_bsz, size=tg_bsz)
            bg_x = bg_x[idxs, :, :, :]
        elif bg_bsz < tg_bsz:
            idxs = np.random.choice(tg_bsz, size=bg_bsz)
            tg_x = tg_x[idxs, :, :, :]

        tg_x, bg_x = tg_x.to(DEVICE), bg_x.to(DEVICE)

        optimizer.zero_grad()

        cvae_dict = model(tg_x, bg_x)

        loss = loss_fn(tg_x, bg_x, cvae_dict)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    return total_loss / (n_batches * batch_size)

def test_loop(model, tg_dataloader, bg_dataloader, loss_fn):
    
    model.eval()

    n_batches = len(tg_dataloader)
    total_loss = 0
    bg_iterator = iter(bg_dataloader)

    with torch.no_grad():
        batch_size = None
        # NOTE: this only works when batch sizes are identical in the
        # two iterators
        for i, (tg_x, tg_y) in tqdm.tqdm(enumerate(tg_dataloader)):
            
            try:
                bg_x, bg_y = next(bg_iterator)
            # if we've reached the end of the background data, just start
            # at the beginning and grab another random batch.
            except StopIteration:
                # print ("Can't iterate over background.")
                # start over iterating over the backgrounds
                bg_iterator = iter(bg_dataloader)
                bg_x, bg_y = next(bg_iterator)

            if batch_size is None:
                batch_size = tg_x.shape[0]

            # if batch sizes mismatch, subset
            # NOTE: should do this randomly
            bg_bsz, tg_bsz = bg_x.shape[0], tg_x.shape[0]
            if bg_bsz > tg_bsz:
                idxs = np.random.choice(bg_bsz, size=tg_bsz)
                bg_x = bg_x[idxs, :, :, :]
            elif bg_bsz < tg_bsz:
                idxs = np.random.choice(tg_bsz, size=bg_bsz)
                tg_x = tg_x[idxs, :, :, :]

            tg_x, bg_x = tg_x.to(DEVICE), bg_x.to(DEVICE)

            cvae_dict = model(tg_x, bg_x)
            loss = loss_fn(tg_x, bg_x, cvae_dict)
            total_loss += loss.item()
    
    return total_loss / (n_batches * batch_size)


def plot_example(model, tg_dataloader, bg_dataloader, plot_name: str):

    model.eval()
    bg_iterator = iter(bg_dataloader)
    with torch.no_grad():
        batch_size = None
        # NOTE: this only works when batch sizes are identical in the
        # two iterators
        for i, (tg_x, tg_y) in tqdm.tqdm(enumerate(tg_dataloader)):
            try:
                bg_x, bg_y = next(bg_iterator)
            # if we've reached the end of the background data, just start
            # at the beginning and grab another random batch.
            except StopIteration:
                # start over iterating over the backgrounds
                bg_iterator = iter(bg_dataloader)
                bg_x, bg_y = next(bg_iterator)
            # add batch dimension to single example
            # tg_x = torch.unsqueeze(tg_x[0].to(DEVICE), dim=0)
            # bg_x = torch.unsqueeze(bg_x[0].to(DEVICE), dim=0)
            cvae_dict = model(tg_x.to(DEVICE), bg_x.to(DEVICE))
            tg_x_hat, bg_x_hat, fg_x_hat = cvae_dict["tg_out"], cvae_dict["bg_out"], cvae_dict["fg_out"]
            break

    f, axarr = plt.subplots(2, 2, figsize=(8, 8))

    for ij, x, name in zip(
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        (tg_x, fg_x_hat, tg_x_hat, bg_x_hat),
        (
            "Original",
            "Reconstructed from salient",
            "Reconstructed from salient + irrelevant",
            "Reconstructed from irrelevant",
        ),
    ):
        i, j = ij
        x = x.cpu().numpy()[0]
        if (i > 0) or (j > 0):
            x = np.exp(x)
        # else:
        #     x /= np.sum(x)

        # undo log1p
        # x = np.exp(x) - 1
        # x /= np.sum(x)

        ind = np.arange(x.shape[0])
        axarr[i, j].bar(ind, x, 1, color=CMAP)
        axarr[i, j].set_title(name)
    f.tight_layout()
    f.savefig(plot_name, dpi=200)
    plt.close()


print("Start training VAE...")

res = []

for epoch in range(EPOCHS):

    plot_example(model, tg_test, bg_test, plot_name=f"fig/reconstructions/cvae/{epoch}.png")
   
    train_loss = train_loop(model, tg_train, bg_train, loss_fn, optimizer)
    test_loss = test_loop(model, tg_test, bg_test, loss_fn)
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
    scheduler.step(train_loss)

res_df = pd.DataFrame(res)

f, ax = plt.subplots()
sns.lineplot(data=res_df, x="epoch", y="loss_val", hue="loss_kind", ax=ax)
f.tight_layout()
sns.despine(ax=ax)
f.savefig("cvae.loss.png", dpi=200)

print("Finish!!")

model.eval()


s_reps, z_reps = [], []
labels = []
bg_iterator = iter(bg_test)
with torch.no_grad():
    for i, (tg_x, tg_y) in tqdm.tqdm(enumerate(tg_test)):
        try:
            bg_x, bg_y = next(bg_iterator)
        except StopIteration:
            bg_iterator = iter(bg_test)
            bg_x, bg_y = next(bg_iterator)

        bg_bsz, tg_bsz = bg_x.shape[0], tg_x.shape[0]
        if bg_bsz > tg_bsz:
            bg_x = bg_x[:tg_bsz, :, :, :]
            bg_y = bg_y[:tg_bsz]
        elif bg_bsz < tg_bsz:
            tg_x = tg_x[:bg_bsz, :, :, :]
            tg_y = tg_y[:bg_bsz]
        tg_x, bg_x = tg_x.to(DEVICE), bg_x.to(DEVICE)

        cvae_dict = model(tg_x, bg_x)
        # extract salient z
        salient = cvae_dict["tg_s"].cpu().numpy()
        irrelevant = cvae_dict["tg_z"].cpu().numpy()
        s_reps.append(salient)
        z_reps.append(irrelevant)
        labels.append(tg_y)


f, axarr = plt.subplots(2, figsize=(6, 8))#, sharex=True, sharey=True)

labels = np.concatenate(labels)
s_reps = np.concatenate(s_reps, axis=0)
z_reps = np.concatenate(z_reps, axis=0)

print (labels.shape, s_reps.shape, z_reps.shape)

clf = LogisticRegressionCV(cv=5)
titles = ["Sample embeddings in salient latent space", "Sample embeddings in irrelevant latent space"]
for i, (reps, title) in enumerate(zip((s_reps, z_reps), titles)):

    clf.fit(reps, labels)
    preds = clf.predict(reps)

    if LATENT_DIM_S > 2:
        pca = PCA(n_components=2)
        reps = pca.fit_transform(reps)
    axarr[i].scatter(reps[:, 0], reps[:, 1], c=labels, alpha=0.5)
    silhouette = str(round(silhouette_score(reps, labels), 3))
    title += f"\n(silhouette score = {silhouette})"
    axarr[i].set_title(title)

f.tight_layout()
f.savefig("cvae.coords.png", dpi=200)

def plot_reconstructed(model, r0=(-4, 4), r1=(-4, 4), n=4):

    f, axarr = plt.subplots(n, n, figsize=(8, 8))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            # sample a salient vector
            z = torch.Tensor([[x, y]]).to(DEVICE)
            # sample 0s for the irrelevant vector
            z_ = torch.Tensor([[0] * LATENT_DIM_Z]).to(DEVICE)
            x_hat = model.decoder(torch.cat((z, z_), dim=1))
            x_hat = x_hat.to('cpu').detach().numpy()[0]
            ind = np.arange(x_hat.shape[0])
            x_hat = np.exp(x_hat)

            axarr[i, j].bar(ind, x_hat, 1, color=CMAP)
            axarr[i, j].set_xticks([])
            axarr[i, j].set_yticks([])
    f.tight_layout()
    f.suptitle("Reconstructed mutation spectra from latent space")
    #plt.subplots_adjust(wspace=0, hspace=0)
    f.savefig("cvae.recons.png", dpi=100)

if LATENT_DIM_S == 2:
    plot_reconstructed(model)
