from torchvision.datasets import MNIST, CIFAR10, CelebA
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import torch
import torchvision

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


KERNEL_SIZE = (3, 3)
PADDING = (1, 1)
OUTPUT_PADDING = (1, 1)
STRIDE = (2, 2)

H, W = 64, 64
CHANNELS = 3
LR = 1e-3
EPOCHS = 10
HIDDEN_DIMS = [32, 64, 128, 256]

BATCH_SIZE = 128
LATENT_DIM_S = 5
LATENT_DIM_Z = 15
INTERMEDIATE_DIM = 128

DEVICE = torch.device("cuda")


def extract_celeb_data(dataset, ai, bi):

    targets = dataset.attr

    target_idxs = np.where(targets[:, ai] + targets[:, bi] == 1)[0]
    background_idxs = np.where(targets[:, ai] + targets[:, bi] == 0)[0]

    # target_idxs = np.random.choice(target_idxs, size=1_000, replace=False)
    # background_idxs = np.random.choice(background_idxs, size=1_000, replace=False)

    target = torch.utils.data.Subset(dataset, target_idxs)
    background = torch.utils.data.Subset(dataset, background_idxs)

    print (target_idxs.shape, background_idxs.shape)

    return target, background


celeb_transform = transforms.Compose(
    [
        # transforms.Grayscale(),
        transforms.Resize(size=(H, W)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

celeb_train = CelebA(
    root="data",
    split="train",
    target_type="attr",
    transform=celeb_transform,
    download=True,
)

celeb_val = CelebA(
    root="data",
    split="valid",
    target_type="attr",
    transform=celeb_transform,
    download=True,
)

glasses_attr_idx = celeb_train.attr_names.index("Eyeglasses")
hat_attr_idx = celeb_train.attr_names.index("Wearing_Hat")

tg_train, bg_train = extract_celeb_data(celeb_train, glasses_attr_idx, hat_attr_idx)
tg_test, bg_test = extract_celeb_data(celeb_val, glasses_attr_idx, hat_attr_idx)


# tg = torchvision.datasets.ImageFolder(
#     "data/simulated/target/",
#     transform=celeb_transform,
# )

# bg = torchvision.datasets.ImageFolder(
#     "data/simulated/background/",
#     transform=celeb_transform,
# )

# tg_train, tg_test = torch.utils.data.random_split(tg, [0.9, 0.1])
# bg_train, bg_test = torch.utils.data.random_split(bg, [0.9, 0.1])


# target = torchvision.datasets.ImageFolder(
#     "data/corrupted/target/",
#     transform=celeb_transform,
# )
# background = torchvision.datasets.ImageFolder(
#     "data/corrupted/background/",
#     transform=celeb_transform,
# )

# tg_train, tg_test = torch.utils.data.random_split(target, [0.8, 0.2])
# bg_train, bg_test = torch.utils.data.random_split(background, [0.8, 0.2])


# NOTE: these must be shuffled!!!
tg_train = DataLoader(dataset=tg_train, batch_size=BATCH_SIZE, shuffle=True)
tg_test = DataLoader(dataset=tg_test, batch_size=BATCH_SIZE, shuffle=True)
bg_train = DataLoader(dataset=bg_train, batch_size=BATCH_SIZE, shuffle=True)
bg_test = DataLoader(dataset=bg_test, batch_size=BATCH_SIZE, shuffle=True)

qs_encoder = models.Encoder(
    in_channels=CHANNELS,
    latent_dim=LATENT_DIM_S,
    kernel_size=KERNEL_SIZE,
    stride=STRIDE,
    padding=PADDING,
    hidden_dims=HIDDEN_DIMS,
    intermediate_dim=INTERMEDIATE_DIM,
    in_HW=(H, W),
)

qz_encoder = models.Encoder(
    in_channels=CHANNELS,
    latent_dim=LATENT_DIM_Z,
    kernel_size=KERNEL_SIZE,
    stride=STRIDE,
    padding=PADDING,
    hidden_dims=HIDDEN_DIMS,
    intermediate_dim=INTERMEDIATE_DIM,
    in_HW=(H, W),
)

decoder = models.Decoder(
    out_channels=CHANNELS,
    latent_dim=LATENT_DIM_S + LATENT_DIM_Z,
    kernel_size=KERNEL_SIZE,
    stride=STRIDE,
    padding=PADDING,
    output_padding=OUTPUT_PADDING,
    hidden_dims=HIDDEN_DIMS,
    intermediate_dim=INTERMEDIATE_DIM,
    in_HW=(H, W),
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

optimizer = torch.optim.Adam(model.parameters(), lr=LR, )

loss_fn = losses.CVAELoss()

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
            # tg_x_hat = tg_x_hat[0], bg_x_hat[0], fg_x_hat[0]
            # tg_x = tg_x[0]
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
        x = x.view(BATCH_SIZE, CHANNELS, H, W).cpu().numpy()
        x = np.transpose(x, (0, 2, 3, 1))
        axarr[i, j].imshow(x[0])
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

res_df = pd.DataFrame(res)

f, ax = plt.subplots()
sns.lineplot(data=res_df, x="epoch", y="loss_val", hue="loss_kind", ax=ax)
f.tight_layout()
sns.despine(ax=ax)
f.savefig("cvae.loss.png", dpi=200)

print("Finish!!")

model.eval()


reps = []
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
        salient_z = cvae_dict["tg_s"].cpu().numpy()
        reps.append(salient_z)
        labels.append(tg_y)

reps = np.concatenate(reps, axis=0)
labels = np.concatenate(labels)[:, glasses_attr_idx]
# labels[labels > 1] = 1

clf = LogisticRegressionCV(cv=5)
clf.fit(reps, labels)
preds = clf.predict(reps)

f, ax = plt.subplots()
if LATENT_DIM_S > 2:
    clf = PCA(n_components=2)
    reps = clf.fit_transform(reps)
ax.scatter(reps[:, 0], reps[:, 1], c=labels, alpha=0.5)
ax.set_title(round(accuracy_score(labels, preds), 3))
f.savefig("cvae.coords.png", dpi=200)

def plot_reconstructed(model, r0=(-4, 4), r1=(-4, 4), n=8):

    f, axarr = plt.subplots(n, n, figsize=(8, 8))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            # sample a salient vector
            z = torch.Tensor([[x, y]]).to(DEVICE)
            # sample 0s for the irrelevant vector
            z_ = torch.Tensor([[0] * LATENT_DIM_Z]).to(DEVICE)
            x_hat = model.decoder(torch.cat((z, z_), dim=1))
            x_hat = x_hat.to('cpu').detach().numpy()[0, :, :]
            axarr[i, j].imshow(np.transpose(x_hat, (1, 2, 0)))
            axarr[i, j].set_xticks([])
            axarr[i, j].set_yticks([])
    f.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    f.savefig("cvae.recons.png")

if LATENT_DIM_S == 2:
    plot_reconstructed(model)
