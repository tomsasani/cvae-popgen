from torchvision.datasets import MNIST, CIFAR10, CelebA, FashionMNIST
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import torch
import torchvision


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
import pandas as pd
import itertools

import models
import losses

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


BATCH_SIZE = 128
LATENT_DIM = 5
INTERMEDIATE_DIM = 128

KERNEL_SIZE = (3, 3)
PADDING = (1, 1)
OUTPUT_PADDING = (1, 1)
STRIDE = (2, 2)

H, W = 64, 64
CHANNELS = 3
INPUT_DIM = H * W
LR = 1e-3
EPOCHS = 10
HIDDEN_DIMS = [32, 64, 128, 256]
# HIDDEN_DIMS = [h * 2 for h in HIDDEN_DIMS]
DEVICE = torch.device("cuda")


transform = transforms.Compose([
        # transforms.Grayscale(),
        transforms.Resize(size=(H, W)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),])


celeb_train = CelebA(
    root="data",
    split="train",
    target_type="attr",
    transform=transform,
    download=True,
)

celeb_val = CelebA(
    root="data",
    split="valid",
    target_type="attr",
    transform=transform,
    download=True,
)

glasses_attr_idx = celeb_train.attr_names.index("Eyeglasses")
hat_attr_idx = celeb_train.attr_names.index("Wearing_Hat")

tg_train, bg_train = extract_celeb_data(celeb_train, glasses_attr_idx, hat_attr_idx)
tg_test, bg_test = extract_celeb_data(celeb_val, glasses_attr_idx, hat_attr_idx)

# tg = torchvision.datasets.ImageFolder(
#     "data/simulated/target/",
#     transform=transform,
# )

# tg_train, tg_test = torch.utils.data.random_split(tg, [0.9, 0.1])


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


encoder = models.Encoder(
    in_channels=CHANNELS,
    latent_dim=LATENT_DIM,
    kernel_size=KERNEL_SIZE,
    stride=STRIDE,
    padding=PADDING,
    hidden_dims=HIDDEN_DIMS,
    intermediate_dim=INTERMEDIATE_DIM,
    in_HW=(H, W),
)

decoder = models.Decoder(
    out_channels=CHANNELS,
    latent_dim=LATENT_DIM,
    kernel_size=KERNEL_SIZE,
    stride=STRIDE,
    padding=PADDING,
    output_padding=OUTPUT_PADDING,
    hidden_dims=HIDDEN_DIMS,
    intermediate_dim=INTERMEDIATE_DIM,
    in_HW=(H, W),
)

model = models.VAE(encoder=encoder, decoder=decoder,)
model = model.to(DEVICE)
print (model)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print (pytorch_total_params)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

loss_fn = losses.VAELoss(kld_weight=1.)


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
            loss = loss_fn(x, x_hat, mean, log_var)
            total_loss += loss.item()
    
    return total_loss / (n_batches * batch_size)


def plot_example(model, dataloader, plot_name: str):

    f, axarr = plt.subplots(5, 2, figsize=(8, 4))

    dataloader_iter = iter(dataloader)

    model.eval()
    with torch.no_grad():

        for i in range(5):
            # grab first example
            xs, ys = next(dataloader_iter)
            # add batch dimension to single example
            x = torch.unsqueeze(xs[0].to(DEVICE), dim=0)

            x_hat, mu, log_var, z = model(x)
        
            x = x.view(1, CHANNELS, H, W).cpu().numpy()
            x_hat = x_hat.view(1, CHANNELS, H, W).cpu().numpy()

            x = np.transpose(x, (0, 2, 3, 1))
            x_hat = np.transpose(x_hat, (0, 2, 3, 1))

            axarr[i, 0].imshow(x[0])
            axarr[i, 1].imshow(x_hat[0])

            for j in (0, 1):
                axarr[i, j].set_xticks([])
                axarr[i, j].set_yticks([])

    # ax1.set_title("Original image")
    # ax2.set_title("Reconstructed image")
    f.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    f.savefig(plot_name, dpi=200)
    plt.close()


print("Start training VAE...")

res = []

for epoch in range(EPOCHS):
    plot_example(model, test_loader, plot_name=f"fig/reconstructions/vae/{epoch}.png")

    train_loss = train_loop(model, train_loader, loss_fn, optimizer)
    test_loss = test_loop(model, test_loader, loss_fn)

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
    for batch_idx, (x, y) in enumerate(tqdm.tqdm(test_loader)):
        x = x.to(DEVICE)

        x_hat, mu, log_var, z = model(x)
        z = z.cpu().numpy()
        reps.append(z)
        labels.append(y)
reps = np.concatenate(reps)
labels = np.concatenate(labels)[:, glasses_attr_idx]

clf = LogisticRegressionCV(cv=5)
clf.fit(reps, labels)
preds = clf.predict(reps)

f, ax = plt.subplots()
if LATENT_DIM > 2:
    clf = PCA(n_components=2)
    reps = clf.fit_transform(reps)
ax.scatter(reps[:, 0], reps[:, 1], c=labels, alpha=0.5)
ax.set_title(round(accuracy_score(labels, preds), 3))
f.savefig("vae.coords.png", dpi=200)



def plot_reconstructed(model, r0=(-4, 4), r1=(-4, 4), n=12):

    f, axarr = plt.subplots(n, n, figsize=(8, 8))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            # sample a salient vector
            z = torch.Tensor([[x, y]]).to(DEVICE)
            # sample 0s for the irrelevant vector
            x_hat = model.decoder(z)
            x_hat = x_hat.to('cpu').detach().numpy()[0, :, :]
            axarr[i, j].imshow(np.transpose(x_hat, (1, 2, 0)))
            axarr[i, j].set_xticks([])
            axarr[i, j].set_yticks([])
    f.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    f.savefig("vae.recons.png")

if LATENT_DIM == 2:
    plot_reconstructed(model)
