from torchvision.datasets import MNIST, CIFAR10, CelebA
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import torch
import torchvision

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
import pandas as pd
from typing import List
import models
import losses


BATCH_SIZE = 100
LATENT_DIM = 2
INTERMEDIATE_DIM = 128
H, W = 64, 64
CHANNELS = 1
INPUT_DIM = H * W
LR = 1e-3
EPOCHS = 25
HIDDEN_DIMS = [32, 64, 128]#, 128, 256]
DEVICE = torch.device("cuda")


mnist_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(size=(H, W)),
        transforms.ToTensor(),
])



train_tg = torchvision.datasets.ImageFolder(
    "data/corrupted/train/",
    transform=mnist_transform,
)
# test_tg = torchvision.datasets.ImageFolder(
#     "data/corrupted/test/",
#     transform=mnist_transform,
# )
train_bg = torchvision.datasets.ImageFolder(
    "data/background_frog/",
    transform=mnist_transform,
)


# NOTE: these must be shuffled!!!
tg_train_dataloader = DataLoader(dataset=train_tg, batch_size=BATCH_SIZE, shuffle=True)
# tg_test_dataloader = DataLoader(dataset=test_tg, batch_size=BATCH_SIZE)
bg_train_dataloader = DataLoader(dataset=train_bg, batch_size=BATCH_SIZE, shuffle=True)

qs_encoder = models.Encoder(
    in_channels=CHANNELS,
    latent_dim=LATENT_DIM,
    kernel_size=3,
    stride=2,
    padding=1,
    hidden_dims=HIDDEN_DIMS,
    intermediate_dim=INTERMEDIATE_DIM,
    in_H=H,
)

qz_encoder = models.Encoder(
    in_channels=CHANNELS,
    latent_dim=LATENT_DIM,
    kernel_size=3,
    stride=2,
    padding=1,
    hidden_dims=HIDDEN_DIMS,
    intermediate_dim=INTERMEDIATE_DIM,
    in_H=H,
)

decoder = models.Decoder(
    out_channels=CHANNELS,
    latent_dim=LATENT_DIM * 2,
    kernel_size=3,
    stride=2,
    padding=1,
    output_padding=1,
    hidden_dims=HIDDEN_DIMS,
    intermediate_dim=INTERMEDIATE_DIM,
    in_H=H,
)

model = models.CVAE(s_encoder=qs_encoder, z_encoder=qz_encoder, decoder=decoder)
model = model.to(DEVICE)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print (pytorch_total_params)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

loss_fn = losses.CVAELoss()


def train_loop(model, tg_dataloader, bg_dataloader, loss_fn, optimizer):
    
    model.train()

    n_batches = len(tg_dataloader)
    total_loss = 0
    batch_size = None

    # NOTE: this assumes that we have more (or the same number)
    # of targets than we have backgrounds

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
            # start over iterating over the backgrounds
            bg_iterator = iter(bg_dataloader)
            bg_x, bg_y = next(bg_iterator)
        
        if batch_size is None:
            batch_size = tg_x.shape[0]

        # if batch sizes mismatch, subset
        # NOTE: should do this randomly
        bg_bsz, tg_bsz = bg_x.shape[0], tg_x.shape[0]
        if bg_bsz > tg_bsz:
            bg_x = bg_x[:tg_bsz, :, :, :]

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
    tg_iterator = iter(tg_dataloader)

    with torch.no_grad():
        batch_size = None
        # NOTE: this only works when batch sizes are identical in the
        # two iterators
        for i, (bg_x, bg_y) in tqdm.tqdm(enumerate(bg_dataloader)):
            
            tg_x, tg_y = next(tg_iterator)
            
            if batch_size is None:
                batch_size = tg_x.shape[0]

            tg_x, bg_x = tg_x.to(DEVICE), bg_x.to(DEVICE)

            cvae_dict = model(tg_x, bg_x)
            loss = loss_fn(tg_x, bg_x, cvae_dict)
            total_loss += loss.item()
    
    return total_loss / (n_batches * batch_size)


# def plot_example(model, tg_dataloader, bg_dataloader, plot_name: str):

#     model.eval()
#     tg_iterator = iter(tg_dataloader)
#     with torch.no_grad():
#         batch_size = None
#         # NOTE: this only works when batch sizes are identical in the
#         # two iterators
#         for i, (bg_x, bg_y) in tqdm.tqdm(enumerate(bg_dataloader)):
#             tg_x, tg_y = next(tg_iterator)
#             # add batch dimension to single example
#             tg_x = torch.unsqueeze(tg_x[0].to(DEVICE), dim=0)
#             bg_x = torch.unsqueeze(bg_x[0].to(DEVICE), dim=0)
#             cvae_dict = model(tg_x, bg_x)
#             tg_out = cvae_dict["tg_out"]
        
#     x = x.view(1, CHANNELS, H, W).cpu().numpy()
#     x_hat = x_hat.view(1, CHANNELS, H, W).cpu().numpy()

#     x = np.transpose(x, (0, 2, 3, 1))
#     x_hat = np.transpose(x_hat, (0, 2, 3, 1))

#     f, (ax1, ax2) = plt.subplots(1, 2)
#     ax1.imshow(x[0])
#     ax2.imshow(x_hat[0])
#     ax1.set_title("Original image")
#     ax2.set_title("Reconstructed image")
#     f.tight_layout()
#     f.savefig(plot_name, dpi=200)
#     plt.close()


print("Start training VAE...")

res = []

for epoch in range(EPOCHS):
    # plot_example(model, tg_valid_dataloader, plot_name=f"fig/reconstructions/{epoch}.png")
    # print (model.qs.fc_intermediate.weight)
     #print (model.qz.fc_intermediate.weight)
    train_loss = train_loop(model, tg_train_dataloader, bg_train_dataloader, loss_fn, optimizer)
    # test_loss = test_loop(model, tg_valid_dataloader, bg_valid_dataloader, loss_fn)
    test_loss = 0
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
f.savefig("loss.png", dpi=200)

print("Finish!!")

model.eval()

f, ax = plt.subplots()

reps = []
labels = []
bg_iterator = iter(bg_train_dataloader)
with torch.no_grad():
    for i, (tg_x, tg_y) in tqdm.tqdm(enumerate(tg_train_dataloader)):

        if i > 20: break
        try:
            bg_x, bg_y = next(bg_iterator)
        except StopIteration:
            bg_iterator = iter(bg_train_dataloader)
            bg_x, bg_y = next(bg_iterator)

        tg_x, bg_x = tg_x.to(DEVICE), bg_x.to(DEVICE)
        cvae_dict = model(tg_x, bg_x)
        # extract salient z
        salient_z = cvae_dict["tg_s"].cpu().numpy()
        reps.append(salient_z)
        labels.extend(list(tg_y.cpu().numpy()))

reps = np.concatenate(reps)
labels = np.array(labels)

clf = PCA(n_components=2)
X_new = clf.fit_transform(reps)
ax.scatter(reps[:, 0], reps[:, 1], c=labels)

f.savefig("coords.png", dpi=200)

def plot_reconstructed(model, r0=(-4, 4), r1=(-4, 4), n=8):

    f, axarr = plt.subplots(n, n)
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            # sample a salient vector
            z = torch.Tensor([[x, y]]).to(DEVICE)
            # sample 0s for the irrelevant vector
            x_hat = model.decoder(torch.cat((z, torch.zeros_like(z)), dim=-1))
            x_hat = x_hat.to('cpu').detach().numpy()[0, :, :]
            axarr[i, j].imshow(np.transpose(x_hat, (1, 2, 0)))
            axarr[i, j].set_xticks([])
            axarr[i, j].set_yticks([])
    f.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    f.savefig("recons.png")


plot_reconstructed(model)
