import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import torch
from datasets import MyDataset
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
import models
import losses
from losses import PoissonMultinomial
from make_training_data_cancer import poisson_resample
import argparse
import seaborn as sns


def train_loop(
    model,
    tg_dataloader,
    bg_dataloader,
    loss_fn,
    optimizer,
):

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

        if bg_bsz == 1 or tg_bsz == 1:
            continue

        if bg_bsz > tg_bsz:
            idxs = np.random.choice(bg_bsz, size=tg_bsz)
            bg_x = bg_x[idxs, :]
            bg_y = bg_y[idxs]
        elif bg_bsz < tg_bsz:
            idxs = np.random.choice(tg_bsz, size=bg_bsz)
            tg_x = tg_x[idxs, :]
            tg_y = tg_y[idxs]

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

            if bg_bsz == 1 or tg_bsz == 1:
                continue

            if bg_bsz > tg_bsz:
                idxs = np.random.choice(bg_bsz, size=tg_bsz)
                bg_x = bg_x[idxs, :]
                bg_y = bg_y[idxs]
            elif bg_bsz < tg_bsz:
                idxs = np.random.choice(tg_bsz, size=bg_bsz)
                tg_x = tg_x[idxs, :]
                tg_y = tg_y[idxs]

            tg_x, bg_x = tg_x.to(DEVICE), bg_x.to(DEVICE)

            cvae_dict = model(tg_x, bg_x)
            loss = loss_fn(tg_x, bg_x, cvae_dict)
            total_loss += loss.item()

    return total_loss / (n_batches * batch_size)


def plot_example(model, tg_dataloader, bg_dataloader, plot_name: str):

    model.eval()
    bg_iterator = iter(bg_dataloader)
    with torch.no_grad():
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
            tg_x_hat, bg_x_hat, fg_x_hat = (
                cvae_dict["tg_out"],
                cvae_dict["bg_out"],
                cvae_dict["fg_out"],
            )
            break

    f, axarr = plt.subplots(2, 2, figsize=(8, 8))

    # print (tg_x[0])
    # print (tg_x_hat[0])
    # print (np.exp(tg_x_hat.cpu().numpy()[0]))

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
            x = np.random.poisson(x)

        ind = np.arange(x.shape[0])
        axarr[i, j].bar(ind, x, 1, color=CMAP)
        axarr[i, j].set_title(name)
    f.tight_layout()
    f.savefig(plot_name, dpi=200)
    plt.close()


def plot_reconstructed(model, lib_size, r0=(-4, 4), r1=(-4, 4), n=4):

    f, axarr = plt.subplots(n, n, figsize=(8, 8))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            # sample a salient vector
            z = torch.Tensor([[x, y]]).to(DEVICE)
            # sample 0s for the irrelevant vector
            z_ = torch.Tensor([[0] * LATENT_DIM_Z]).to(DEVICE)

            # fake log libsize
            x_hat, _ = model.decoder(
                torch.cat((z, z_), dim=1),
                lib_size,
            )
            x_hat = x_hat.to("cpu").detach().numpy()[0]
            ind = np.arange(x_hat.shape[0])

            axarr[i, j].bar(ind, x_hat, 1, color=CMAP)
            axarr[i, j].set_xticks([])
            axarr[i, j].set_yticks([])

    f.tight_layout()
    f.savefig("cvae.recons.png", dpi=200)


LR = 1e-3

BATCH_SIZE = 128
LATENT_DIM_S = 2
LATENT_DIM_Z = 4
HIDDEN_DIMS = [128, 256]

CMAP = np.repeat(
    ["blue", "black", "red", "grey", "green", "pink"],
    16,
)

DEVICE = torch.device("mps")

rng = np.random.default_rng(42)


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

    # get empirical mean and variance of log library size
    bg_sums = np.sum(X_bg, axis=1)
    tg_sums = np.sum(X_tg, axis=1)
    log_size = torch.Tensor(
        [
            np.mean(
                np.concatenate(
                    [
                        bg_sums,
                        tg_sums,
                    ]
                )
            )
        ]
    ).to(DEVICE)

    # lib_mu, lib_var = (
    #     torch.Tensor([np.mean(log_size)]),
    #     torch.Tensor([np.var(log_size)]),
    # )

    X_bg, y_bg = torch.from_numpy(X_bg), torch.from_numpy(y_bg)
    X_tg, y_tg = torch.from_numpy(X_tg), torch.from_numpy(y_tg)

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
        in_W=96,
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
        lib_size=log_size,
    )
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    # loss_fn = losses.CVAELoss(PoissonMultinomial())
    loss_fn = losses.CVAELoss(losses.NB())
    # loss_fn = losses.CVAELoss(torch.nn.functional.mse_loss)

    print("Start training VAE...")

    res = []

    for epoch in range(args.epochs):

        if args.plot:
            plot_example(model, tg_test, bg_test, f"fig/reconstructions/cvae/{epoch}.png")

        train_loss = train_loop(model, tg_train, bg_train, loss_fn, optimizer)
        test_loss = test_loop(model, tg_test, bg_test, loss_fn)
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
        scheduler.step(train_loss)

    res_df = pd.DataFrame(res)

    f, ax = plt.subplots()
    sns.lineplot(x="epoch", y="loss_val", hue="loss_kind", ax=ax, data=res_df)
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
            if bg_bsz == 1 or tg_bsz == 1:
                continue
            if bg_bsz > tg_bsz:
                idxs = np.random.choice(bg_bsz, size=tg_bsz)
                bg_x = bg_x[idxs, :]
                bg_y = bg_y[idxs]
            elif bg_bsz < tg_bsz:
                idxs = np.random.choice(tg_bsz, size=bg_bsz)
                tg_x = tg_x[idxs, :]
                tg_y = tg_y[idxs]
            tg_x, bg_x = tg_x.to(DEVICE), bg_x.to(DEVICE)

            cvae_dict = model(tg_x, bg_x)
            # extract salient z
            salient = cvae_dict["tg_s"].cpu().numpy()
            irrelevant = cvae_dict["tg_z"].cpu().numpy()
            s_reps.append(salient)
            z_reps.append(irrelevant)
            labels.append(tg_y)

    labels = np.concatenate(labels)
    s_reps = np.concatenate(s_reps, axis=0)
    z_reps = np.concatenate(z_reps, axis=0)

    # clf = LogisticRegressionCV(cv=5)
    titles = [
        "Sample embeddings in salient latent space",
        "Sample embeddings in irrelevant latent space",
    ]

    f, axarr = plt.subplots(2, figsize=(6, 8))
    for i, (reps, title) in enumerate(zip((s_reps, z_reps), titles)):

        # clf.fit(reps, labels)
        # preds = clf.predict(reps)

        if LATENT_DIM_S > 2:
            pca = PCA(n_components=2)
            reps = pca.fit_transform(reps)
        axarr[i].scatter(reps[:, 0], reps[:, 1], c=labels, alpha=0.5)

        silhouette = "-1"
        if np.unique(labels).shape[0] > 1:
            silhouette = str(round(silhouette_score(reps, labels), 3))
        # accuracy = str(round(accuracy_score(labels, preds), 3))

        kind = "salient" if i == 0 else "irrelevant"
        res_df[f"{kind}_silhouette"] = silhouette
        # res_df[f"{kind}_accuracy"] = accuracy

        title += f"\n(silhouette score = {silhouette})"
        axarr[i].set_title(title)

    f.tight_layout()
    f.savefig("cvae.coords.png", dpi=200)

    res_df.to_csv(args.out, index=False)

    if args.plot:
        plot_reconstructed(model, log_size, n=4)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bg")
    p.add_argument("--tg")
    p.add_argument("--out")
    p.add_argument("-epochs", type=int, default=10)
    p.add_argument("-resample", action="store_true")
    p.add_argument("-plot", action="store_true")
    args = p.parse_args()
    main(args)
