import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from make_training_data_cancer import poisson_resample
import tqdm
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

CMAP = np.repeat(["blue", "black", "red", "grey", "green", "pink"], 16)


def log2_transform(X: np.ndarray):
    # assume shape is N, K
    N, K = X.shape
    # get sums across kmers for each sample
    sample_sums = np.sum(X, axis=1)[:, None]

    # take log2 of sums
    l2_sums = np.log2(sample_sums)

    # multiply by log2 and divide by regular sum
    X_new = X * l2_sums
    X_new /= sample_sums

    return X_new


def cpca(
    tg: np.ndarray,
    bg: np.ndarray,
    alpha: float = 1,
    rank: int = 2,
):

    # construct feature covariance
    tg_cov = np.dot(tg.T, tg)
    bg_cov = np.dot(bg.T, bg)

    sigma = tg_cov - (alpha * bg_cov)
    n = sigma.shape[0]

    # get eigenvecs and eigen vals
    # we can use eigh since we're dealing with a symmetric covariance matrix
    vals, vectors = scipy.linalg.eigh(
        sigma,
        subset_by_index=(n - (rank), n - (rank - 1)),
    )

    return vals[::-1], vectors[::-1]


def main(args):

    rng = np.random.default_rng(42)

    bg = np.load(args.bg)
    tg = np.load(args.tg)

    X_bg, y_bg = bg["X"], bg["y"]
    X_tg, y_tg = tg["X"], tg["y"]

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

    (
        X_bg_train,
        X_bg_test,
        y_bg_train,
        y_bg_test,
    ) = train_test_split(X_bg, y_bg, test_size=0.2)

    (
        X_tg_train,
        X_tg_test,
        y_tg_train,
        y_tg_test,
    ) = train_test_split(X_tg, y_tg, test_size=0.2)


    # fit NMF

    errs = []
    for n in tqdm.tqdm(range(2, 25)):
        clf = NMF(n_components=n, max_iter=200)
        W = clf.fit_transform(log2_transform(X_tg))
        H = clf.components_
        errs.append((n, clf.reconstruction_err_))

    best_n = sorted(errs, key=lambda k: k[1])[0][0]


    clf = NMF(n_components=best_n)
    W = clf.fit_transform(X_tg)
    H = clf.components_
    H /= np.sum(H, axis=1)[:, None]

    silhouette = round(silhouette_score(W, y_tg), 3)
    print (silhouette)

    res_df = pd.DataFrame({"salient_silhouette": [silhouette]})

    res_df.to_csv(args.out, index=False)

    if args.plot:
        # cluster
        g = sns.clustermap(data=W, cmap="bone")
        g.savefig('o.png', dpi=200)
        ind = np.arange(96)
        f, axarr = plt.subplots(best_n, figsize=(6, best_n * 2), sharex=True)
        for i in range(best_n):
            axarr[i].bar(ind, H[i], 1, color=CMAP)
            axarr[i].set_ylabel("Fraction")
            if i == best_n - 1:
                axarr[i].set_xlabel("Mutation type")
            axarr[i].set_title(f"Inferred signature {i + 1}")
        f.tight_layout()
        f.savefig("signatures.png", dpi=200)



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bg")
    p.add_argument("--tg")
    p.add_argument("--out")
    p.add_argument("-resample", action="store_true")
    p.add_argument("-plot", action="store_true")
    args = p.parse_args()
    main(args)
