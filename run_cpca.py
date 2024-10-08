import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import scipy
import matplotlib.pyplot as plt
import seaborn as sns

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

bg = np.load("data/background_spectra.npz")
tg = np.load("data/target_spectra.npz")

X_bg, y_bg = bg["X"], bg["y"]
X_tg, y_tg = tg["X"], tg["y"]

X_bg_train, X_bg_test, y_bg_train, y_bg_test = train_test_split(X_bg, y_bg, test_size=0.2)
X_tg_train, X_tg_test, y_tg_train, y_tg_test = train_test_split(X_tg, y_tg, test_size=0.2)

def cpca(tg: np.ndarray, bg: np.ndarray, alpha: float=1, rank: int = 2):

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

clf = NMF(n_components=5)
W = clf.fit_transform(log2_transform(X_tg))
H = clf.components_
print (W.shape, H.shape)

H /= np.sum(H, axis=1)[:, None]

f, axarr = plt.subplots(5)
ind = np.arange(96)
for i in range(H.shape[0]):
    axarr[i].bar(ind, H[i], 1)
f.savefig("nmf.png")

g = sns.clustermap(data=W)
g.fig.suptitle(round(silhouette_score(W, y_tg), 3))
g.savefig("nmf.cluster.png")


X_tg_train = X_tg_train / np.sum(X_tg_train, axis=1)[:, None]
X_tg_test = X_tg_test / np.sum(X_tg_test, axis=1)[:, None]

clf = StandardScaler()
X_tg_train = clf.fit_transform(X_tg_train)
X_tg_test = clf.fit_transform(X_tg_test)


clf = PCA(n_components=2)
clf.fit(X_tg_train)
X_new = clf.transform(X_tg_test)
f, ax = plt.subplots()
ax.scatter(X_new[:, 0], X_new[:, 1], c=y_tg_test)
f.savefig("pca.png")

_, cpcs = cpca(X_tg, X_bg, alpha=1, rank=2)
# project onto cpcs
X_new = np.dot(X_tg, cpcs)
print (X_new.shape)
f, ax = plt.subplots()
ax.scatter(X_new[:, 0], X_new[:, 1], c=y_tg)
f.savefig("cpca.png")

