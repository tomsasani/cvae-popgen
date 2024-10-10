import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def poisson_resample(
    rng: np.random.default_rng,
    X: np.ndarray,
    n: int,
):
    # assume X is shape N, K
    X_p = X / np.sum(X, axis=1)[:, None]
    X_p *= n
    X_downsampled = rng.poisson(X_p, size=X_p.shape)

    return X_downsampled.astype(np.float32)


rng = np.random.default_rng(42)

subtypes = pd.read_excel(
    "data/SupplementaryTables.xlsx",
    sheet_name="Table S6",
)
spectra = pd.read_excel(
    "data/SupplementaryTables.xlsx",
    sheet_name="Table S7",
)

# spectra = spectra[spectra["sample"].str.contains("GEL")]

kmers = spectra.columns[1:]

spectra = spectra.merge(subtypes)

# number of examples per organ
spectra_counts = (
    spectra.groupby("organ").size().reset_index().rename(columns={0: "n_samples"})
)
f, ax = plt.subplots()
sns.barplot(data=spectra_counts, x="organ", y="n_samples", ax=ax)
plt.xticks(rotation=90)
f.tight_layout()
f.savefig("sample_counts.png", dpi=200)

tissues = spectra["organ"].unique()

tg_tissues = [
    "Colorectal",
    "Breast",
    "Lung",
    "Kidney",
]
o2y = dict(zip(tg_tissues, range(len(tg_tissues))))
spectra["label"] = spectra["organ"].apply(lambda o: o2y[o] if o in o2y else -1)

# figure out percentiles of the total count distribution
totals = np.sum(spectra[kmers], axis=1)
spectra["total"] = totals

pctile_lo, pctile_hi = np.percentile(totals, 5), np.percentile(totals, 95)

spectra = spectra[spectra["total"].between(pctile_lo, pctile_hi)]
min_count = spectra["total"].min()

CMAP = np.repeat(["blue", "black", "red", "grey", "green", "pink"], 16)

# calculate cossims between tissues
tissue_spectra = np.zeros((len(tissues), 96))
for ti, (tissue, tissue_df) in enumerate(spectra.groupby("organ")):
    mean_spectrum = np.mean(tissue_df[kmers].values, axis=0)
    tissue_spectra[ti] = mean_spectrum

f, ax = plt.subplots()
cossims = cosine_similarity(tissue_spectra)
sns.heatmap(cossims)
ax.set_xticks(np.arange(len(tissues)) + 0.5)
ax.set_xticklabels(tissues, rotation=90)
ax.set_yticks(np.arange(len(tissues)) + 0.5)
ax.set_yticklabels(tissues, rotation=0)
f.tight_layout()
f.savefig("cossims.png", dpi=200)

res = []
for i in range(len(tissues)):
    for j in range(len(tissues)):
        if i == j: 
            continue
        res.append((tissues[i], tissues[j], cossims[i, j]))

# print (sorted(res, key=lambda c: c[2]))


f, axarr = plt.subplots(len(tg_tissues) + 1, figsize=(6, 3 * len(tg_tissues)))
ind = np.arange(96)
for ti, tissue in enumerate(tg_tissues):
    s = spectra[spectra["organ"] == tissue][kmers].values
    s = s / np.sum(s, axis=1)[:, None]
    mean, std = np.mean(s, axis=0), np.std(s, axis=0)
    axarr[ti].bar(ind, mean, 1, yerr=std, color=CMAP)
    axarr[ti].set_title(tissue)
    sns.despine(ax=axarr[ti])

other_s = spectra[~spectra["organ"].isin(tg_tissues)][kmers].values
other_s = other_s / np.sum(other_s, axis=1)[:, None]
mean, std = np.mean(other_s, axis=0), np.std(other_s, axis=0)
axarr[-1].bar(ind, mean, 1, yerr=std, color=CMAP)
axarr[-1].set_title("All other tissues")
sns.despine(ax=axarr[-1])
f.tight_layout()
f.savefig("a.png", dpi=200)


tg = spectra[spectra["organ"].isin(tg_tissues)]
print (tg.shape)
X_tg = tg[kmers].values
y_tg = tg["label"].values

bg = spectra[~spectra["organ"].isin(tg_tissues)]
print (bg.shape)

X_bg = bg[kmers].values
y_bg = bg["label"].values

f, ax = plt.subplots()
ax.hist(np.sum(X_tg, axis=1), histtype="step")
ax.hist(np.sum(X_bg, axis=1), histtype="step")
f.savefig("counts.png")

np.savez("data/cancer_bg_spectra.npz", X=X_bg, y=y_bg)
np.savez("data/cancer_tg_spectra.npz", X=X_tg, y=y_tg)
