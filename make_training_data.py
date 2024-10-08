import pathlib
import numpy as np
from PIL import Image
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from typing import List, Union

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


def allocate_signatures(
    rng: np.random.default_rng,
    bg_signatures: List[str],
    fg_signatures: List[str] = [],
    fg_fraction: float = 0.1,
    bg_fractions: List[float] = None,
):
    # figure out the total number of signatures we're adding
    signatures = bg_signatures + fg_signatures

    # randomly decide on the contributions of each background signature.
    # the first proportion is randomly decided, and we bound the 
    # total fraction of background signatures to be less than some threshold
    # so that foreground signatures can take up the rest
    bg_proportion = (1 - fg_fraction) if len(fg_signatures) > 0 else 1
    proportions = [rng.uniform(0, bg_proportion)]
    # first, allocate background signatures
    while len(proportions) < len(bg_signatures) - 1:
        prop = rng.uniform(0, bg_proportion - sum(proportions))
        proportions.append(prop)

    # if there are no foreground signatures, we allocate
    # the remaining signature proportion to the final background
    # signatures
    if len(fg_signatures) == 0:
        proportions.append(1 - sum(proportions))
    # otherwise, we add a random amount of the final background
    # signature, and then add foreground proportions
    else:
        # fill in necessary amount of final background signature
        proportions.append(bg_proportion - sum(proportions))
        while len(proportions) < len(bg_signatures) + len(fg_signatures) - 1:
            proportions.append(rng.uniform(0, 1 - sum(proportions)))
        proportions.append(1 - sum(proportions))


    return dict(zip(signatures, proportions))


cosmic = pd.read_csv("data/COSMIC_v3.4_SBS_GRCh38.txt", sep="\t")
cosmic["base"] = cosmic["Type"].apply(lambda t: t.split("[")[-1].split("]")[0])
cosmic = cosmic.sort_values("base")

rng = np.random.default_rng(42)

sig_names = cosmic.columns[1:]

N_BACKGROUND = 10_000
BACKGROUND_SIGNATURES = ["SBS1", "SBS5", "SBS30"]
# BACKGROUND_SIGNATURES = ["SBS5", "SBS8", "SBS9"]

N_FOREGROUND = N_BACKGROUND // 2
FOREGROUND_SIGNATURES = ["SBS18", "SBS12"]

N_MUTATIONS_BG = 100
N_MUTATIONS_FG = 100

background_spectra = np.zeros((N_BACKGROUND, 96))

ind = np.arange(96)
cmap = np.repeat(["blue", "black", "red", "grey", "green", "pink"], 16)

f, axarr = plt.subplots(3, figsize=(9, 6))

# create background data that is a combination of SBS1 and SBS5
for i in tqdm.tqdm(range(N_BACKGROUND)):
    
    # randomly allocate proportions to each signatures
    contributions = allocate_signatures(rng, BACKGROUND_SIGNATURES, [])
    # print (contributions, sum(contributions.values()))
    for signature, proportion in contributions.items():
        # total mutations to be drawn from this signature
        total_mutations = int(N_MUTATIONS_BG * proportion)
        # figure out which signature to sample from
        mutation_probabilities = cosmic[signature].values
        # draw a single mutation from a poisson distribution with probabilities
        # equal to signature contributions of each mutation type
        mutations_drawn = np.random.poisson(mutation_probabilities * total_mutations)
        background_spectra[i] += mutations_drawn

# background_spectra = log2_transform(background_spectra)
# background_spectra = np.log1p(background_spectra)
print (background_spectra.shape)
# background_spectra = background_spectra / np.sum(background_spectra, axis=1)[:, None]
# mmin, mmax = np.min(background_spectra, axis=1)[:, None], np.max(background_spectra, axis=1)[:, None]
# background_spectra = (background_spectra - mmin) / (mmax - mmin)

axarr[0].bar(ind, np.mean(background_spectra, axis=0), 1, yerr=np.std(background_spectra, axis=0), color=cmap, ec="w", lw=1)


foreground_spectra = []
ys = []
for fg_i, fg_signatures in enumerate(FOREGROUND_SIGNATURES):
    fg_spectra = np.zeros((N_FOREGROUND, 96))
    for i in tqdm.tqdm(range(N_FOREGROUND)):
        # randomly allocate proportions to each signature
        contributions = allocate_signatures(
            rng,
            BACKGROUND_SIGNATURES,
            [fg_signatures],
            fg_fraction=0.1,
        )
        # print (contributions, sum(contributions.values()))
        for signature, proportion in contributions.items():
            # total mutations to be drawn from this signature
            total_mutations = int(N_MUTATIONS_FG * proportion)
            # figure out which signature to sample from
            mutation_probabilities = cosmic[signature].values
            # draw a single mutation from a poisson distribution with probabilities
            # equal to signature contributions of each mutation type
            mutations_drawn = np.random.poisson(mutation_probabilities * total_mutations)
            fg_spectra[i] += mutations_drawn
        ys.append(fg_i)

    # fg_spectra = log2_transform(fg_spectra)
    # fg_spectra = np.log1p(fg_spectra)
    # fg_spectra = fg_spectra / np.sum(fg_spectra, axis=1)[:, None]
    mmin, mmax = np.min(fg_spectra, axis=1)[:, None], np.max(fg_spectra, axis=1)[:, None]

    # fg_spectra = (fg_spectra - mmin) / (mmax - mmin)
    axarr[fg_i + 1].bar(ind, np.mean(fg_spectra, axis=0), 1, yerr=np.std(fg_spectra, axis=0), color=cmap, ec="w", lw=1)
    foreground_spectra.append(fg_spectra)

foreground_spectra = np.concatenate(foreground_spectra)
for i in range(3):
    axarr[i].set_title("Background" if i == 0 else "Target w/ SBS18" if i == 1 else "Target w/ SBS12")
    axarr[i].set_ylabel("# of mutations\n(mean +/- SD)")
    axarr[i].set_xticks(ind[::16] + 8)
    axarr[i].set_xticklabels(cosmic["base"].values[::16])#
    sns.despine(ax=axarr[i])
f.tight_layout()
f.savefig("spectra.png", dpi=200)

# background_spectra = np.expand_dims(background_spectra, axis=(1, 2))
# foreground_spectra = np.expand_dims(foreground_spectra, axis=(1, 2))

print (background_spectra.shape, foreground_spectra.shape)
# background_spectra = np.reshape(background_spectra, (N_BACKGROUND, 6, 1, 16))
# foreground_spectra = np.reshape(foreground_spectra, (N_FOREGROUND * 2, 6, 1, 16))

np.savez("data/background_spectra.npz", X=background_spectra, y=np.zeros(N_BACKGROUND))
np.savez("data/target_spectra.npz", X=foreground_spectra, y=np.array(ys))

background_spectra = np.reshape(background_spectra, (N_BACKGROUND, 6, 1, 16))
f, ax = plt.subplots()
sns.heatmap(background_spectra[0, :, 0, :], ax=ax)
f.savefig("heatmap.png")
