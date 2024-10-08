"""
Utility functions and classes (including default parameters).
Author: Sara Mathieson, Rebecca Riley
Date: 9/27/22
"""

# python imports
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torchvision

# our imports
import global_vars

def calculate_channel_means(batch):
    bsz, c, h, w = batch.shape
    # get means across the batch, per channel
    means = torch.mean(batch, dim=(0, 2, 3))
    stds = torch.std(batch, dim=(0, 2, 3))
    # make sure we don't divide by 0 when adjusting for stdev
    stds[-1] = 1
    # then, get means across the batch per channel
    return means, stds


def sort_min_diff_numpy(X):
    '''this function takes in a SNP matrix with indv on rows and returns the same matrix with indvs sorted by genetic similarity.
    this problem is NP, so here we use a nearest neighbors approx.  it's not perfect, but it's fast and generally performs ok.
    assumes your input matrix is a numpy array'''

    C, H, W = X.shape

    X_sorted = np.zeros((C, H, W))
    X_genotypes = X[0, :, :]

    mb = NearestNeighbors(
        n_neighbors=len(X_genotypes),
        metric="manhattan",
    ).fit(X_genotypes)

    v = mb.kneighbors(X_genotypes)
    smallest = np.argmin(np.sum(v[0], axis=1))

    X_genotypes_sorted = X_genotypes[v[1][smallest]]
    X_sorted[0, :, :] = X_genotypes_sorted
    X_sorted[1, :, :] = X[1, :, :]

    return X_sorted


def inter_snp_distances(positions: np.ndarray, norm_len: int) -> np.ndarray:
    if positions.shape[0] > 0:
        dist_vec = [0]
        for i in range(positions.shape[0] - 1):
            # NOTE: inter-snp distances always normalized to simulated region size
            dist_vec.append((positions[i + 1] - positions[i]) / norm_len)
    else:
        dist_vec = []
    dist_vec = np.array(dist_vec)

    return np.array(dist_vec)


def find_segregating_idxs(X: np.ndarray, filter_singletons: bool = False):
    n_snps, n_haps = X.shape
    # initialize mask to store "good" sites
    to_keep = np.ones(n_snps)

    # remove sites that are non-segregating (i.e., if we didn't
    # add any information to them because they were multi-allelic
    # or because they were a silent mutation)
    acs = np.sum(X, axis=1)

    assert acs.shape[0] == n_snps

    non_segregating = np.where((acs == 0) | (acs == n_haps))[0]
    to_keep[non_segregating] = 0

    if filter_singletons:
        singletons = np.where((acs == 1) | (acs == n_haps - 1))[0]
        to_keep[singletons] = 0

    return np.where(to_keep)[0]


def check_for_missing_data(
    batch: np.ndarray,
):
    N, C, H, W = batch.shape
    # count the number of haplotypes with 0 genotypes at each
    # site for all batches
    haps_with_zero = np.count_nonzero(batch[:, 0, :, :] == 0, axis=1)
    # figure out number of batches with 0-padded data
    batches_with_missing = np.any(haps_with_zero == H, axis=1)
    return np.sum(batches_with_missing)


def convert_to_af(X: np.ndarray):
    """convert diploid image to 2-channel AF and dists

    Args:
        X (np.ndarray): _description_
    """
    C, H, W = X.shape

    X_new = np.zeros((3, W))

    gts = X[0, :, :]
    ac =  np.sum(gts, axis=0)
    an = gts.shape[0]
    af = ac / an


    assert np.max(af) <= 0.5
    af *= 2

    X_new[0] = af
    X_new[1] = X[1, 0, :]

    return X_new




def process_region(
    X: np.ndarray,
    positions: np.ndarray,
    norm_len: int = global_vars.L,
    convert_to_rgb: bool = False,
    n_snps: int = 32,
    convert_to_diploid: bool = False,
) -> np.ndarray:
    """
    Process an array of shape (n_sites, n_haps, 3), which is produced
    from either generated or real data.

    Zero-pad if necessary.

    Args:
        X (np.ndarray): feature array of shape (n_sites, n_haps, n_channels - 1)
        positions (np.ndarray): array of positions that correspond to each of the
            n_sites in the feature array.

    Returns:
        np.ndarray: _description_
    """
    # figure out how many sites and haplotypes are in the actual
    # multi-dimensional array
    W, H = X.shape
    # final_haps = n_haps // 2 if convert_to_diploid else n_haps
    # make sure we have exactly as many positions as there are sites
    assert W == positions.shape[0]

    # figure out the half-way point (measured in numbers of sites)
    # in the input array
    mid = W // 2
    half_S = n_snps // 2

    H_final = H // 2 if convert_to_diploid else H

    region = np.zeros(
        (
            3 if convert_to_rgb else global_vars.NUM_CHANNELS,
            H_final,
            n_snps,
        ),
        dtype=np.float32,
    )

    # should we divide by the *actual* region length?
    distances = inter_snp_distances(positions, norm_len)

    # first, transpose the full input matrix to be n_haps x n_snps
    X = X.T

    # if we have more than the necessary number of SNPs
    if mid >= half_S:
        # define indices to use for slicing
        i, j = mid - half_S, mid + half_S
        # add sites to output
        X = major_minor(X)

        # if the input data are phased haploids, we need to
        # create diploids
        if convert_to_diploid:
            # convert -1s back to 0s
            X[X == -1] = 0
            X = X.reshape((H_final, 2, W)).sum(axis=1)
            X = np.divide(X, 2)

        region[0, :, :] = X[:, i:j]
        # tile the inter-snp distances down the haplotypes
        # get inter-SNP distances, relative to the simualted region size
        distances_tiled = np.tile(distances[i:j], (H_final, 1))

        # add final channel of inter-snp distances
        region[1, :, :] = distances_tiled


    else:
        other_half_S = half_S + 1 if W % 2 == 1 else half_S
        i, j = half_S - mid, mid + other_half_S
        # use the complete genotype array
        # but just add it to the center of the main array
        X = major_minor(X)
        if convert_to_diploid:
            # convert -1s back to 0s
            X[X == -1] = 0
            X = X.reshape((H_final, 2, W)).sum(axis=1)
            X = np.divide(X, 2)
        region[0, :, i:j] = X

        # tile the inter-snp distances down the haplotypes
        distances_tiled = np.tile(distances, (H_final, 1))
        # add final channel of inter-snp distances
        region[1, :, i:j] = distances_tiled

    return region


def major_minor(matrix):
    """Note that matrix.shape[1] may not be S if we don't have enough SNPs"""

    # NOTE: need to fix potential mispolarization if using ancestral genome?
    n_haps, n_sites = matrix.shape

    # calculate the sum of derived alleles at each site
    derived_total = np.sum(matrix, axis=0)
    assert derived_total.shape[0] == n_sites

    # figure out where the derived sum is greater than half the haps
    switch_idxs = np.where(derived_total > (n_haps / 2))[0]
    matrix[:, switch_idxs] = 1 - matrix[:, switch_idxs]

    # change 0s to -1s
    matrix[matrix == 0] = -1

    return matrix


def average_crops(model, batch):
    bsz, c, ncrops, h, w = batch.shape
    batch = batch.permute(0, 2, 1, 3, 4)
    # merge ncrops and batchsize
    batch_reshaped = batch.reshape(bsz * ncrops, c, h, w)
    enc, proj = model(batch_reshaped)  # fuse batch size and ncrops
    _, enc_dims = enc.shape
    _, proj_dims = proj.shape
    # separate ncrops and batchsize again
    enc_reshaped = enc.reshape(bsz, ncrops, enc_dims)
    proj_reshaped = proj.reshape(bsz, ncrops, proj_dims)
    enc_avg = enc_reshaped.mean(1)  # avg over crops
    proj_avg = proj_reshaped.mean(1)  # avg over crops
    return enc_avg, proj_avg


class MultiCropTransform:
    """Create n random crops of the same image.
    returns shape (bsz, n_views, ...)"""
    def __init__(self, transform, n: int = 2):
        self.transform = transform
        self.n = n

    def __call__(self, x):
        stack = []
        for _ in range(self.n):
            stack.append(self.transform(x))
        return torch.stack(stack, dim=1)


if __name__ == "__main__":

    img = torch.rand(size=(1, 3, 256, 32))

    print (img.shape)

    trans = torchvision.transforms.RandomCrop(size=32)

    pre = TwoCropTransform(trans)
    print (pre(img).shape)
