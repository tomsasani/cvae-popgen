import argparse

import generator_fake
import util
import global_vars

import tskit
import pyslim
import numpy as np
import msprime
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from PIL import Image

import warnings
warnings.simplefilter('ignore', msprime.TimeUnitsMismatchWarning)


def simplify_tree(rts, n_samples, rng):
    alive_inds = pyslim.individuals_alive_at(rts, 0)
    keep_indivs = rng.choice(alive_inds, n_samples, replace=False)
    keep_nodes = []
    for i in keep_indivs:
        keep_nodes.extend(rts.individual(i).nodes)

    sts = rts.simplify(keep_nodes, keep_input_roots=True)

    return sts


def main(args):

    rng = np.random.default_rng(42)

    orig_ts = tskit.load(args.tree)
    rts = pyslim.recapitate(
        orig_ts,
        recombination_rate=1.25e-8, # same as in SLiM code
        ancestral_Ne=22552, # same as in SLiM code
        random_seed=42,
    )
    sts = simplify_tree(rts, args.n_smps, rng)
    sts = msprime.sim_mutations(
        sts,
        rate=1.25e-8,
        model=msprime.BinaryMutationModel(state_independent=False),
        random_seed=rng.integers(1, 2**32),
        discrete_genome=False,
    )

    # ensure all sites are segregating
    X, positions = generator_fake.prep_simulated_region(
        sts,
        filter_singletons=args.filter_singletons,
    )

    region = util.process_region(
        X,
        positions,
        norm_len=global_vars.L,
        convert_to_rgb=True,
        convert_to_diploid=True,
        n_snps=args.n_snps,
    )
    region = np.expand_dims(region, axis=0)

    region = region[0, 0, :, :]
    region = np.uint8(region * 255)

    img = Image.fromarray(region, mode="L")

    img.save(args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tree")
    p.add_argument("--out")
    p.add_argument("-filter_singletons", action="store_true")
    p.add_argument("-n_smps", type=int, default=100)
    p.add_argument("-n_snps", type=int, default=32)
    args = p.parse_args()
    main(args)
