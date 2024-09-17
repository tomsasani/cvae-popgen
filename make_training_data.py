import demographies
import generator_fake
import params
import util
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# we'll only be changing the size of the "A" population at present,
# after a bottleneck from a population size of N3
PARAM_NAMES = ["N1", "N2", "T1", "T2", "growth"]

# initialize basic engine
engine = generator_fake.Generator(
    demographies.simulate_exp,
    PARAM_NAMES,
    42,
    convert_to_rgb=True,
    n_snps=64,
    convert_to_diploid=True,
    seqlen=100_000,
    sort=True,
    filter_singletons=False,
)

sim_params = params.ParamSet()
rng = np.random.default_rng(42)

# define parameter values we'll use for target dataset,
# in which we have a population size change in the admixed
# population
TG_PARAM_VALUES = [
    [23_231, 29_962, 4_870, 581, 0.00531],  # YRI
    [22_552, 3_313, 3_589, 1_050, 0.00535],  # CEU
    [9_000, 5_000, 2_000, 350, 0.001],
]  # CHB

N_SMPS = 128
TOTAL_REGIONS = 90_000
target_regions = TOTAL_REGIONS / len(TG_PARAM_VALUES)

for model_i in range(len(TG_PARAM_VALUES)):
    counted = 0
    while counted < target_regions:

        param_values = TG_PARAM_VALUES[model_i]

        region = engine.sample_fake_region(
            [N_SMPS],
            param_values=param_values,
        )

        n_batches_zero_padded = util.check_for_missing_data(region) > 0
      
        if n_batches_zero_padded:
            continue

        if counted % 100 == 0:
            print(counted)

        outpref = "train" if counted < int(target_regions * 0.8) else "test"

        region = region[0, 0, :, :]
        region = np.uint8(region * 255)

        img = Image.fromarray(region, mode="L")

        outpath = f"data/simulated/target/{outpref}/{model_i}"
       
        p = pathlib.Path(outpath)
        if not p.is_dir():
            p.mkdir(parents=True)

        img.save(f"{outpath}/{counted}.png")

        counted += 1


# counted = 0
# while counted < 1:
    
#     region = engine.sample_fake_region(
#             [N_SMPS],
#             param_values=[0.005],
#         )
#     n_batches_zero_padded = util.check_for_missing_data(region) > 0
      
#     if n_batches_zero_padded:
#         continue

#     if counted % 100 == 0:
#         print(counted)

#     outpref = "train" if counted < int(TOTAL_REGIONS * 0.8) else "test"

#     region = region[0, 0, :, :]
#     region = np.uint8(region * 255)

#     img = Image.fromarray(region, mode="L")

#     outpath = f"data/simulated/background/{outpref}/0"
    
#     p = pathlib.Path(outpath)
#     if not p.is_dir():
#         p.mkdir(parents=True)

#     img.save(f"{outpath}/{counted}.png")

#     counted += 1
