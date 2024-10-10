import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

cvae = []
for fh in glob.glob("results/cvae/*.csv"):
    df = pd.read_csv(fh).head(1)
    cvae.append(df)
cvae = pd.concat(cvae)
cvae["method"] = "cVAE"

vae = []
for fh in glob.glob("results/vae/*.csv"):
    df = pd.read_csv(fh).head(1)
    vae.append(df)
vae = pd.concat(vae)
vae["method"] = "VAE"

nmf = []
for fh in glob.glob("results/nmf/*.csv"):
    df = pd.read_csv(fh).head(1)
    nmf.append(df)
nmf = pd.concat(nmf)
nmf["method"] = "NMF"

combined = pd.concat([vae, cvae, nmf])
print (combined)

f, ax = plt.subplots()
sns.boxplot(
    data=combined,
    x="method",
    y="salient_silhouette",
    color='w',
    ax=ax,
    order=["NMF", "VAE", "cVAE"],
    fliersize=0,
)
sns.stripplot(
    data=combined,
    x="method",
    y="salient_silhouette",
    ax=ax,
    order=["NMF", "VAE", "cVAE"],
    hue="method"
)
# ax.set_title("cVAE separates cancer tissue types better than\nboth VAE and NMF")
ax.set_ylabel("Silhouette score")
ax.set_xlabel("Method")
sns.despine(ax=ax)
f.savefig("perf.png", dpi=200)
