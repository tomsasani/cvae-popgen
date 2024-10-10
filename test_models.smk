TG = "data/target_spectra.npz"
BG = "data/background_spectra.npz"

TG = "data/cancer_tg_spectra.npz"
BG = "data/cancer_bg_spectra.npz"

rule all:
    input:
        expand("results/cvae/{TRIAL}.csv", TRIAL=range(5)),
        expand("results/vae/{TRIAL}.csv", TRIAL=range(5)),
        expand("results/nmf/{TRIAL}.csv", TRIAL=range(5))

rule run_cvae:
    input:
        tg = TG,
        bg = BG,
        py = "train_cvae_spectra.py"
    output: "results/cvae/{TRIAL}.csv"
    shell:
        """
        python {input.py} --bg {input.bg} \
                        --tg {input.tg} \
                        --out {output} \
                        -epochs 25 \
                        -resample
        """

rule run_vae:
    input:
        tg = TG,
        bg = BG,
        py = "train_vae_spectra.py"
    output: "results/vae/{TRIAL}.csv"
    shell:
        """
        python {input.py} --bg {input.bg} \
                                  --tg {input.tg} \
                                  --out {output} \
                                  -epochs 25 \
                                  -resample
        """

rule run_nmf:
    input:
        tg = TG,
        bg = BG,
        py = "run_cpca.py"
    output: "results/nmf/{TRIAL}.csv"
    shell:
        """
        python {input.py} --bg {input.bg} \
                        --tg {input.tg} \
                        --out {output} \
        """
