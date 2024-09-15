SELECTION_VALS = ["0.005", "0.01"]

TREE_NUMS, SELECTION_PARAMS = [], []

N_NEUTRAL = 10_000
N_SELECTION = N_NEUTRAL // len(SELECTION_VALS)

CLASS_LABELS = []

for si, s in enumerate(SELECTION_VALS):
    
    TREE_NUMS.extend(list(range(N_SELECTION)))
    SELECTION_PARAMS.extend([s] * N_SELECTION)
    CLASS_LABELS.extend([si] * N_SELECTION)

rule all:
    input:
        # expand("data/slim/foreground/{SELECTION}_{TREE_NUM}.png", zip, SELECTION=SELECTION_PARAMS, TREE_NUM=TREE_NUMS),
        expand("data/slim/foreground/{CLASS_LABEL}/{SELECTION}_{TREE_NUM}.png", zip, CLASS_LABEL=CLASS_LABELS, SELECTION=SELECTION_PARAMS, TREE_NUM=TREE_NUMS),
        expand("data/slim/background/0/neutral_{TREE_NUM}.png", TREE_NUM=list(range(N_NEUTRAL)))


rule run_selection:
    input:
        slim_script = "slim/simulate_exp_selection.slim",
        slim_binary = "/uufs/chpc.utah.edu/common/HIPAA/u1006375/src/build/slim"
    output:
        trees = temp("slim/output/tree_selection_{SELECTION_STRENGTH}_{TREE_NUM}.trees"),
        other = temp("slim/output/tree_selection_{SELECTION_STRENGTH}_{TREE_NUM}.junk")
    shell:
        """
        {input.slim_binary} -d mut={wildcards.SELECTION_STRENGTH} \
                            -d outpref=\\"slim/output/tree_selection_{wildcards.SELECTION_STRENGTH}_{wildcards.TREE_NUM}\\" \
                            -s {wildcards.TREE_NUM} \
                            {input.slim_script} > {output.other}
        """

rule run_neutral:
    input:
        slim_script = "slim/simulate_exp_neutral.slim",
        slim_binary = "/uufs/chpc.utah.edu/common/HIPAA/u1006375/src/build/slim"
    output:
        trees = temp("slim/output/tree_neutral_{TREE_NUM}.trees"),
        other = temp("slim/output/tree_neutral_{TREE_NUM}.junk")
    shell:
        """
        {input.slim_binary} -d mut=0 \
                            -d outpref=\\"slim/output/tree_neutral_{wildcards.TREE_NUM}\\" \
                            -s {wildcards.TREE_NUM} \
                            {input.slim_script} > {output.other}
        """

rule process_selection:
    input:
        tree = "slim/output/tree_selection_{SELECTION_STRENGTH}_{TREE_NUM}.trees",
        py_script = "process_slim_trees.py"
    output:
        "data/slim/foreground/{CLASS_LABEL}/{SELECTION_STRENGTH}_{TREE_NUM}.png"
    shell:
        """
        python {input.py_script} --tree {input.tree} --out {output} -n_smps 128 -n_snps 64
        """

rule process_neutral:
    input:
        tree = "slim/output/tree_neutral_{TREE_NUM}.trees",
        py_script = "process_slim_trees.py"
    output:
        "data/slim/background/0/neutral_{TREE_NUM}.png"
    shell:
        """
        python {input.py_script} --tree {input.tree} --out {output} -n_smps 128 -n_snps 64
        """
