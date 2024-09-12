SELECTION_VALS = ["0.01"]#, "0.025", "0.05", "0.1"]

TREE_NUMS, SELECTION_PARAMS = [], []

N_NEUTRAL = 5_000
N_SELECTION = 5_000

for s in SELECTION_VALS:
    
    TREE_NUMS.extend(list(range(N_SELECTION)))
    SELECTION_PARAMS.extend([s] * N_SELECTION)

print (min(TREE_NUMS), max(TREE_NUMS))
print (len(SELECTION_PARAMS), len(TREE_NUMS))

rule all:
    input:
        expand("data/slim/background/neutral_{TREE_NUM}.png", TREE_NUM=list(range(N_NEUTRAL))),
        expand("data/slim/foreground/{SELECTION}_{TREE_NUM}.png", zip, SELECTION=SELECTION_PARAMS, TREE_NUM=TREE_NUMS)


rule run_selection:
    input:
        slim_script = "slim/simulate_exp_selection.slim",
        slim_binary = "/uufs/chpc.utah.edu/common/HIPAA/u1006375/src/build/slim"
    output:
        trees = "slim/output/tree_selection_{SELECTION_STRENGTH}_{TREE_NUM}.trees",
        other = "slim/output/tree_selection_{SELECTION_STRENGTH}_{TREE_NUM}.junk"
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
        trees = "slim/output/tree_neutral_{TREE_NUM}.trees",
        other = "slim/output/tree_neutral_{TREE_NUM}.junk"
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
        "data/slim/foreground/{SELECTION_STRENGTH}_{TREE_NUM}.png"
    shell:
        """
        python {input.py_script} --tree {input.tree} --out {output} -n_smps 32 -n_snps 32
        """

rule process_neutral:
    input:
        tree = "slim/output/tree_neutral_{TREE_NUM}.trees",
        py_script = "process_slim_trees.py"
    output:
        "data/slim/background/neutral_{TREE_NUM}.png"
    shell:
        """
        python {input.py_script} --tree {input.tree} --out {output} -n_smps 32
        """
