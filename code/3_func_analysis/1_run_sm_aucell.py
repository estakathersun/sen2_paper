
import scanpy as sc
import numpy as np
import pandas as pd
from tqdm import tqdm

import decoupler as dc


adata = sc.read_h5ad("./data/adata_all_harm_ribo.h5ad")
sp_scores = pd.read_csv(
    "./data/sen_paper/sp_2nd_scores_is_sen_labels.csv",
    index_col=0,
)

adata.obs.drop(columns=["sp_score_sigs"], inplace=True, errors="ignore")
adata.obs = adata.obs.join(sp_scores, how="left")
adata = adata[adata.obs["is_sen_GMMk2_median_ct"].notna()].copy()

adata.obs["sen_group"] = pd.NA
# --- 2. Sc (сенесцентные по GMM) ---
mask_sc = adata.obs["is_sen_GMMk2_median_ct"]
adata.obs.loc[mask_sc, "sen_group"] = "Sc"
adata.obs.loc[~mask_sc, "sen_group"] = "NSc"


filepath = "./data/senepy_denovo_signatures_code/SAUL_SEN_MAYO.v2026.1.Hs.txt"
with open(filepath, "r") as f:
    senmayo_genes = [line.strip() for line in f]

sen_genes = list(set(senmayo_genes) & set(adata.var_names))
net = pd.DataFrame({"source": ["SenMayo"] * len(sen_genes), "target": sen_genes})

grouped = adata.obs.groupby(["Tissue_global", "cell_type_final"], observed=True)
adata.obs["sm_score_aucell"] = np.nan

for (t, ct), idx in tqdm(grouped.groups.items(), total=len(grouped)):
    adata_sub = adata[idx].copy()
    adata_sub = adata_sub[adata_sub.obs["sen_group"].notna()].copy()
    # adata_sub = adata_sub[adata_sub.obs.n_counts < 75000].copy()
    dc.mt.aucell(
        data=adata_sub,
        net=net,
        # tmin= фильтр на слишком маленькие сигнатуры из net
        # layer=
        raw=False,  # if True, игнорирует layers
        empty=True,  # удаляет пустые клетки и гены
        verbose=True,
        n_up=200, # change values
    )
    scores = adata_sub.obsm["score_aucell"]["SenMayo"]
    adata.obs.loc[adata_sub.obs_names, "sm_score_aucell"] = scores

adata.obs.to_csv('./data/sen_paper/1_obs_sm_aucell200.csv')