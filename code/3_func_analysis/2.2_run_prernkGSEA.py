import os
import pandas as pd
from tqdm.auto import tqdm
import pickle

import gseapy as gp


deg_df = pd.read_csv('./data/sen_paper/2_pbDEGs_edgeR_NScFull.csv', index_col=0)

with open('./data/sen_paper/2.2_gsea_libraries.pkl', 'rb') as f:
    libs_dict = pickle.load(f)


out_path = './data/sen_paper/2.2_prnk_MSigDB_NScFull.csv'
# os.makedirs(out_path, exist_ok=True)

results = []
for t in tqdm(sorted(deg_df.Tissue.unique()), desc="Processing Tissues"):
    t_degs = deg_df[deg_df.Tissue == t]
    for ct in sorted(t_degs.cell_type.unique()):
        print(f'✨----- {t} - {ct} -----')
        ct_degs = t_degs[t_degs.cell_type == ct].copy()

        # Вычисляем метрику: -log10(FDR) * sign(logFC)
        # Она автоматически распределит гены от самых значимых UP до самых значимых DOWN
        # Заменяем 0 в FDR на минимальное ненулевое значение, чтобы избежать log(0)
        min_fdr = ct_degs['FDR'][ct_degs['FDR'] > 0].min()
        fdr_clean = ct_degs['FDR'].replace(0, min_fdr)

        # ct_degs['ranking_metric'] = -np.log10(fdr_clean) * np.sign(ct_degs['logFC'])
        ct_degs = ct_degs.set_index('gene_name')

        # Сортируем строго по созданной метрике
        # ct_degs = ct_degs.sort_values(by='ranking_metric', ascending=False)

        sigs = libs_dict["MSigDB_Hallmark_2020"]
        pre_res = gp.prerank(
            rnk=ct_degs.sort_values(by=['logFC', 'FDR'], ascending=[False, True])['logFC'], 
            outdir=None,
            gene_sets=sigs,
            ascending=None,
            min_size=2,
            format=None,
            verbose=False,
        )
        res2 = pre_res.res2d
        res2['Name'] = 'logFC'
        res2['Tissue'] = t
        res2['cell_type'] = ct
        results.append(res2)

res_df = pd.concat(results, axis=0)
res_df.to_csv(out_path)

