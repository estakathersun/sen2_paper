import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import scanpy as sc
from make_hubs import *
import multiprocessing as mp


adata_path = './data/senepy_denovo_signatures_code/6.3_adata_ribo_downsampled.h5ad'
output_dir = './data/sp_downsampled_perm_out'
n_permutations = 1000
# n_workers = 64
n_workers = 32



def main():
    current_working_directory = os.getcwd()
    print(f"Current Working Directory: {current_working_directory}")
    os.chdir(current_working_directory)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Make folder {output_dir}')
    else:
        print(f'Folder {output_dir} already exists')

    if not os.path.isfile(adata_path):
        print(f"File '{adata_path}' not found. Exit...")
        sys.exit(1)

    print(f'Process file: {adata_path}')
    adata = sc.read_h5ad(adata_path)
    adata

    if 'Age_bin' not in adata.obs.columns:
        adata.obs['Age_bin'] = adata.obs['Age'].map(calc_age_bin)

    # all t-ct pairs
    tissue_cell_combinations = [(tissue, cell_type) for tissue in adata.obs['Tissue_global'].unique() 
                                for cell_type in adata.obs[adata.obs['Tissue_global'] == tissue]['cell_type_final'].unique().sort_values()]
    # example
    # [('Skin', 'CD4T'),
    # ('Skin', 'CD8T'),
    # ('Skin', 'DC'),
    # ...
    # ('Heart_Nuclei', 'Plasma cell'),
    # ('Heart_Nuclei', 'Unlabeled'),
    # ('Heart_Nuclei', 'Ventricular Cardiomyocyte')]

    for tissue, ct in tqdm(tissue_cell_combinations, 
                            total=len(tissue_cell_combinations), desc='Tissue.Cell combinations',
                            position=0, leave=True):
        if tissue == 'Heart_Nuclei':
            continue
        if ct == 'Unlabeled':
            continue

        tc_df_path = os.path.join(output_dir, f'{tissue}.{ct}.main_metrics.csv')
        perm_df_path = os.path.join(output_dir, f'{tissue}.{ct}.permutations_GAD.csv')

        if os.path.exists(tc_df_path) & os.path.exists(perm_df_path):
            print(f'Files already exist: {tc_df_path}, {perm_df_path}', file=sys.stderr)
            continue

        print(f'Обработка {tissue}.{ct}', file=sys.stderr)
        sub_adata = adata[(adata.obs.Tissue_global == tissue) & (adata.obs.cell_type_final == ct)].copy()
        if sub_adata.n_obs == 0:
            print(f'For {tissue}.{ct} no observations - skip.', file=sys.stderr)
            continue
        print(f'Filt genes (min_cells=1)', file=sys.stderr)
        sc.pp.filter_genes(sub_adata, min_cells=1)
        print(f'AnnData object with n_obs × n_vars = {sub_adata.n_obs} × {sub_adata.n_vars}', file=sys.stderr)
            
        print('Calculate GAD', file=sys.stderr)
        df = calculate_gene_proportions(sub_adata)
        tc_df = calculate_gad_score(df)
        print('Data permutation:', file=sys.stderr)
        perm_df = generate_null_distribution(sub_adata, n_permutations=n_permutations, n_workers=n_workers)
        genes = pd.Series(list(sub_adata.var_names))
        perm_df.insert(0, column='gene', value=genes)

        # if ct in ['MyoFib/VSMC', 'Mono/mac']:
        #     ct = ct.replace('/', '+')

        print(f'Saving: {tc_df_path}', file=sys.stderr)
        tc_df.to_csv(tc_df_path, index=False)
        print(f'Saving: {perm_df_path}', file=sys.stderr)
        perm_df.to_csv(perm_df_path, index=False)

if __name__ == "__main__":
    mp.freeze_support()  # !!!
    main()

