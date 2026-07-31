import os
import sys
import argparse
import pickle
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
import math
import time
import gzip

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests
import numba as nb
from multiprocessing import Pool, get_context
from functools import partial

AGE_BINS = np.array([10, 21, 31, 41, 51, 61, 71])

def calc_age_bin(x):
    '''return lower age bin bound'''
    for upper_bound in range(21,80,10):
        if x < 20:
            return 10
        elif upper_bound > x:
            return upper_bound - 10
    # for the oldest:
    return upper_bound


def _compute_proportions_from_matrix(X, obs, var_names, min_cells=100):
    """
    X : матрица (n_cells x n_genes) — numpy array или scipy sparse
    obs : DataFrame с индексом, соответствующим строкам X (содержит 'Age_bin' и т.д.)
    age_bins : список возрастных бинов, например [10,21,31,...]
    var_names : array-like с именами генов (len == n_genes)
    min_cells : порог, ниже которого ставим cell_proportion = -1
    Возвращает DataFrame с колонками ['gene','age_bin','cell_proportion']
    """
    genes = np.asarray(var_names)
    rows = []
    age_bins = AGE_BINS

    for age in age_bins:
        mask = (obs['Age_bin'] == age).values
        n_cells = int(mask.sum())

        if n_cells > min_cells:
            # подматрица
            Xsub = X[mask, :]
            if sparse.issparse(Xsub):
                counts = np.asarray(Xsub.getnnz(axis=0)).ravel()
            else:
                counts = np.asarray((Xsub > 0).sum(axis=0)).ravel()

            proportions = counts / float(n_cells)

            # заменяем нули на 1/n_cells 
            zero_mask = (proportions == 0)
            if zero_mask.any():
                proportions = proportions.astype(float)
                proportions[zero_mask] = 1.0 / float(n_cells)

            df_bin = pd.DataFrame({
                'gene': genes,
                'age_bin': age,
                'cell_proportion': proportions
            })

        else:
            df_bin = pd.DataFrame({
                'gene': genes,
                'age_bin': age,
                'cell_proportion': -1.0
            })

        rows.append(df_bin)

    return pd.concat(rows, ignore_index=True)


def calculate_gene_proportions(adata_subset, min_cells=100):
    """
    Векторизированный подсчёт доли клеток с экспрессией (UMI >= 1)
    для каждого гена и каждого возрастного бина.

    Возвращает DataFrame с колонками ['gene','age_bin','cell_proportion'].
    Для бинa с размером <= min_cells значение cell_proportion = -1.
    Для генов с нулевой долей внутри бина устанавливается 1/n_cells.
    """
    X = adata_subset.X
    obs = adata_subset.obs
    var_names = np.asarray(adata_subset.var_names)
    
    # делегируем всю логику вспомогательной функции
    return _compute_proportions_from_matrix(X, obs, var_names, min_cells=min_cells)



''' x =  cell proportion (%)'''

@nb.njit # Numba "No Python Just-In-Time"
def calc_old(x):
    # преобразуем х в np массив вне зависимости от того что за х
    # но из-за декортатора мы входные данные заранеее превращаем в нампай, так что тут смысла нет 
    # x = np.asarray(x)
    x = x * 100 # (%)
    # Создаёт пустой массив того же размера и формы, что x, но с типом float
    # его далее заполняем 
    result = np.empty(x.shape[0], dtype=np.float64)

    # 0 < x ≤ 3
    mask1 = (x > 0) & (x <= 3)
    result[mask1] = 1 / (1 + np.exp(-2 * x[mask1]))

    # 3 < x ≤ 20
    mask2 = (x > 3) & (x <= 20)
    result[mask2] = 1.0

    # x > 20
    mask3 = (x > 20)
    result[mask3] = -0.25 * x[mask3] + 6

    return result

@nb.njit 
def calc_young(x):
    # x = np.asarray(x)
    x = x * 100 # (%)
    result = np.empty(x.shape[0], dtype=np.float64)

    # x < 5
    mask1 = (x < 5)
    result[mask1] = 1.0

    # otherwise (x ≥ 5)
    mask2 = ~mask1
    result[mask2] = -0.5 * x[mask2] + 3.5

    return result

@nb.njit 
def calc_gain(x):
    # x = np.asarray(x)
    x = x * 100   # (%)
    result = np.empty(x.shape[0], dtype=np.float64)

    # if x < 5:
    mask1 = (x < 5)
    result[mask1] = x[mask1] / 5

    # if x > 15:
    mask2 = (x > 15)
    result[mask2] = -x[mask2]/5 + 4

    # otherwise (5 ≤ x ≤ 15)
    mask3 = ~(mask1 | mask2)
    result[mask3] = 1.0

    return result


# 2.3 Вычисление GAD-скоров (Gene Age-Dynamic scores) для генов
def get_metrics(row):
    '''где-то здесь сделать pivot табличку. Или изменить подход
    здесь row = cell proportion'''
    age_bins = np.array(AGE_BINS)
    row = np.asarray(row)
    mask = row > -1 # берем то, где клеток набралось достаточно для вычисления доли
    if mask.sum() < 2: # для линрегр мин 2 точки
        return (np.nan, np.nan, np.nan, np.nan, np.nan, int(mask.sum()), np.nan, np.nan, np.nan, np.nan)

    ages_filt = age_bins[mask]
    cell_props_filt = row[mask]

    lin = stats.linregress(ages_filt, cell_props_filt) # простая лин. регрессия x_filt ~ ages_filt (т.е. зависимость долей кл от возраста)

    # !!!* берем самый молодой бин из присутств. или делаем экстраполяцию
    if any(x in ages_filt for x in (10, 21, 31)):
        young_prop = cell_props_filt[0]   # ages_filt отсортирован по возрастанию
    else:
        young_prop = max(0, min(1, lin.slope * 21 + lin.intercept))

    old_prop = cell_props_filt[-1]

    old_vs_young_delta = old_prop - young_prop
    max_prop = cell_props_filt.max()

    # Наклон линейной регрессии в процентных пунктах
    # slope_pct = lin.slope * 100
  
    # Разница между самым старым бином и бином с макс. экспрессией
    max_idx = np.argmax(cell_props_filt)
    age_max = int(ages_filt[-1])              
    delta_max =  ages_filt[max_idx] - age_max 

    # по статье ур.6:
    # gad = old(x) + gain(x) + young(x) + dMax(delta_max) + aMax(age_max) + m * 5
    ##Slope (m) is represented in percentage points and given a weight multiplier of 5
    return (lin.slope, lin.rvalue, lin.pvalue, int(mask.sum()), max_prop, young_prop, old_prop, 
            old_vs_young_delta, delta_max, age_max)

'''далее к этим метрикам применяем формулы и считаем GAD - 
в зависимости от типа данных, нужно будет сопоставить функции для GAD с инпутом

если это input is pivot table, GAD считать можно сюда же, применяя формулы к столбцам'''

# tc_gene_props_df = calculate_gene_proportions(adata_subset)
def calculate_gad_score(tc_gene_props_df):
    """
    tc_gene_props_df: DataFrame с колонками ['gene','age_bin','cell_proportion']
    Возвращает tc_df с колонками и рассчитанным 'GAD'
    """
    age_bins = AGE_BINS
    tc_df = tc_gene_props_df.pivot(index='gene', columns='age_bin', values='cell_proportion').reset_index()
    metrics_cols = ['lin.slope', 'lin.rvalue', 'lin.pvalue', 'n_bins', 'max_prop', 
                    'young_prop', 'old_prop', 'old_vs_young_delta', 'delta_max', 'age_max']
    results = tc_df[age_bins].apply(get_metrics, axis=1)   # Series of tuples # если это дорого — можно заменить на numba вариант
    res_df = pd.DataFrame(results.tolist(), index=tc_df.index, columns=metrics_cols)
    tc_df[metrics_cols] = res_df

    tc_df['old_x'] = calc_old(tc_df['old_prop'].to_numpy(dtype=np.float64))
    tc_df['young_x'] = calc_young(tc_df['young_prop'].to_numpy(dtype=np.float64))
    tc_df['gain'] = calc_gain(tc_df['old_vs_young_delta'].to_numpy(dtype=np.float64))
    dm, am = tc_df['delta_max'], tc_df['age_max']
    tc_df['dMax'] = np.where(dm.isna(), np.nan, np.where(dm < 50, 1.0, -(dm - 50.0)))
    tc_df['aMax'] = np.where(am.isna(), np.nan, np.where(am >= 41, 1.0, am - 41.0))
    tc_df['GAD'] = tc_df['old_x'] + tc_df['gain'] + tc_df['young_x'] + tc_df['dMax'] + tc_df['aMax'] + (tc_df['lin.slope'] * 100) * 5
    return tc_df



def _perm_worker_shuffles(X_orig, obs, var_names,  min_cells, n_perms, seed_start=0):
    """
    Работает в отдельном процессе: делает n_perms пермутаций, для каждой:
      - перемешивает по столбцам X_work
      - считает пропорции через compute_proportions_from_matrix
      - считает GAD через calculate_gad_from_props
    Возвращает список Series GAD (index=genes).
    """
    # локальная копия матрицы
    if sparse.issparse(X_orig):
        X_work = X_orig.toarray()
    else:
        X_work = X_orig.copy()

    genes = np.asarray(var_names)
    results = []

    rng = np.random.RandomState(seed_start)

    for i in tqdm(range(n_perms), total=n_perms, desc='Permutations',
                  position=1, leave=False):
    # for i in range(n_perms):
        # перемешиваем по столбцам: используем RandomState.permutation для reproducibility
        for j in range(X_work.shape[1]):
            # inplace shuffle column j
            col_perm = rng.permutation(X_work[:, j])
            X_work[:, j] = col_perm

        # считаем пропорции по age_bins
        props_df = _compute_proportions_from_matrix(X_work, obs, genes, min_cells=min_cells)

        # считаем GAD
        gad_df = calculate_gad_score(props_df)
        gad_series = gad_df['GAD']
        gad_series.name = f'perm_{seed_start + i + 1}'
        results.append(gad_series)


    return results


def generate_null_distribution(adata_subset, n_permutations=1000, n_workers=4, min_cells=100, seed=42):
    """
    Параллельная генерация Null distribution.
    Возвращает DataFrame index=genes, колонки perm_1..perm_k с GAD.
    """
    X = adata_subset.X
    obs = adata_subset.obs
    var_names = adata_subset.var_names
    age_bins = AGE_BINS

    # распределим пермутации по воркерам
    base = n_permutations // n_workers
    counts = [base] * n_workers
    for i in range(n_permutations - base * n_workers):
        counts[i] += 1

    # создаём pool 
    ctx = get_context("fork")
    tasks = []
    seed_cursor = seed
    with ctx.Pool(processes=n_workers) as pool:
        for c in counts:
            tasks.append(pool.apply_async(_perm_worker_shuffles,
                                         (X, obs, var_names, min_cells, c, seed_cursor)))
            seed_cursor += c

        all_results = []

        for t in tasks:
            # блокируемся до получения результата воркера (список Series)
            res = t.get()
            all_results.extend(res)
                # обновляем прогресс на число пермутаций, сделанных этим воркером
                # pbar.update(len(res))    

    # объединяем Series в DataFrame
    perm_dfs = []
    for i, s in enumerate(all_results):
        # s уже Series с name
        perm_dfs.append(s)
    res_df = pd.concat(perm_dfs, axis=1)
    return res_df
