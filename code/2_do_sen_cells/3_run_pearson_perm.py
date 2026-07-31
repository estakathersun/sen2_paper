import itertools
import os
# before numpy import
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import scanpy as sc
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from scipy import stats
# from joblib import Parallel, delayed
from multiprocessing import Pool
import multiprocessing as mp
# при большом кол-ве пар можно рассмотреть вариант при котором 
# сначала вычисляются все корреляции, а затем пермутации для тех что |r| > thr

# run: python -u run_pearson_perm.py 2>&1 | tee -a log.txt

########### logger setup ################
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)  # не ломает прогрессбар

def setup_logger(name="name"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = TqdmLoggingHandler()
    formatter = logging.Formatter(
        "%(asctime)s | [%(levelname)s] %(name)s - %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger
################################################

def pearson_r_and_p_from_vectors(x, y):
    """
    Быстрый вычислитель r и p по векторам (1D numpy, dtype float/bool/int).
    Возвращает (r, p_two_sided).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 3:
        return 0.0, 1.0 # иначе приколы с t_stat
    # центрируем
    xm = x.mean()
    ym = y.mean()
    xc = x - xm
    yc = y - ym
    denom = np.sqrt(np.sum(xc*xc) * np.sum(yc*yc))
    if denom == 0:
        return 0.0, 1.0  # константный вектор -> r=0, p=1
    r = (xc @ yc) / denom
    # ограничим r в (-1,1)
    r = max(min(r, 0.9999999), -0.9999999)
    # p-value по t-распределению (двусторонний)
    t_stat = r * np.sqrt((n - 2) / (1 - r*r))
    p = 2.0 * stats.t.sf(abs(t_stat), df=n-2)
    return r, p

def pair_permutation_stats(a1, a2, n_perms=500, rng=None):
    """
    Выполняет permutation test для пары векторов (перемешиваются оба)
    - a1, a2: векторы генов, 1D numpy arrays (обычно булевы 0/1).
    - n_perms: число пермутаций.
    Возвращает:
      observed_r, observed_p, rnd_r_mean, rnd_p_mean, rnd_r_q99, rnd_p_q01, rnd_r_sd, rnd_p_sd
    """
    if rng is None:
        rng = np.random.default_rng()
    # наблюдаемое
    r_obs, p_obs = pearson_r_and_p_from_vectors(a1, a2)

    rs = np.empty(n_perms, dtype=float)
    ps = np.empty(n_perms, dtype=float)
    n = a1.size # предполагается что a1.size == a2.size

    for i in range(n_perms):
        perm_idx1 = rng.permutation(n)
        perm_idx2 = rng.permutation(n)
        a1_perm = a1[perm_idx1]
        a2_perm = a2[perm_idx2]

        rs[i], ps[i] = pearson_r_and_p_from_vectors(a1_perm, a2_perm)
        
    return (r_obs, p_obs,
            rs.mean(), ps.mean(),
            np.quantile(rs, 0.99), np.quantile(ps, 0.01),
            rs.std(ddof=1), ps.std(ddof=1))


# Глобальные переменные для пула
X_global = None
sub_var2idx_global = None

def init_worker(memmap_path, sub_var2idx):
    """Загружает memmap один раз на процесс вместо копирования AnnData."""
    global X_global, sub_var2idx_global
    X_global = np.load(memmap_path, mmap_mode='r')  # <-- memmap read-only
    sub_var2idx_global = sub_var2idx


# Переписываем process_pair для multiprocessing
def process_pair_mp(args):
    """args = (tissue, cell, gene1, gene2, n_perms, seed)
        Использует X_global и sub_var2idx_global"""
    tissue, cell, gene1, gene2, n_perms, seed = args
    if seed is None:
        seed = 0
    rng = np.random.default_rng(seed)
    i1 = sub_var2idx_global[gene1]
    i2 = sub_var2idx_global[gene2]
    # извлекаем столбцы только для этих двух генов из sub (без преобразования всех данных в dense)
    col1 = X_global[:, i1]
    col2 = X_global[:, i2]
    # binarize
    a1 = (col1 > 0).astype(np.uint8)
    a2 = (col2 > 0).astype(np.uint8)

    r_obs, p_obs, rnd_r_mean, rnd_p_mean, rnd_r_q99, rnd_p_q01, rnd_r_sd, rnd_p_sd = \
        pair_permutation_stats(a1, a2, n_perms=n_perms, rng=rng)

    return [tissue, cell, gene1, gene2, r_obs, p_obs,
            rnd_r_mean, rnd_p_mean, rnd_r_q99, rnd_p_q01, rnd_r_sd, rnd_p_sd]


def filter_genes_intersect(sub_df, qval_thr=0.05, pval_thr=0.01, hypergeom_pval_thr=0.05, gene_set_thr=0):
    """
    Фильтрация генов с проверкой статистического обогащения пересечения.
    
    Отбирает гены, удовлетворяющие условию:
        (q_bh < qval_thr) ИЛИ (origFilt=True И pval < pval_thr)
    
    Затем проверяет значимость пересечения отобранных генов с биологически 
    отфильтрованными генами (origFilt=True) с помощью гипергеометрического теста.
    
    Параметры гипергеометрического теста:
        M = len(sub_df)          — размер универсума (все гены в sub_df)
        n = len(bio_set)         — число генов с origFilt=True
        N = len(sel_set)         — число отобранных генов
        k = len(intersect)       — размер пересечения
    
    Args:
        sub_df (pd.DataFrame): Датафрейм с колонками 'gene', 'GAD', 'q_bh', 
            'pval', 'origFilt'.
        qval_thr (float): Порог FDR-скорректированного p-value (q_bh). 
            По умолчанию 0.05.
        pval_thr (float): Порог исходного p-value для генов с origFilt=True. 
            По умолчанию 0.01.
        hypergeom_pval_thr (float): Порог p-value гипергеометрического теста 
            для подтверждения обогащения пересечения. По умолчанию 0.05.
        gene_set_thr (int): Минимально допустимый размер пересечения. 
            По умолчанию 0 (отключено).
    
    Returns:
        set: Множество генов из пересечения, прошедших все фильтры.
             Возвращает пустое множество, если:
             - гипергеометрический тест не значим (p > hypergeom_pval_thr),
             - p-value теста некорректен (NaN),
             - размер пересечения < gene_set_thr.
    
    Примечание:
        Сортировка по 'GAD' выполняется для внутренней упорядоченности данных,
        не влияет на результат фильтрации.
    """
        # если процессировать весь df - создать словарь sel = {}, в конце sel[c] = list(сет генов интереса)
        
    sub_df = sub_df.sort_values('GAD', ascending = False)
    sub_df = sub_df.reset_index(drop = True)
    selected_df = sub_df[(sub_df.q_bh < qval_thr) | ((sub_df.origFilt) & (sub_df.pval < pval_thr))]
    bio_set = set(sub_df[sub_df.origFilt].gene) # множество генов, True по биологическому фильтру
    sel_set = set(selected_df.gene) # основной фильтр
    intersect = bio_set.intersection(sel_set)
    # d = len(sel_set) - len(bio_set)

    p = stats.hypergeom.sf(len(intersect) - 1, len(sub_df), len(bio_set), len(sel_set))

    if p > hypergeom_pval_thr or np.isnan(p) or len(intersect) < gene_set_thr:
            return set()
    return intersect



    ############ RUN ##########
def main():
    # инициализация логгера
    logger = setup_logger("pearson_perm")
    logger.info('read adata...\n')
    adata = sc.read_h5ad('./data/senepy_denovo_signatures_code/6.3_adata_ribo_downsampled.h5ad')
    logger.info('read tc_df...\n')
    tc_df = pd.read_csv('./data/all_tc_GAD_stats_downsampled.csv', index_col=0)

    tc_groups = np.sort(tc_df['tissue.cell'].unique())

    # Параллельность
    n_processes = 128
    out_dir = './data/senepy_denovo_signatures_code/sp_pearson_perm_output_dw_filt2/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        logger.info(f'Создаю папку {out_dir}')
    else:
        logger.warning(f'Директория {out_dir} уже существует')

    for tc in tqdm(tc_groups, total=len(tc_groups), desc='Tissue.Cell combinations', position=0, leave=True):
        logger.info(f'обработка {tc}')
        out = []
        out_path = os.path.join(out_dir, f'{tc}.pearson_perm_stats.csv')

        if os.path.exists(out_path):
            logger.warning(f'файл уже существует: {out_path}')
            continue

        tissue, cell = tc.split('.')

        # ограничиваем бином 51 включительно
        sub_adata = adata[(adata.obs.Tissue_global == tissue) \
                        & (adata.obs.cell_type_final == cell) \
                        & (adata.obs.Age >= 51)].copy()
        if sub_adata.n_obs < 100:
            logger.warning(f'❗в подвыборке {tc} недостаточно клеток: {sub_adata.n_obs}. skip...')
            continue

        sub_var2idx = {gene: idx for idx, gene in enumerate(sub_adata.var_names)}

        ######## фильтрация генов #########
        tc_filt_df = tc_df[(tc_df['tissue.cell'] == tc) & (tc_df['lin.slope']>0)] 
        # по lin.slope в самом начале обязательно! это критерий динамичного гена вне завис-ти от фильтров далее

        # # 1) основной фильтр, который даст нам кучу генов:
        # tc_filt_df = tc_filt_df[(tc_filt_df.q_bh < 0.01) | ((tc_filt_df.origFilt) & (tc_filt_df.pval < 0.01))]
        # genes_set = set(tc_filt_df.gene)

        # 2) берем только пересечение основного фильтра с биологическим:
        genes_set = filter_genes_intersect(tc_filt_df, qval_thr=0.01, pval_thr=0.01, hypergeom_pval_thr=0.05, gene_set_thr=0)
        if len(genes_set) == 0:     # множество пустое. или выставить тут gene_set_thr
            logger.warning(f'❗множество генов для {tc} пустое. skip...')
            continue
        
        logger.info(f"начинаем пермутации... генов в множестве: {len(genes_set)}")

        #  сохраняем X в memmap
        X_dense = sub_adata.X
        if hasattr(X_dense, "toarray"):
            X_dense = X_dense.toarray()
        memmap_file = f"/tmp/{tc}_X_memmap.npy"
        np.save(memmap_file, X_dense)  # сохраняем для memmap
        del X_dense  # освобождаем память родителя

        pairs = itertools.combinations(genes_set, 2) # создали генератор
        # создаем список задач для пула:
        tasks = [(tissue, cell, g1, g2, 500, hash((g1, g2)) & 0xffffffff) for g1, g2 in pairs]

        n_pairs = len(genes_set) * (len(genes_set) - 1) / 2
        chunksize = max(2, min(1000, int(n_pairs // (2 * n_processes))))
        # создаем пул с initializer
        with Pool(processes=n_processes, initializer=init_worker, initargs=(memmap_file, sub_var2idx)) as pool:
            results = list(tqdm(pool.imap(process_pair_mp, tasks, chunksize=chunksize), total=len(tasks),
                                desc=f"{tc} | permutation", leave=False))

        out.extend(results)

        if len(results) == 0:
            logger.error(f'❗с результатами для {tc} что-то пошло не так - список пустой')
            continue

        test = pd.DataFrame(out, columns=['tissue', 'cell', 'gene1', 'gene2', 'r', 'p',
                                        'rnd_r_mean', 'rnd_p_mean', 'rnd_r_q99', 'rnd_p_q01', 'rnd_r_sd', 'rnd_p_sd'])
        test.to_csv(out_path, index=False)
        logger.info(f'✅датафрейм сохранен в {out_path}\n🎉🎉✨')

        del results
        del test


if __name__ == "__main__":
    mp.freeze_support()  # безопасно и корректно
    main()