import os, math
from pathlib import Path
import scanpy as sc
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scanpy.external as sce
from tqdm import tqdm
from scipy import stats
from scipy.stats import pearsonr, spearmanr, gaussian_kde
from sklearn.mixture import GaussianMixture

def filt_outlier_gsms(obs, score_col="ds", return_filt=False):
    '''определеяет в каждом кл типе доноров, медиана которых отклоняется от медианы gse. 
       При return_filt=True возвращает отфильтрованный датафрейм'''

    outlier_index = []  # индексы клеток на удаление
    
    for (t, ct, gse), df_gse in obs.groupby(
        ["Tissue_global", "cell_type_final", "GSE_id"],
        observed=True):
        vals = df_gse[score_col].values

        # --- GSE уровень ---
        median_gse = np.median(vals)
        mad_gse = stats.median_abs_deviation(vals)

        # защита
        if mad_gse == 0:
            continue

        threshold = median_gse + 2.5 * mad_gse

        gsm_medians = (df_gse.groupby("GSM_id")[score_col].median())

        gsm_age = (df_gse.groupby("GSM_id")["Age"].first()) # один на GSM

        outliers = gsm_medians[gsm_medians > threshold]
        for gsm, val in outliers.items():
            n_cells_gsm = (df_gse["GSM_id"] == gsm).sum()
            n_cells_total = len(df_gse)
            age = gsm_age.loc[gsm]
            print(f"{t} | {ct} | {gse} | {gsm} | age={age} | "
            f"median={val:.3f} | cells (gsm/gse)={n_cells_gsm}/{n_cells_total}")
            outlier_index.extend(df_gse[df_gse["GSM_id"] == gsm].index)

    if return_filt == True:
        obs_clean = obs.drop(index=outlier_index)
        return obs_clean


def plot_violins_by_gsm(obs, score_col=None, grid=False):
    for t in obs.Tissue_global.unique():
        obs_t = obs[obs.Tissue_global == t].copy()

        # глобальный диапазон как у тебя
        all_scores = obs_t[score_col].dropna()
        # xmin, xmax = np.quantile(all_scores, 0.001), np.quantile(all_scores, 0.999)
        # xmin, xmax = all_scores.min(), all_scores.max()

        cell_types = sorted(obs_t['cell_type_final'].unique())

        ncols = 2
        n_plots = len(cell_types)
        nrows = math.ceil(n_plots / ncols)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            # figsize=(7 * ncols, 10* nrows),
            figsize=(18*ncols, 7*nrows),
            # sharey=True
            )
        axes = axes.flatten()

        # --- цвета для GSE ---
        gse_list = sorted(obs_t['GSE_id'].unique())
        palette = dict(zip(gse_list, sns.color_palette("Set2", len(gse_list))))

        for i, ct in enumerate(cell_types):
            ax = axes[i]
            df_ct = obs_t[obs_t['cell_type_final'] == ct].copy()
            xmin, xmax = np.quantile(df_ct[score_col].dropna(), 0.001), np.quantile(df_ct[score_col].dropna(), 0.999)

            # --- агрегируем мета для подписей ---
            gsm_meta = (
                df_ct
                .groupby("GSM_id")
                .agg(Age=("Age", "first"),
                    n_cells=(score_col, "count"),
                    GSE_id=("GSE_id", "first"))
                .reset_index()
                .sort_values(["GSE_id", "Age"], ascending=True))

            # порядок GSM
            order = gsm_meta["GSM_id"].tolist()

            # подписи
            label_map = {row.GSM_id: f"{row.GSM_id}\n{int(row.Age)} | n={row.n_cells}" 
                        for _, row in gsm_meta.iterrows()}

            df_ct["GSM_label"] = df_ct["GSM_id"].map(label_map)

            order_labels = [label_map[gsm] for gsm in order]

            # --- violin ---
            sns.violinplot(
                data=df_ct,
                x="GSM_label",
                y=score_col,
                hue="GSE_id",
                order=order_labels,
                palette=palette,
                fill=False, inner='quarters', linewidth=2,
                ax=ax
            )

            # стрипплот из благостных истоков, на всякий:
            # sns.stripplot(data=df, x='donor_label', y=score_col, hue='GSE_id', ax=ax, size=2, jitter=True)

            # убираем дубли легенды (важно!)
            ax.legend_.remove()

            ax.set_title(ct)
            ax.set_ylim(xmin, xmax)
            # Автоматически подобрать основные деления
            # ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
            # Добавить мелкие деления между ними
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
            # ax.tick_params(axis='y')
            ax.tick_params(axis='x', rotation=90) # для вертикальных скрипок и категорий по х
            # Сетка
            if grid:
                ax.grid(True, which='major', axis='y', alpha=0.5)
                ax.grid(True, which='minor', axis='y', alpha=0.15)
        # удаляем пустые оси
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # --- общая легенда ---
        handles = [
            plt.Line2D([0], [0], color=palette[g], lw=4)
            for g in gse_list]

        fig.legend(handles, gse_list, title="GSE_id", loc="upper right", bbox_to_anchor=(0.98, 0.98))

        fig.suptitle(t, fontsize=16)

        plt.tight_layout(rect=[0, 0, 0.95, 0.96])
        plt.show()


def binarize_df_raw(df, score_col="ds", group_col="SnC", verbose=True):
    from sklearn.mixture import GaussianMixture

    # --- 1. Подготовка данных ---
    data = df[score_col].values.reshape(-1, 1)

    # --- 2. Инициализация GMM ---
    means_init = np.array([
        [np.median(data)],
        [np.percentile(data, 90)]
    ])

    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=0,
        means_init=means_init)

    # --- 3. Обучение ---
    gmm.fit(data)
    assignments = gmm.predict(data)

    # --- 4. Определение "высокого" кластера ---
    max_assignment = assignments[np.argmax(df[score_col].values)]

    # --- 5. Поиск threshold ---
    sorted_indices = df[score_col].argsort().values

    threshold = None
    for idx in reversed(sorted_indices):
        if assignments[idx] != max_assignment:
            threshold = df.iloc[idx][score_col]
            break

    if threshold is None:
        print('thr is None')
        return None

    # --- 6. Бинаризация ---
    df["binary"] = np.where(df[score_col] > threshold,"SnC","Normal")

    # --- 7. Обработка group_col ---
    if group_col not in df.columns:
        df[group_col] = "in-vivo"

    # --- 8. Визуализация ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    colors = ["#ff1b6b", "#45caff"]

    # --- Plot 1 ---
    sns.histplot(
        df[score_col],
        kde=False,
        ax=axes[0],
        color="lightgrey",
        bins=50,
        stat="density",
    )

    for idx, level in enumerate(df[group_col].unique()):
        sns.kdeplot(
            df[df[group_col] == level][score_col],
            ax=axes[0],
            label=level,
            color=colors[idx % len(colors)],
        )

    axes[0].set_title("Score distribution by condition")
    axes[0].legend(title=group_col)
    axes[0].set_xlabel(score_col)
    axes[0].set_ylabel("Density")
    axes[0].set_xlim(df[score_col].min(), df[score_col].max())

    # --- Plot 2 ---
    x = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)
    pdf = np.exp(gmm.score_samples(x))

    axes[1].hist(data, bins=30, density=True, alpha=0.5, color="gray", label="Data",
                    range=(data.min(), data.max()))
    axes[1].plot(x, pdf, label="Mixture of 2 Gaussians", color="red")

    weights = gmm.weights_
    means = gmm.means_.flatten()
    covariances = np.sqrt(gmm.covariances_.reshape(-1))

    for weight, mean, cov in zip(weights, means, covariances):
        component_pdf = (
            weight
            * (1 / (np.sqrt(2 * np.pi) * cov))
            * np.exp(-0.5 * ((x - mean) / cov) ** 2)
        )
        axes[1].plot(x, component_pdf, label=f"Gaussian: μ={mean:.2f}, σ={cov:.2f}")

    axes[1].axvline(x=threshold, color="blue", linestyle="--")

    axes[1].set_xlabel(score_col)
    axes[1].set_ylabel("Density")
    axes[1].set_title("Mixture")
    axes[1].legend()
    axes[1].set_xlim(data.min(), data.max())

    plt.tight_layout()
    plt.show()

    if verbose:
        print(f"Threshold: {threshold:.4f}")

    return df


def kde_mode_and_sigma(x, grid_size=1000):
    """
    Оценивает моду распределения через KDE и локальную ширину пика.
    Ширина считается через FWHM (Full Width at Half Maximum) и переводится в sigma:
        sigma ≈ FWHM / 2.355
    NB: sigma — не точная дисперсия, а локальная оценка ширины (работает для skewed данных).
    """
    kde = gaussian_kde(x)
    grid = np.linspace(x.min(), x.max(), grid_size)
    density = kde(grid)
    
    mode_idx = np.argmax(density)
    mode = grid[mode_idx]
    peak = density[mode_idx]
    
    # half max
    half = peak / 2
    
    # ищем точки слева и справа
    left_idx = np.where(density[:mode_idx] <= half)[0]
    right_idx = np.where(density[mode_idx:] <= half)[0]
    
    if len(left_idx) == 0 or len(right_idx) == 0:
        return mode, np.std(x)  # fallback
    
    left = grid[left_idx[-1]]
    right = grid[mode_idx + right_idx[0]]
    
    fwhm = right - left
    
    # FWHM -> sigma для гауссианы
    sigma = fwhm / 2.355
    return mode, sigma


def plot_gmm_results(df_raw, data, gmm, threshold, high_comp, order, score_col):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    # --- Plot 1 ---
    sns.histplot(
        df_raw[score_col],
        kde=False,
        ax=axes[0],
        color="lightgrey",
        bins=50,
        stat="density",
    )
    sns.kdeplot(df_raw[score_col],ax=axes[0],)

    axes[0].set_title("Raw score distribution")
    axes[0].set_xlabel(score_col)
    axes[0].set_ylabel("Density")
    axes[0].set_xlim(df_raw[score_col].min(), df_raw[score_col].max())

    # --- Plot 2 ---
    x = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)
    pdf = np.exp(gmm.score_samples(x))
    axes[1].hist(
        data,
        bins=30,
        density=True,
        alpha=0.5,
        color="gray",
        label="Data",
        range=(data.min(), data.max())
    )
    axes[1].plot(x, pdf, label=f"Mixture of {gmm.n_components} Gaussians", color="red")

    # --- универсальная работа с компонентами ---
    means = gmm.means_.flatten()[order]
    weights = gmm.weights_[order]
    covariances = gmm.covariances_.reshape(-1)[order]

    for i, (weight, mean, cov) in enumerate(zip(weights, means, np.sqrt(covariances))):
        component_pdf = (
            weight
            * (1 / (np.sqrt(2 * np.pi) * cov))
            * np.exp(-0.5 * ((x - mean) / cov) ** 2)
        )
        label = f"Gaussian {i}: μ={mean:.2f}, σ={cov:.2f}"
        if i == high_comp:
            label += " (high)"
        axes[1].plot(x, component_pdf, label=label)

    # --- threshold ---
    if threshold is not None:
        axes[1].axvline(x=threshold, color="blue", linestyle="--")

    # --- posterior ---
    probs_plot = gmm.predict_proba(x)[:, order][:, high_comp]
    axes[1].plot(x, probs_plot, linestyle=":", label="Posterior (high)")

    axes[1].set_xlabel(score_col)
    axes[1].set_ylabel("Density")
    axes[1].set_title("Mixture")
    axes[1].legend()
    axes[1].set_xlim(df_raw[score_col].min(), df_raw[score_col].max())

    plt.tight_layout()
    plt.show()


def binarize_df_GMMk2(df_raw, df_crop=None, score_col="ds", group_col="SnC", verbose=True):
    from sklearn.mixture import GaussianMixture
    # =========================
    # ПАРАМЕТРЫ (в теле функции)
    # =========================
    target_weight_k2 = 0.1        # таргетная доля "высокой" компоненты
    weight_tolerance = 0.05       # допустимый диапазон (пока не используем жестко, но оставляем)
    posterior_thr = 0.6           # порог по апостериорной вероятности
    separability_thr = 0.5        # минимальная разделимость
    max_reruns = 10               # максимум перезапусков
    base_percentile = 40
    step_percentile = 2           # шаг увеличения percentiles 

    # --- 1. Подготовка данных ---
    if df_crop is None:
        df_crop = df_raw.copy()
    data = df_crop[score_col].values.reshape(-1, 1)
    p99 = np.percentile(data, 99.9)
    data = np.clip(data, None, p99)

    # --- 2. Базовая инициализация ---
    rerun = 0
    converged = False

    while rerun <= max_reruns:
        p = base_percentile + rerun * step_percentile
        p = min(p, 99.9)  # защита от выхода за пределы

        means_init = np.array([[np.percentile(data, 5)],[np.percentile(data, 95)]])
        weights_init = np.array([1 - target_weight_k2,target_weight_k2])

        gmm = GaussianMixture(
            n_components=2,
            random_state=0,
            means_init=means_init,
            weights_init=weights_init,
            precisions_init=np.array([3 / data.var(), 10 / data.var()]).reshape(2, 1, 1),
            reg_covar=1e-6,
            n_init=5,
            max_iter=100
        )
        # --- 3. Обучение ---
        gmm.fit(data)
        converged = gmm.converged_
        n_iter_ = gmm.n_iter_

        # --- 4. Определение "высокой" компоненты ---
        means = gmm.means_.flatten()
        high_comp = np.argmax(means)

        # --- 5. Апостериорные вероятности ---
        probs = gmm.predict_proba(data)[:, high_comp]

        # --- 6. Threshold через posterior ---
        # берем минимальное значение, где вероятность >= 0.5
        mask = probs >= posterior_thr

        if not np.any(mask):
            threshold = np.quantile(df_raw[score_col], 0.85)
        else:
            threshold = np.min(data[mask])

        # --- 7. Separability ---
        # используем нормированное расстояние между средними
        variances = gmm.covariances_.flatten()
        stds = np.sqrt(variances)

        sep = np.abs(means[1] - means[0]) / np.sqrt(stds[0]**2 + stds[1]**2)

        if verbose:
            print(f"run: p={p}, sep={sep:.3f}, converged={converged}, n_iter={n_iter_}")

        # --- 8. Проверка separability ---
        if sep >= separability_thr and threshold is not None:
            break
        else:
            if verbose:
                print(f"re-run: k2 ~ p{p/100:.2f}, sep={sep:.3f}, n_iter={n_iter_}")
            rerun += 1

    # --- 9. Бинаризация ---
    df_raw["binary"] = np.where(df_raw[score_col] > threshold, "SnC", "Normal")
    mask = df_raw[score_col] > threshold
    n_sen = np.sum(mask)
    n_total = np.sum(~np.isnan(df_raw[score_col]))  # если вдруг есть NaN
    sen_percentage = n_sen / n_total * 100 if n_total > 0 else 0
    if sen_percentage > 70:
        threshold = np.quantile(df_raw[score_col], 0.85)  # защита от слишком большого процента сенесцентных клеток
        sen_percentage = np.sum(df_raw[score_col] > threshold) / n_total * 100 if n_total > 0 else 0
        print(f"Adjusted threshold to {threshold:.4f} to limit senescent fraction to {sen_percentage:.2f}%")
    if verbose and threshold:
        print(f"\nthreshold: {threshold:.5f}")
        print(f"Sen cells: {n_sen}/{n_total} ({sen_percentage:.2f}%)")

    # ВИЗУАЛИЗАЦИЯ
    if verbose:
        order = np.argsort(gmm.means_.flatten())

        plot_gmm_results(
            df_raw=df_raw,
            data=data,
            gmm=gmm,
            threshold=threshold,
            high_comp=high_comp,
            order=order,
            score_col=score_col
        )

    if threshold is None:
        print("thr is None")
        return None
    return threshold


def binarization_wrap(df_raw, score_col='ds', sep_thr=3, binarize_kwargs=None):
    """
    - GMM (2 компоненты)
    - если separability > sep_thr → binarize_df_GMMk2(df_raw)
    - иначе → KDE crop → binarize_df_GMMk2(df_raw, df_crop)
    """
    if binarize_kwargs is None:
        binarize_kwargs = {}
    
    x = df_raw[score_col].values.astype(float)
    
    # 1. GMM с инициализацией через 25 и 75 перцентили - проверка на бимодальность
    q25, q75 = np.percentile(x, [25, 75])
    gmm = GaussianMixture(n_components=2,means_init=np.array([[q25], [q75]]),covariance_type='full',random_state=42)
    gmm.fit(x.reshape(-1, 1))

    # 2. Separability check
    mu1, mu2 = gmm.means_.flatten()
    var1, var2 = gmm.covariances_.flatten()
    sep = np.abs(mu1 - mu2) / np.sqrt(0.5 * (var1 + var2))
    print(f"[INFO] separability = {sep:.3f}")
    
    # --- CASE 1: хорошее разделение ---
    if sep > sep_thr:
        return binarize_df_GMMk2(df_raw, score_col=score_col, **binarize_kwargs)
    
    # --- CASE 2: плохое разделение → KDE → поиск максимума → фильтр ---
    mode, sigma = kde_mode_and_sigma(x)
    left_thr = mode - 0.8 * sigma
    print(f"[INFO] KDE max = {mode:.3f}, thr = {left_thr:.3f}")
    
    df_crop = df_raw[df_raw[score_col] >= left_thr].copy()
    
    return binarize_df_GMMk2(df_raw, df_crop=df_crop, score_col=score_col, **binarize_kwargs)


def binarize_df_GMMk3(df, score_col="ds", group_col="SnC", verbose=True):
    posterior_thr = 0.5  # фиксировано по условию

    # --- 1. Подготовка данных ---
    data = df[score_col].values.reshape(-1, 1)
    p99 = np.percentile(data, 99.9)
    data = np.clip(data, None, p99)

    # --- GMM параметры ---
    R = data.max() - data.min()
    max_sigma = 0.4 * R / 6
    init_var = max_sigma**2

    means_init = np.array([[np.percentile(data, 5)],
                           [np.percentile(data, 50)],
                           [np.percentile(data, 95)]])

    # --- 2. GMM (3 компоненты) ---
    gmm = GaussianMixture(
        n_components=3,
        means_init=means_init,
        precisions_init=np.array([[[2/init_var]],
                                [[2/init_var]],
                                [[3/init_var]]]),
        covariance_type="full",
        random_state=0
    )
    gmm.fit(data)

    converged = gmm.converged_
    n_iter_ = gmm.n_iter_

    # --- 3. Сортировка компонент по средним ---
    means = gmm.means_.flatten()
    order = np.argsort(means)

    means = means[order]
    weights = gmm.weights_[order]
    covariances = gmm.covariances_.reshape(-1)[order]

    # переупорядочим вероятности тоже
    probs_full = gmm.predict_proba(data)[:, order]

    # --- 4. Старшая компонента ---
    high_comp = 2  # после сортировки
    probs = probs_full[:, high_comp]

    # --- 5. Threshold ---
    mask = probs >= posterior_thr

    if not np.any(mask):
        threshold = None
    else:
        threshold = np.min(data[mask])

    if verbose:
        print(f"sep-like check skipped (k=3), converged={converged}, n_iter={n_iter_}")

    # --- 6. Бинаризация ---
    df["binary"] = np.where(df[score_col] > threshold, "SnC", "Normal")
    mask = df[score_col] > threshold
    n_sen = np.sum(mask)
    n_total = np.sum(~np.isnan(df[score_col]))
    sen_percent = n_sen / n_total * 100

    if verbose and threshold:
        print(f"\nthreshold: {threshold:.5f}")
        print(f"Sen cells: {n_sen}/{n_total} ({sen_percent:.2f}%)")
    if sen_percent > 50:
        threshold = None

    # ===== визуализация =====
    if verbose:
        plot_gmm_results(
            df_raw=df,
            data=data,
            gmm=gmm,
            threshold=threshold,
            high_comp=high_comp,
            order=order,
            score_col=score_col
        )

    if threshold is None:
        print("thr is None")
        return None

    return threshold


def count_plot_sen_corrs(obs_filt, score_col, count_means_corr=False, ready_df=False):
    """
    Если count_means_corr=True:
        Для каждой группы
        [Tissue_global, cell_type_final, GSE_id, GSM_id]
        считается среднее по score_col,
        затем строятся корреляции среднего значения с Age.
    Если False:
        Используется score_col как есть (count sen fractions).
    """
    if ready_df:
        donor_df = obs_filt.copy()
    if not ready_df:
        obs_filt = obs_filt.dropna(subset=[score_col]).copy()
        subgroup_cols = ['Tissue_global', "cell_type_final", "GSE_id", "GSM_id"]
        
        obs_filt["sen_fraction"] = (obs_filt.groupby(subgroup_cols, observed=True)[score_col].transform("mean"))

        donor_df = (
            obs_filt
            .groupby(["Tissue_global", "cell_type_final", "GSE_id", "GSM_id"], observed=True)
            .agg(sen_fraction=("sen_fraction", "first"),  # уже одинаково внутри группы
                Age=("Age", "first"),
                n_cells=("sen_fraction", "size") # size считает сколько всего строк в группе, можно по любому другому столбцу
            ).reset_index())

    stats_dict = {}
    for t in sorted(donor_df["Tissue_global"].unique()):
        df_t = donor_df[donor_df["Tissue_global"] == t]
        ct_groups = df_t.groupby("cell_type_final", observed=True)

        n_cols = 3
        n_rows = int(np.ceil(len(ct_groups) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 3*n_rows))
        axes = axes.flatten()

        # цвета для GSE
        tissue_gse = sorted(df_t["GSE_id"].unique())
        cmap = mpl.colormaps.get_cmap("tab20").resampled(len(tissue_gse))
        gse_color = dict(zip(tissue_gse, cmap.colors))

        for i, (ct, df_ct) in enumerate(ct_groups):
            ax = axes[i]

            # x = df_ct["Age"].values
            # y = df_ct["sen_fraction"].values
            # gse = df_ct["GSE_id"].values
            tmp = df_ct[["Age", "sen_fraction", "GSE_id", "n_cells"]].copy()
            tmp["Age"] = pd.to_numeric(tmp["Age"], errors="coerce")
            tmp["sen_fraction"] = pd.to_numeric(tmp["sen_fraction"], errors="coerce")
            tmp = tmp.dropna(subset=["Age", "sen_fraction"])
            x = tmp["Age"].to_numpy(dtype=float)
            y = tmp["sen_fraction"].to_numpy(dtype=float)
            gse = tmp["GSE_id"].to_numpy()
            sizes = tmp["n_cells"].to_numpy(dtype=float)

            colors = [gse_color[g] for g in gse]

            # scatter
            # sizes = df_ct["n_cells"].values # для размера точек на графике
            ax.scatter(x, y, c=colors, s=sizes/10, alpha=0.6)

            # корреляции
            if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0: # то есть не одинаковые 
                pear_r, pear_p = pearsonr(x, y)
                spear_r, spear_p = spearmanr(x, y)
            else:
                pear_r = pear_p = spear_r = spear_p = np.nan

            # регрессия
            if len(x) > 2:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                order = np.argsort(x)
                ax.plot(x[order], p(x[order]), alpha=0.7)
            # Заголовок
            ax.set_title(
                f"{ct}\n"
                f"Pearson r={pear_r:.2f} p={pear_p:.2g}\n"
                f"Spearman r={spear_r:.2f} p={spear_p:.2g}",
                fontsize=12
            )
            ax.set_xlabel("Age")
            if count_means_corr:
                ax.set_ylabel("Sen Score Means")
            else:
                ax.set_ylabel("Fraction")

            # Условие для обводки рамкой
            if pear_p < 0.05 and spear_p < 0.05:
                # Если оба p-value < 0.05, рамка оранжевая
                color = '#FF8C00'  # Оранжевый (закатное солнце)
            elif pear_p < 0.05 or spear_p < 0.05:
                # Если хотя бы одно p-value < 0.05, рамка желтая
                color = '#FFDD00'  # Желтый
            elif pear_p < 0.1 or spear_p < 0.1:
                color = 'grey'  # Желтый
            else:
                color = 'none'
            # Установим цвет рамки для осей
            for spine in ax.spines.values():
                spine.set_edgecolor(color)  # Устанавливаем цвет рамки
                if color != 'none':
                    spine.set_linewidth(2)  # default 1
            
            stats_dict[(t, ct)] = {"pearson_r": pear_r, "pearson_p": pear_p, "spearman_r": spear_r, "spearman_p": spear_p}

        # удалить лишние оси
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        # легенда
        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", color=gse_color[g], label=g)
            for g in tissue_gse]

        fig.legend(handles=handles, loc="lower right", title="GSE")
        fig.suptitle(t, fontsize=14)
        plt.tight_layout()
        plt.show()
    return stats_dict


def assign_bins(df):
    q25 = np.quantile(df["n_expressed_genes"], 0.25)
    q70 = np.quantile(df["n_expressed_genes"], 0.70)
    # q99 = np.quantile(df["n_expressed_genes"], 0.99)

    bins = []
    for x in df["n_expressed_genes"]:
        if x <= q25:
            bins.append("low")
        elif x > q25 and x <= q70:
            bins.append("med")
        elif x > q70:
            bins.append("hi")
    return pd.Series(bins, index=df.index)



