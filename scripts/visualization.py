import gc
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from os.path import join as pjoin
import scripts.config as config
from matplotlib.animation import FuncAnimation

def percentile_stretch(array, lower=2, upper=98):
    p_lower = np.percentile(array[~np.isnan(array)], lower)
    p_upper = np.percentile(array[~np.isnan(array)], upper)
    array_clipped = np.clip(array, p_lower, p_upper)
    return (array_clipped - p_lower) / (p_upper - p_lower)

def _compute_means(band_name, cluster_id, spatial_labels_monthly, bands_all, timestamps):
    band = bands_all[band_name]
    band_monthly = band.resample(time=config.RESAMPLE_FREQ).median()
    means = []
    for t_idx, ts in enumerate(timestamps):
        cluster_mask = spatial_labels_monthly[t_idx] == cluster_id
        band_slice = np.flipud(band_monthly.sel(time=ts, method='nearest').values)
        cluster_pixels = band_slice[cluster_mask]
        valid_pixels = cluster_pixels[np.isfinite(cluster_pixels)]
        means.append(np.mean(valid_pixels) if len(valid_pixels) > 0 else np.nan)
    return means

def load_monthly_rgb(stack):
    monthly_rgb = stack.sel(band=[6, 4, 2]).resample(time='MS').mean()
    del stack
    gc.collect()
    return monthly_rgb


def get_timestamps_for_years(bands_all, years):
    ref_band = bands_all[list(bands_all.keys())[0]]
    return ref_band.sel(time=ref_band.time.dt.year.isin(years))\
                   .resample(time=config.RESAMPLE_FREQ)\
                   .median()\
                   .time.values


def get_all_monthly_timestamps(bands_all):
    ref_band = bands_all[list(bands_all.keys())[0]]
    return list(ref_band.resample(time=config.RESAMPLE_FREQ).median().time.values)


def timeseries_gif(monthly_rgb, spatial_labels_monthly, timestamps, 
                       all_timestamps, suffix, from_year=2023):
    print(" - Creating timeseries animation")
    post_timestamps = [ts for ts in timestamps if pd.Timestamp(ts).year >= from_year]

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    first_ts = post_timestamps[0]
    first_spatial_idx = all_timestamps.index(first_ts)

    first_rgb = monthly_rgb.sel(
        time=(monthly_rgb.time.dt.year == pd.Timestamp(first_ts).year) &
             (monthly_rgb.time.dt.month == pd.Timestamp(first_ts).month)
    ).squeeze().values.transpose(1, 2, 0)

    im_rgb = axes[0].imshow(np.flipud(percentile_stretch(first_rgb)), origin='upper')
    axes[0].set_title(pd.Timestamp(first_ts).strftime('%B %Y') + ' - RGB')

    cmap = plt.cm.get_cmap('tab10', config.GMM_COMPONENTS)
    cmap.set_bad(color='black')
    im_cls = axes[1].imshow(np.flipud(spatial_labels_monthly[first_spatial_idx]),
                            cmap=cmap, vmin=-0.5, vmax=config.GMM_COMPONENTS - 0.5,
                            interpolation='nearest', origin='upper')
    plt.colorbar(im_cls, ax=axes[1], label='Cluster ID', ticks=range(config.GMM_COMPONENTS))
    axes[1].set_title(pd.Timestamp(first_ts).strftime('%B %Y') + ' - Clusters')
    plt.tight_layout()

    def update(t_idx):
        ts = post_timestamps[t_idx]
        spatial_idx = all_timestamps.index(ts)
        ts_str = pd.Timestamp(ts).strftime('%B %Y')

        rgb = monthly_rgb.sel(
            time=(monthly_rgb.time.dt.year == pd.Timestamp(ts).year) &
                 (monthly_rgb.time.dt.month == pd.Timestamp(ts).month)
        ).squeeze().values.transpose(1, 2, 0)

        im_rgb.set_data(np.flipud(percentile_stretch(rgb)))
        axes[0].set_title(f'{ts_str} - RGB')
        im_cls.set_data(np.flipud(spatial_labels_monthly[spatial_idx]))
        axes[1].set_title(f'{ts_str} - Clusters')

        return im_rgb, im_cls

    ani = FuncAnimation(fig, update, frames=len(post_timestamps), interval=800, blit=True)
    path = pjoin(config.FIGURES_DIR, f'timeseries_comparison_{config.RUN_NAME}{suffix}.gif')
    ani.save(path, writer='pillow', dpi=150)
    print(f"Saved comparison gif to {path}")
    plt.show()
    return ani

def save_monthly_comparisons(model_type, monthly_rgb, labels_monthly, timestamps, 
                              all_timestamps, output_dir, from_year=None):
    """Save side-by-side RGB vs cluster label plots for each month."""
    os.makedirs(output_dir, exist_ok=True)
    if from_year is not None:
        timestamps_to_plot = [ts for ts in timestamps if pd.Timestamp(ts).year >= from_year]
    else:
        timestamps_to_plot = list(timestamps)

    cmap = plt.cm.get_cmap('tab10', config.GMM_COMPONENTS)
    cmap.set_bad(color='black')

    for ts in tqdm(timestamps_to_plot, desc="Saving monthly comparisons"):
        spatial_idx = all_timestamps.index(ts)
        year  = pd.Timestamp(ts).year
        month = pd.Timestamp(ts).month
        ts_str = pd.Timestamp(ts).strftime('%Y %m')

        rgb = monthly_rgb.sel(
            time=(monthly_rgb.time.dt.year == year) &
                 (monthly_rgb.time.dt.month == month)
        ).squeeze().values.transpose(1, 2, 0)

        fig, axes = plt.subplots(1, 2, figsize=(24, 10))

        axes[0].imshow(np.flipud(percentile_stretch(rgb)), origin='upper')
        axes[0].set_title(f'{ts_str} - RGB')

        im = axes[1].imshow(np.flipud(labels_monthly[spatial_idx]),
                            cmap=cmap, vmin=-0.5, vmax=config.GMM_COMPONENTS - 0.5,
                            interpolation='nearest', origin='upper')
        plt.colorbar(im, ax=axes[1], label='Cluster ID', ticks=range(config.GMM_COMPONENTS))
        axes[1].set_title(f'{ts_str} - Clusters')

        plt.tight_layout()
        plt.savefig(pjoin(output_dir, f'{model_type}_{ts_str.replace(" ", "_")}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_cluster_means(model, feature_names, model_type=None):
    """Plot cluster means as a heatmap to inspect feature contributions."""
    if model_type == 'gmm':
        means = model.means_
    elif model_type == 'kmeans':
        means = model.cluster_centers_
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Expected 'gmm' or 'kmeans'.")

    means_df = pd.DataFrame(means, columns=feature_names)
    means_df.index.name = 'Cluster'

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(means_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax, linewidths=0.5)
    ax.set_title(f'{model_type.upper()} Cluster Means per Feature (standardized)')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Cluster')
    plt.tight_layout()
    plt.savefig(pjoin(config.FIGURES_DIR, f'cluster_means_{model_type}_{config.RUN_NAME}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    return means_df


def plot_cluster_band_timeseries(model_type, cluster_id, index_bands_to_plot,
                                  spatial_labels_monthly, bands_all, timestamps, 
                                  raw_bands_to_plot=None, rolling_window=None):
    
    def apply_rolling(means):
        if rolling_window is not None:
            return pd.Series(means).rolling(window=rolling_window, center=True).mean().values
        return means

    if raw_bands_to_plot is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        for band_name in raw_bands_to_plot:
            means = _compute_means(band_name, cluster_id, spatial_labels_monthly, bands_all, timestamps)
            ax1.plot(timestamps, means, alpha=0.3, linewidth=1, color=None)
            if rolling_window is not None:
                ax1.plot(timestamps, apply_rolling(means), label=band_name, marker='o', markersize=3)
            else:
                ax1.plot(timestamps, means, label=band_name, marker='o', markersize=3)

        for band_name in index_bands_to_plot:
            means = _compute_means(band_name, cluster_id, spatial_labels_monthly, bands_all, timestamps)
            ax2.plot(timestamps, means, alpha=0.3, linewidth=1)
            if rolling_window is not None:
                ax2.plot(timestamps, apply_rolling(means), label=band_name, marker='o', markersize=3, linestyle='--')
            else:
                ax2.plot(timestamps, means, label=band_name, marker='o', markersize=3, linestyle='--')

        ax1.set_ylabel('Mean Reflectance (raw bands)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax2.set_ylabel('Index Value')
        ax2.set_xlabel('Date')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        fig.suptitle(f'{model_type.upper()} — Cluster {cluster_id} — Mean Band Values Over Time'
                     + (f' (rolling {rolling_window})' if rolling_window else ''))

    else:
        fig, ax2 = plt.subplots(figsize=(14, 6))

        for band_name in index_bands_to_plot:
            means = _compute_means(band_name, cluster_id, spatial_labels_monthly, bands_all, timestamps)
            ax2.plot(timestamps, means, alpha=0.3, linewidth=1)
            if rolling_window is not None:
                ax2.plot(timestamps, apply_rolling(means), label=band_name, marker='o', markersize=3, linestyle='--')
            else:
                ax2.plot(timestamps, means, label=band_name, marker='o', markersize=3, linestyle='--')

        ax2.set_ylabel('Index Value')
        ax2.set_xlabel('Date')
        ax2.set_title(f'{model_type.upper()} — Cluster {cluster_id} — Mean Index Values Over Time'
                      + (f' (rolling {rolling_window})' if rolling_window else ''))
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if rolling_window is not False:
        png_name = f'{model_type}_cluster_{cluster_id}_RA_timeseries_{config.RUN_NAME}.png'
    else:
        png_name = f'{model_type}_cluster_{cluster_id}_timeseries_{config.RUN_NAME}.png'
        
    plt.savefig(pjoin(config.FIGURES_DIR, png_name), dpi=300, bbox_inches='tight')
    plt.show()

def plot_band_cluster_timeseries(model_type, band_name, cluster_ids, spatial_labels_monthly,
                                  bands_all, timestamps, rolling_window=None):

    def apply_rolling(means):
        if rolling_window is not None:
            return pd.Series(means).rolling(window=rolling_window, center=True).mean().values
        return means

    fig, ax = plt.subplots(figsize=(14, 6))

    for cluster_id in cluster_ids:
        means = _compute_means(band_name, cluster_id, spatial_labels_monthly, bands_all, timestamps)
        line, = ax.plot(timestamps, means, alpha=0.3, linewidth=1)
        if rolling_window is not None:
            ax.plot(timestamps, apply_rolling(means), label=f'Cluster {cluster_id}',
                    marker='o', markersize=3, color=line.get_color())
        else:
            ax.plot(timestamps, means, label=f'Cluster {cluster_id}',
                    marker='o', markersize=3, color=line.get_color())

    ax.set_ylabel('Mean Reflectance')
    ax.set_xlabel('Date')
    ax.set_title(f'{model_type.upper()} — {band_name} — Mean Reflectance per Cluster Over Time'
                 + (f' (rolling {rolling_window})' if rolling_window else ''))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    if rolling_window is not None:
        png_name = f'{model_type}_band_{band_name}_RA_timeseries_{config.RUN_NAME}.png'
    else:
        png_name = f'{model_type}_band_{band_name}_timeseries_{config.RUN_NAME}.png'

    plt.tight_layout()
    plt.savefig(pjoin(config.FIGURES_DIR, png_name), dpi=300, bbox_inches='tight')
    plt.show()


# def plot_pca_clusters(data, labels, gmm):
#     print("Reducing to 2D using PCA...")
#     pca = PCA(n_components=2)
#     pca_result = pca.fit_transform(data)

#     fig, ax = plt.subplots(figsize=(12, 9))
#     colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, config.GMM_COMPONENTS))

#     for i in range(config.GMM_COMPONENTS):
#         mask = labels == i
#         ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
#                    c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=30, edgecolors='none')

#     centers = pca.transform(np.asarray(gmm.means_))
#     ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='*',
#                s=500, edgecolors='black', linewidth=2, label='Centers', zorder=10)

#     ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
#     ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
#     ax.set_title(f'GMM Clustering ({config.GMM_COMPONENTS} components)')
#     ax.legend(); ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(pjoin(config.FIGURES_DIR, f'gmm_clusters_pca_{config.RUN_NAME}.png'), dpi=300, bbox_inches='tight')
#     plt.show()
#     return pca_result


# def plot_spatial_map(spatial, name):
#     """Plot a spatial map (already processed, flipped array)."""
#     cmap = plt.cm.get_cmap('tab10', config.GMM_COMPONENTS)
 
#     fig, ax = plt.subplots(figsize=(14, 10))
#     im = ax.imshow(spatial, cmap=cmap, vmin=-0.5, vmax=config.GMM_COMPONENTS-0.5,
#                    interpolation='nearest', origin='upper')
#     plt.colorbar(im, ax=ax, label='Cluster ID', ticks=range(config.GMM_COMPONENTS))
#     ax.set_title(f'2022 Classification ({config.GMM_COMPONENTS} classes)')
#     plt.tight_layout()
#     plt.savefig(pjoin(config.FIGURES_DIR, f'{name}.png'), dpi=300, bbox_inches='tight')
#     plt.show()