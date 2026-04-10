import gc
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from sklearn.decomposition import PCA
from os.path import join as pjoin
import scripts.config as config
from matplotlib.animation import FuncAnimation


# ================================================================================
# =============================== Helper Functions ===============================
# ================================================================================

def percentile_stretch(array, lower=2, upper=98):
    p_lower = np.percentile(array[~np.isnan(array)], lower)
    p_upper = np.percentile(array[~np.isnan(array)], upper)
    array_clipped = np.clip(array, p_lower, p_upper)
    return (array_clipped - p_lower) / (p_upper - p_lower)



def apply_rolling(data, rolling_window):
    if rolling_window is not None:
        return pd.Series(data).rolling(window=rolling_window, center=True).mean().values
    return data



def _compute_means(band_name, cluster_id, labels_monthly, bands_all, timestamps):
    """
    Compute the average reflectance values of a band for a cluster across time.
    """
    band = bands_all[band_name]
    band_monthly = band.resample(time=config.RESAMPLE_FREQ).median()
    means = []
    for t_idx, ts in enumerate(timestamps):
        cluster_mask = labels_monthly[t_idx] == cluster_id
        band_slice = np.flipud(band_monthly.sel(time=ts, method='nearest').values)
        cluster_pixels = band_slice[cluster_mask]
        valid_pixels = cluster_pixels[np.isfinite(cluster_pixels)]
        means.append(np.mean(valid_pixels) if len(valid_pixels) > 0 else np.nan)
    return means



def compute_cluster_pixel_counts(cluster_ids, labels_monthly, rolling_window=None):
    """
    Compute the total number of pixels assigned to specified clusters across time.
    """
    
    pixel_counts = {}
    
    for cluster_id in cluster_ids:
        counts = []
        for spatial_labels in labels_monthly:
            cluster_mask = spatial_labels == cluster_id
            pixel_count = np.sum(cluster_mask)
            counts.append(pixel_count)
        
        counts = np.array(counts, dtype=np.float32)
        
        # Apply rolling window smoothing if specified
        if rolling_window is not None:
            counts = apply_rolling(counts, rolling_window)
        
        pixel_counts[cluster_id] = counts
    
    return pixel_counts


def compute_cluster_pixel_counts(cluster_id, labels_monthly, rolling_window=None):
    """
    Compute the total number of pixels assigned to specified clusters across time.
    """
    
    pixel_counts = []
    for spatial_labels in labels_monthly:
        cluster_mask = spatial_labels == cluster_id
        pixel_count = np.sum(cluster_mask)
        pixel_counts.append(pixel_count)
    
    pixel_counts = np.array(pixel_counts, dtype=np.float32)
    
    return pixel_counts


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


# ================================================================================
# =========================== Visualization Pipelines ============================
# ================================================================================


def timeseries_gif(file_name, monthly_rgb, labels_monthly, timestamps, all_timestamps, from_year=None):
    """
    Create a gif of all monthly predictions next to the corresponding rgb image of that month.
    """
    print(" - Creating timeseries animation")
    if from_year is not None:
        timestamps_to_plot = [ts for ts in timestamps if pd.Timestamp(ts).year >= from_year]
    else:
        timestamps_to_plot = list(timestamps)

    fig, ax = plt.subplots(1, 2, figsize=(24, 10))

    first_ts = timestamps_to_plot[0]
    first_spatial_idx = all_timestamps.index(first_ts)

    first_rgb = monthly_rgb.sel(
        time=(monthly_rgb.time.dt.year == pd.Timestamp(first_ts).year) &
             (monthly_rgb.time.dt.month == pd.Timestamp(first_ts).month)
    ).squeeze().values.transpose(1, 2, 0)

    im_rgb = ax[0].imshow(np.flipud(percentile_stretch(first_rgb)), origin='upper')
    ax[0].set_title(pd.Timestamp(first_ts).strftime('%B %Y') + ' - RGB')

    cmap = plt.cm.get_cmap('tab10', config.N_CLUSTERS)
    cmap.set_bad(color='black')
    im_cls = ax[1].imshow(np.flipud(labels_monthly[first_spatial_idx]),
                            cmap=cmap, vmin=-0.5, vmax=config.N_CLUSTERS - 0.5,
                            interpolation='nearest', origin='upper')
    plt.colorbar(im_cls, ax=ax[1], label='Cluster ID', ticks=range(config.N_CLUSTERS))
    ax[1].set_title(pd.Timestamp(first_ts).strftime('%B %Y') + ' - Clusters')
    plt.tight_layout()

    def update(t_idx):
        ts = timestamps_to_plot[t_idx]
        spatial_idx = all_timestamps.index(ts)
        ts_str = pd.Timestamp(ts).strftime('%B %Y')

        rgb = monthly_rgb.sel(
            time=(monthly_rgb.time.dt.year == pd.Timestamp(ts).year) &
                 (monthly_rgb.time.dt.month == pd.Timestamp(ts).month)
        ).squeeze().values.transpose(1, 2, 0)

        im_rgb.set_data(np.flipud(percentile_stretch(rgb)))
        ax[0].set_title(f'{ts_str} - RGB')
        im_cls.set_data(np.flipud(labels_monthly[spatial_idx]))
        ax[1].set_title(f'{ts_str} - Clusters')

        return im_rgb, im_cls

    ani = FuncAnimation(fig, update, frames=len(timestamps_to_plot), interval=800, blit=True)
    path = pjoin(config.FIGURES_DIR, file_name)
    ani.save(path, writer='pillow', dpi=150)
    print(f"Saved comparison gif to {path}")
    plt.show()
    return ani



def plot_cluster_pixel_timeseries(file_name,
                                  model_type,
                                  cluster_ids,
                                  labels_monthly,
                                  timestamps,
                                  x_minmax=tuple(),
                                  rolling_window=None):
    """
    Plot the number of pixels assigned to specified clusters across time.
    """
    if isinstance(cluster_ids, int):
        cluster_ids = [cluster_ids]
        
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Add shaded region if x_minmax provided (useful for marking event periods)
    if x_minmax:
        xmin = float(mdates.date2num(pd.Timestamp(x_minmax[0])))
        xmax = float(mdates.date2num(pd.Timestamp(x_minmax[1])))
        ax.axvspan(xmin=xmin, xmax=xmax, color='red', alpha=0.1, zorder=1)
    
    # Plot each cluster's pixel counts
    for cluster_id in cluster_ids:
        pixel_count = compute_cluster_pixel_counts(cluster_id, labels_monthly, rolling_window)
        line, = ax.plot(timestamps, pixel_count, label=f'Cluster {cluster_id}', markersize=4, linewidth=1, alpha=0.3)
        if rolling_window is not None:
            ax.plot(timestamps, apply_rolling(pixel_count, rolling_window), label=f'Cluster {cluster_id}', marker='o', markersize=3, color=line.get_color())
        else:
            ax.plot(timestamps, pixel_count, label=f'Cluster {cluster_id}', marker='o', markersize=3, color=line.get_color())
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Number of Pixels', fontsize=11)
    title = f'{model_type.upper()} — Cluster Pixel Counts Over Time'
    if rolling_window:
        title += f' (rolling window: {rolling_window})'
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(pjoin(config.FIGURES_DIR, file_name), dpi=300, bbox_inches='tight')
    plt.show()



def save_monthly_comparisons(model_type, monthly_rgb, labels_monthly, timestamps, 
                              all_timestamps, output_dir, from_year=None):
    """Save side-by-side RGB vs cluster label plots for each month."""

    dir = pjoin(config.FIGURES_DIR, output_dir)

    os.makedirs(dir, exist_ok=True)
    if from_year is not None:
        timestamps_to_plot = [ts for ts in timestamps if pd.Timestamp(ts).year >= from_year]
    else:
        timestamps_to_plot = list(timestamps)

    cmap = plt.cm.get_cmap('tab10', config.N_CLUSTERS)
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

        fig, ax = plt.subplots(1, 2, figsize=(24, 10))

        ax[0].imshow(np.flipud(percentile_stretch(rgb)), origin='upper')
        ax[0].set_title(f'{ts_str} - RGB')

        im = ax[1].imshow(np.flipud(labels_monthly[spatial_idx]),
                            cmap=cmap, vmin=-0.5, vmax=config.N_CLUSTERS - 0.5,
                            interpolation='nearest', origin='upper')
        plt.colorbar(im, ax=ax[1], label='Cluster ID', ticks=range(config.N_CLUSTERS))
        ax[1].set_title(f'{ts_str} - Clusters')

        plt.tight_layout()
        plt.savefig(pjoin(dir, f'{model_type}_{ts_str.replace(" ", "_")}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)



def plot_cluster_means(file_name, model, feature_names, model_type=None):
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
    plt.savefig(pjoin(config.FIGURES_DIR, file_name), dpi=300, bbox_inches='tight')
    plt.show()
    return means_df



def plot_band_cluster_timeseries(file_name,
                                 model_type, 
                                 band_name, 
                                 cluster_ids, 
                                 labels_monthly,
                                 bands_all, 
                                 timestamps, 
                                 x_minmax=tuple(), 
                                 rolling_window=None):
    """
    Plots the average of one chosen feature (i.e. band) for all clusters across time.
    """

    fig, ax = plt.subplots(figsize=(14, 6))

    xmin = pd.Timestamp(x_minmax[0])
    xmax = pd.Timestamp(x_minmax[1])

    if x_minmax:
        xmin = float(mdates.date2num(pd.Timestamp(x_minmax[0])))
        xmax = float(mdates.date2num(pd.Timestamp(x_minmax[1])))
        ax.axvspan(xmin=xmin, xmax=xmax, color='red', alpha=0.1, zorder=1)

    for cluster_id in cluster_ids:
        means = _compute_means(band_name, cluster_id, labels_monthly, bands_all, timestamps)
        line, = ax.plot(timestamps, means, alpha=0.3, linewidth=1)
        if rolling_window is not None:
            ax.plot(timestamps, apply_rolling(means, rolling_window), label=f'Cluster {cluster_id}',
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
    
    plt.tight_layout()
    plt.savefig(pjoin(config.FIGURES_DIR, file_name), dpi=300, bbox_inches='tight')
    plt.show()



def plot_cluster_band_timeseries(file_name,
                                 model_type, 
                                 cluster_id, 
                                 index_bands_to_plot,
                                 labels_monthly,
                                 bands_all, 
                                 timestamps, 
                                 x_minmax=tuple(),
                                 raw_bands_to_plot=None,
                                 rolling_window=None):
    
    """
    Plots the average of multiple chosen features (i.e bands) for a specific cluster across time.
    """

    if raw_bands_to_plot is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        if x_minmax:
            xmin = float(mdates.date2num(pd.Timestamp(x_minmax[0])))
            xmax = float(mdates.date2num(pd.Timestamp(x_minmax[1])))
            ax2.axvspan(xmin=xmin, xmax=xmax, color='red', alpha=0.2, zorder=1)

        for band_name in raw_bands_to_plot:
            means = _compute_means(band_name, cluster_id, labels_monthly, bands_all, timestamps)
            ax1.plot(timestamps, means, alpha=0.3, linewidth=1, color=None)
            if rolling_window is not None:
                ax1.plot(timestamps, apply_rolling(means, rolling_window), label=band_name, marker='o', markersize=3)
            else:
                ax1.plot(timestamps, means, label=band_name, marker='o', markersize=3)

        for band_name in index_bands_to_plot:
            means = _compute_means(band_name, cluster_id, labels_monthly, bands_all, timestamps)
            ax2.plot(timestamps, means, alpha=0.3, linewidth=1)
            if rolling_window is not None:
                ax2.plot(timestamps, apply_rolling(means, rolling_window), label=band_name, marker='o', markersize=3, linestyle='--')
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

        xmin = pd.Timestamp(x_minmax[0])
        xmax = pd.Timestamp(x_minmax[1])

        if x_minmax:
            xmin = float(mdates.date2num(pd.Timestamp(x_minmax[0])))
            xmax = float(mdates.date2num(pd.Timestamp(x_minmax[1])))
            ax2.axvspan(xmin=xmin, xmax=xmax, color='red', alpha=0.2, zorder=1)

        for band_name in index_bands_to_plot:
            means = _compute_means(band_name, cluster_id, labels_monthly, bands_all, timestamps)
            ax2.plot(timestamps, means, alpha=0.3, linewidth=1)
            if rolling_window is not None:
                ax2.plot(timestamps, apply_rolling(means, rolling_window), label=band_name, marker='o', markersize=3, linestyle='--')
            else:
                ax2.plot(timestamps, means, label=band_name, marker='o', markersize=3, linestyle='--')

        ax2.set_ylabel('Index Value')
        ax2.set_xlabel('Date')
        ax2.set_title(f'{model_type.upper()} — Cluster {cluster_id} — Mean Index Values Over Time'
                      + (f' (rolling {rolling_window})' if rolling_window else ''))
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
        
    plt.savefig(pjoin(config.FIGURES_DIR, file_name), dpi=300, bbox_inches='tight')
    plt.show()