import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from os.path import join as pjoin
import scripts.config as config


def plot_pca_clusters(data, labels, gmm):
    print("Reducing to 2D using PCA...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)

    fig, ax = plt.subplots(figsize=(12, 9))
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, config.GMM_COMPONENTS))

    for i in range(config.GMM_COMPONENTS):
        mask = labels == i
        ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                   c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=30, edgecolors='none')

    centers = pca.transform(np.asarray(gmm.means_))
    ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='*',
               s=500, edgecolors='black', linewidth=2, label='Centers', zorder=10)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_title(f'GMM Clustering ({config.GMM_COMPONENTS} components)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(pjoin(config.GRAPHS_DIR, f'gmm_clusters_pca_{config.RUN_NAME}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    return pca_result


def plot_spatial_map(spatial, name):
    """Plot a spatial map (already processed, flipped array)."""
    cmap = plt.cm.get_cmap('tab10', config.GMM_COMPONENTS)
 
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(spatial, cmap=cmap, vmin=-0.5, vmax=config.GMM_COMPONENTS-0.5,
                   interpolation='nearest', origin='upper')
    plt.colorbar(im, ax=ax, label='Cluster ID', ticks=range(config.GMM_COMPONENTS))
    ax.set_title(f'2022 Classification ({config.GMM_COMPONENTS} classes)')
    plt.tight_layout()
    plt.savefig(pjoin(config.GRAPHS_DIR, f'{name}.png'), dpi=300, bbox_inches='tight')
    plt.show()