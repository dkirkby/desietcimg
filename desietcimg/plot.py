"""Plotting utilities. Import requires matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches


def draw_ellipse_cov(ax, center, cov, nsigmas=2, **ellipseopts):
    U, s, _ = np.linalg.svd(cov)
    angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    width, height = 2 * nsigmas * np.sqrt(s.T)
    kwargs = dict(color='w', ls='-', lw=2, fill=False)
    kwargs.update(ellipseopts)
    ellipse = matplotlib.patches.Ellipse(center, width, height, angle, **kwargs)
    ax.add_artist(ellipse)


def plot_image(D, W, cov=None, center=[0, 0], ax=None):
    data = D.copy()
    data[W == 0] = np.nan
    ax = ax or plt.gca()
    I = ax.imshow(data, interpolation='none', origin='lower')
    #plt.colorbar(I, ax=ax)
    if cov is not None:
        ny, nx = data.shape
        center = np.asarray(center) + 0.5 * (np.array(data.shape) - 1)
        draw_ellipse_cov(ax, center, cov)
    ax.axis('off')


class Axes(object):
    
    def __init__(self, n, size=4, pad=0.02):
        rows = int(np.floor(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        assert rows * cols >= n
        width = cols * size + (cols - 1) * pad
        height = rows * size + (rows - 1) * pad
        self.fig, axes = plt.subplots(rows, cols, figsize=(width, height))
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=pad, hspace=pad)
        self.axes = axes.flatten()
        self.n = n
        for ax in self.axes:
            ax.axis('off')
