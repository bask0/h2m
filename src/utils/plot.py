
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import numpy as np
from scipy.stats import pearsonr, spearmanr

def default_plot(figwidth=6.96986):
    params = {
        # font
        'font.family': 'serif',
        # 'font.serif': 'Times', #'cmr10',
        'font.size': 7,
        # axes
        'axes.titlesize': 7,
        'axes.labelsize': 7,
        'axes.linewidth': 0.5,
        # ticks
        'xtick.labelsize': 5,
        'ytick.labelsize': 5,
        'xtick.major.width': 0.3,
        'ytick.major.width': 0.3,
        'xtick.minor.width': 0.3,
        'ytick.minor.width': 0.3,
        # legend
        'legend.fontsize': 7,
        'figure.figsize': [figwidth, figwidth/1.6],
        # tex
        'text.usetex': True,
        # layout
        #'constrained_layout': True
        'legend.fontsize': 5,   
        'legend.title_fontsize': 5
    }

    mpl.rcParams.update(params)

def savefig(fig, path, transparent=True):
    fig.savefig(path, dpi=300, transparent=transparent)

def fix_fig(fig, figwidth=['small', 'large'][0], ratio=1.6):
    if figwidth == 'small':
        figwidth = 3.26772
    elif figwidth == 'large':
        figwidth = 4.72441
    else:
        figwidth = figwidth
    fig.set_size_inches(figwidth, figwidth / ratio)

def fix_ax(ax, remove=['top', 'right'], color=None):
    if color is not None:
        for pos in ['bottom', 'top', 'left', 'right']:
            ax.spines[pos].set_color(color)
        ax.tick_params(axis='both', colors=color)
        ax.yaxis.label.set_color('k')
        ax.xaxis.label.set_color('k')

    for pos in remove:
        ax.spines[pos].set_visible(False)

    return ax

def strip_axis(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.patch.set_facecolor('None')
    ax.set_yticks([], minor=[])
    ax.set_xticks([], minor=[])
    ax.set_xlabel('')

    return ax

def remove_axes(ax, *locs):
    for loc in locs:
        ax.spines[loc].set_visible(False)
        ax.tick_params(**{loc: False})

        if loc == 'bottom':
            ax.set_xlabel('')
        elif loc == 'left':
            ax.set_ylabel('')

    return ax

def subplots_with_legend(nrows, ncols, legend_axis_ratio=0.3, figsize=(15, 6), projection=None, cbaxes_per_column=False):

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(nrows + 1, ncols, figure=fig,
        height_ratios=[1.0] * nrows + [legend_axis_ratio])
    axes = []
    for c in range(nrows):
        columns = []
        for r in range(ncols):
            columns.append(fig.add_subplot(gs[c, r], projection=projection))
        axes.append(columns)

    axes = np.array(axes)

    if not cbaxes_per_column:
        leg_ax = fig.add_subplot(gs[-1, :])
        strip_axis(leg_ax)
    else:
        leg_axes = []
        for c in range(ncols):
            leg_axes.append(fig.add_subplot(gs[c, :]))
        leg_ax = np.array(leg_axes)

    return fig, axes, leg_ax

def beautify_axis(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.patch.set_facecolor('None')
    ax.set_xlabel('')

    return ax

def subplots_PlateCarree(*args, **kwargs):
    return plt.subplots(*args, subplot_kw={'projection': cartopy.crs.PlateCarree()}, **kwargs)

def get_cartopy_axis(ax, extent='terr', linewidth=0.1, gridlinewidth=0.1, land_color='0.2',
                    labels_left=True, labels_right=False, labels_bottom=True, labels_top=False, outline=False):

    ax.add_feature(cartopy.feature.COASTLINE, linewidth=linewidth)
    ax.add_feature(cartopy.feature.LAND, facecolor=land_color)
    # ax.outline_patch.set_edgecolor('none')
    ax.background_patch.set_visible(False)
    try:
        gl = ax.gridlines(
            draw_labels=True, xlocs=[-240, -120, -60, 0, 60, 120], ylocs=[-60, -30, 0, 30, 60, 90],
            color='0.5', linewidth=gridlinewidth)
    except:
        gl = ax.gridlines(
            draw_labels=False, xlocs=[-240, -120, -60, 0, 60, 120], ylocs=[-60, -30, 0, 30, 60, 90],
            color='0.5', linewidth=gridlinewidth)
    if not labels_left:
        gl.left_labels = False
    if not labels_right:
        gl.right_labels = False
    if not labels_bottom:
        gl.bottom_labels = False
    if not labels_top:
        gl.top_labels = False

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    if extent == 'terr':
        ax.set_extent([-180, 180, -58, 82], crs=cartopy.crs.PlateCarree())
    elif hasattr(extent, '__iter__'):
        ax.set_extent(extent, crs=cartopy.crs.PlateCarree())

    if not outline:
        ax.outline_patch.set_edgecolor('none')

    return ax

def map_plot(ds, ax=None, title=None, time=None, label=None, cbar_kwargs={}, cartopy_kwargs={}, imshow=True, contourplot=False, **kwargs):

    if ax is None:
        fig, ax = subplots_PlateCarree(figsize=(12, 6))

    if label is not None:
        ds.attrs['units'] = ''
        ds.attrs['long_name'] = label

    ax = get_cartopy_axis(ax, **cartopy_kwargs)
    #cbar_kwargs.update({'rasterize': True})

    if imshow:
        img = ds.plot.imshow(ax=ax, rasterized=False, cbar_kwargs=cbar_kwargs, **kwargs)
    else:
        if contourplot:
            img = ds.plot.contourf(ax=ax, rasterized=False, cbar_kwargs=cbar_kwargs, **kwargs)
        else:
            img = ds.plot.pcolormesh(ax=ax, rasterized=False, cbar_kwargs=cbar_kwargs, **kwargs)
    #img.colorbar.solids.set_rasterized(False)
    #cbaxes = ax.inset_axes([0.3, 0.08, 0.7, 0.03])
    #strip_axis(cbaxes)
    #cb = plt.colorbar(img, ax=cbaxes, orientation='horizontal', fraction=1, aspect=40, pad=0, **cbar_kwargs)
    #cb.solids.set_rasterized(False)
    #ctitle = ds.attrs.get('long_name')
    #if ds.attrs.get('units', '') != '':    
    #    ctitle += f' ({ds.attrs["units"]})'
    #cbaxes.set_title(ctitle)
     
    #plt.colorbar(cax=cbaxes, ticks=[0.,1], orientation='horizontal')
    ax.set_title('')

    return img

def plot_hexbin(
        x, y,
        xlabel='mod', ylabel='obs',
        gridsize=80,
        mincnt=1,
        title='',
        ax=None,
        metrics=['pearsonr', 'spearmanr', 'nse'],
        text_pos=(0.05, 0.95),
        text_v_alignment='top',
        text_h_alignment='left',
        extent=None,
        subplot_kw={},
        colorbar=False,
        bins='log',
        **kwargs):

    if ax is None:
        fig, ax = plt.subplots(**subplot_kw)

    not_missing = x.notnull() & y.notnull()

    x_data = x.values[not_missing.data]
    y_data = y.values[not_missing.data]

    m_str = []
    for m in metrics:
        if m == 'pearsonr':
            m_name = 'r_p'
            v = pearsonr(x_data, y_data)[0]
        elif m == 'spearmanr':
            m_name = 'r_s'
            v = spearmanr(x_data, y_data)[0]
        elif m == 'nse':
            m_name = 'NSE'
            v = (1 - np.sum(np.power(x_data - y_data, 2)) / np.sum(np.power(y_data - np.mean(y_data), 2)))
        else:
            raise ValueError(
                f'Metric `{m}` not implemented, chose from `pearsonr`, `spearmanr`, `mef`)')

        m_str += [f'${m_name}: {v:.2f}$']
    m_str = '\n'.join(m_str)

    hb = ax.hexbin(x_data, y_data, bins=bins, gridsize=gridsize, mincnt=mincnt, extent=extent, **kwargs)

    if colorbar:
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label(r'log_10(N)')
        # cb.remove()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.text(text_pos[0], text_pos[1], f'{m_str}', horizontalalignment=text_h_alignment,
            verticalalignment=text_v_alignment, transform=ax.transAxes)

    return hb


def add_hist(ds, ax, cmap, vmin, vmax, quantile=0.01, histogram_placement=[0.04, 0.15, 0.23, 0.3], add_median_text=True, bins=16, add_contour=True, contour_lw=0.2, **kwargs):

    axh = ax.inset_axes(histogram_placement)

    #cs = global_cell_size(n_lat=len(ds.lat), n_lon=len(ds.lon), normalize=True).values

    ds.name = 'metric'

    cmap = plt.get_cmap(cmap)
    #def get_color(c):
    #    return cmap((c - vmin) / (vmax - vmin))

    #median = weighted_median(ds)
    #median = np.round(median, 1 if median / 10 > 1 else 2)
    
#     xmin_, xmax_ = np.nanquantile(ds, [quantile, 1.0-quantile])

#     if xmin is None:
#         xmin = xmin_
#     if xmax is None:
#         xmax = xmax_

    # s.plot.hist(ax=axh, bins=40, color='0.5', range=(xmin, xmax))
    #ds = ds.values
    #mask = np.isfinite(ds)
    #ds = ds[mask]
    #cs = cs[mask]
    
    n, bns, patches = ds.plot.hist(bins=np.linspace(vmin, vmax, bins), ax=axh, **kwargs)
    if add_contour:
        ds.plot.hist(histtype='step', color='k', bins=bns, lw=contour_lw, ax=axh)

    bin_centers = 0.5 * (bns[:-1] + bns[1:])

    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cmap(c))

#     axh.hist(ds, weights=cs, bins=bins, color='0.5', range=(xmin, xmax))

    axh.set_title('')
    axh.spines['right'].set_visible(False)
    axh.spines['left'].set_visible(False)
    axh.spines['top'].set_visible(False)
    axh.set_yticks([])
    axh.patch.set_facecolor('None')
    axh.set_xlabel('')

    #min_tick = (vmax - vmin) / (bins - 1) / 2
    #max_tick = vmax - min_tick

    axh.set_xticks(np.linspace(bin_centers[0], bin_centers[-1], 3))
    labs = np.linspace(vmin, vmax, 3)
    labs_new = []
    for l in labs:
        if l % 1 == 0:
            labs_new.append(int(l))
        else:
            labs_new.append(l)
    axh.set_xticklabels(labs_new)
    axh.xaxis.set_tick_params(labelsize=5, pad=0.1)
