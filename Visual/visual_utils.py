import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colorbar, ticker
import matplotlib
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.collections import LineCollection

def align_yaxis(ax1, ax2):
    y_lims = np.array([ax.get_ylim() for ax in [ax1, ax2]])

    # force 0 to appear on both axes, comment if don't need
    y_lims[:, 0] = y_lims[:, 0].clip(None, 0)
    y_lims[:, 1] = y_lims[:, 1].clip(0, None)

    # normalize both axes
    y_mags = (y_lims[:,1] - y_lims[:,0]).reshape(len(y_lims),1)
    y_lims_normalized = y_lims / y_mags

    # find combined range
    y_new_lims_normalized = np.array([np.min(y_lims_normalized), np.max(y_lims_normalized)])

    # denormalize combined range to get new axes
    new_lim1, new_lim2 = y_new_lims_normalized * y_mags
    ax1.set_ylim(new_lim1)
    ax2.set_ylim(new_lim2)

def plot_axes(ax, fig=None, geometry=(1,1,1)):
    if fig is None:
        fig = plt.figure()
    if ax.get_geometry() != geometry :
        ax.change_geometry(*geometry)
    ax = fig.axes.append(ax)
    return fig

def set_legend(ax, title='', ncol=1, loc=0):
    import matplotlib as mpl
    myfont1 = mpl.font_manager.FontProperties(fname='/System/Library/Fonts/Supplemental/Arial Italic.ttf')
    myfont2 = mpl.font_manager.FontProperties(fname='/System/Library/Fonts/Supplemental/Arial Bold.ttf')
    lg = ax.legend(fontsize="x-large", title=title, title_fontsize='xx-large', ncol=ncol, loc=loc)
    title = lg.get_title()
    for lh in lg.legendHandles:
        lh.set_alpha(1)
    title.set_fontsize('xx-large')
    title.set_weight('bold')
    # l.set_title(title=title, prop=myfont2)


# not perfect
def create_discrete_colorbar(fig, ax, array_of_values, fontdict_cbar_label=None, fontdict_cbar_ticks=None):
    fontdict_cbar_label_standard = {'label': None, 'fontsize': 35, 'fontweight': 'bold'}
    if fontdict_cbar_label is not None: fontdict_cbar_label_standard.update(fontdict_cbar_label)
    fontdict_cbar_ticks_standard = {'axis': 'y', 'labelsize': 25}
    if fontdict_cbar_ticks is not None: fontdict_cbar_ticks_standard.update(fontdict_cbar_ticks)

    import matplotlib as mpl
    import matplotlib.ticker as ticker
    #import matplotlib.pylab as plt
    cmap = plt.cm.jet  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    #cmaplist[0] = (.5, .5, .5, 1.0)

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    #array_of_values = array_of_values
    bounds = np.linspace(np.min(array_of_values), np.max(array_of_values),
                         num=(np.max(array_of_values)-np.min(array_of_values))+1)
    print(bounds)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # create a second axes for the colorbar
    ax2 = make_axes_locatable(ax).append_axes(position='right', size='3%', pad=0.1)
    cbar_edg = colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional',
                                    ticks=bounds+np.ones_like(bounds)*.5, boundaries=bounds, format='%1i')
    #color_edges = mapp  # list(nx.get_edge_attributes(g, 'weight').values())
    #vmin_edges = np.min(color_edges)
    #vmax_edges = np.max(color_edges)
    #cbar_edg_ticks = np.linspace(vmin_edges, vmax_edges, 3, dtype=int)
    #cbar_edg.set_ticks(cbar_edg_ticks)
    #cbar_edg_ticks_label = ['%d' % i for i in cbar_edg_ticks]
    #cbar_edg.ax.set_yticklabels(cbar_edg_ticks_label)
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(len(np.unique(array_of_values))))
    cbar_edg.ax.tick_params(**fontdict_cbar_ticks_standard)
    ax2.yaxis.set_tick_params(labelsize=fontdict_cbar_ticks_standard['labelsize'])
    #cbar.ax.set_tick_params(labelsize=fontdict_cbar_ticks_standard['labelsize'])
    ax2.set_ylabel(fontdict_cbar_label_standard['label'], fontdict=fontdict_cbar_label_standard)
    return fig, ax, cbar_edg, cmap


def create_colorbar(fig, ax, mapp, array_of_values, valfmt="{x:.2f}", fontdict_cbar_label=None,
                    fontdict_cbar_tickslabel=None, fontdict_cbar_ticks=None):
    fontdict_cbar_label_standard = {'label': None, 'fontsize': 25, 'fontweight': 'bold'}
    if fontdict_cbar_label is not None: fontdict_cbar_label_standard.update(fontdict_cbar_label)

    fontdict_cbar_tickslabel_standard = {'fontsize': 20, 'fontweight': 'normal'}
    if fontdict_cbar_tickslabel is not None: fontdict_cbar_tickslabel_standard.update(fontdict_cbar_tickslabel)

    fontdict_cbar_ticks_standard = {'axis':'y', 'labelsize': 15}
    if fontdict_cbar_ticks is not None: fontdict_cbar_ticks_standard.update(fontdict_cbar_ticks)

    # Colorbar
    # specify the fraction of the plot area that will be used to draw the colorbar
    cbar_ax = make_axes_locatable(ax).append_axes(position='right', size='3%', pad=0.1)
    # Create colorbar
    cbar_edg = fig.colorbar(mappable=mapp, cax=cbar_ax)
    # cbar= fig.colorbar(lc, cax=cbar_ax)
    cbar_edg.ax.set_ylabel(ylabel=fontdict_cbar_label_standard['label'], fontdict=fontdict_cbar_label_standard)
    cbar_edg.ax.tick_params(**fontdict_cbar_ticks_standard)
    color_edges = array_of_values#list(nx.get_edge_attributes(g, 'weight').values())
    vmin_edges = np.min(color_edges)
    vmax_edges = np.max(color_edges)
    cbar_edg_ticks = np.linspace(vmin_edges, vmax_edges, 3)
    cbar_edg.set_ticks(cbar_edg_ticks)
    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)
    cbar_edg_ticks_label = [valfmt(i, None) for i in cbar_edg_ticks]#['%.3f' % i for i in cbar_edg_ticks]
    cbar_edg.ax.set_yticklabels(cbar_edg_ticks_label, **fontdict_cbar_tickslabel_standard)
    return fig, ax, cbar_edg

def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc

def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """
    import matplotlib
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def set_ticks_label(ax, ax_type, data, num=5, valfmt="{x:.2f}", ticks=None, only_ticks=False, tick_lab=None,
                        fontdict_ticks_label={'weight':'bold', 'size': 'x-large'},
                    ax_label='', fontdict_label={'weight': 'bold', 'size': 'xx-large', 'color': 'black'}, scale=None, add_ticks=[]):

    fontdict_ticks_label_standard = {'weight':'bold', 'size': 'x-large' }
    if fontdict_ticks_label is not None:
        fontdict_ticks_label_standard.update(fontdict_ticks_label)

    fontdict_label_standard = {'weight': 'bold', 'size': 'xx-large', 'color': 'black'}
    if fontdict_label is not None:
        fontdict_label_standard.update(fontdict_label)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    #if scale == 'log':
        # ticks = np.logspace(start=min(data), stop=, num=num, endpoint=True)
    if scale == 'log':
         ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    if ticks is None:
        ticks = np.concatenate((np.linspace(start=np.min(data), stop=np.max(data), num=num, endpoint=True), add_ticks))
    if only_ticks:
        tick_lab = [''] * len(ticks)
    else:
        if tick_lab is None:
            tick_lab = [valfmt(i) for i in ticks]
        else:
            tick_lab = [valfmt(i) for i in tick_lab]

    if ax_type == 'x':
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_lab, fontdict=fontdict_ticks_label_standard)
        ax.set_xlabel(ax_label, fontdict=fontdict_label_standard)
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_lab, fontdict=fontdict_ticks_label_standard)
        ax.set_ylabel(ax_label, fontdict=fontdict_label_standard)

