from matplotlib import pyplot as plt
from .visual_utils import align_yaxis, set_legend, set_ticks_label
import numpy as np
import matplotlib

def plot_twinx(x_data, x_label, y_data, y_label, y2_data=[], y2_label=None, save_path=None, figname=None,
               y_scale=['lin', 'lin'], title=None, curve_labels_y=[''], curve_labels_y2=[''], show=False,
               ax=None, y1_ticks=None, y2_ticks=None, x_ticks=None, colors=['red', 'blue']):

    if type(y_data) is not list:
        y_data = [y_data]
    if type(y2_data) is not list:
        y2_data = [y2_data]

    if y1_ticks is None:
        y1_ticks = [np.min(y_data), 0, np.max(y_data)]
    else:
        pass

    # colors = ['red', 'blue']


    if ax is None:
        fig, ax = plt.subplots(figsize=(10,8))
    axes = [ax]
    # make a plot
    if y_scale[0] == 'log':
        ax.semilogy(x_data, y_data, color="red", marker="o", markersize=.5)
    else:
        for (y, cur_lab, c) in zip(y_data, curve_labels_y, colors):
            ax.plot(x_data, y, color=c, label=cur_lab, marker='o',
                linewidth=3, markersize=1)
    if x_ticks is None:
        set_ticks_label(ax=ax, ax_label=x_label, ax_type='x', valfmt="{x:.1f}", data=x_data, num=5)
    else:
        set_ticks_label(ax=ax, ax_label=x_label, ax_type='x', valfmt="{x:.1f}", data=x_data, ticks=x_ticks, num=5)
    if len(y2_data) != 0:
        if y2_ticks is None:
            y2_ticks = [np.min(y2_data), 0, np.max(y2_data)]
        else:
            pass
        # twin object for two different y-axis on the sample plot
        ax2 = ax.twinx()
        axes.append(ax2)
        # make a plot with different y-axis using second axis object
        if y_scale[1] == 'log':
            ax2.semilogy(x_data, y2_data, color="green", marker="o", markersize=.5)
        else:
            for (y, cur_lab) in zip(y2_data, curve_labels_y2):
                ax2.plot(x_data, y, color="green",  marker='o',
                linewidth=3, markersize=1)
        set_ticks_label(ax=ax2, ax_label=y2_label, ax_type='y', data=y2_data, num=5,
                        ticks=y2_ticks,
                        fontdict_label={'color':'green'})
    if title:
        plt.title(title)
    ax.grid()
    if len(y2_data) > 0:
        align_yaxis(ax, ax2)
    if len(y_data) > 1:
        set_legend(ax=ax)
        set_ticks_label(ax=ax, ax_label=y_label, ax_type='y', data=y_data, num=5,
                        ticks=y1_ticks, fontdict_label={'color': c})
    set_ticks_label(ax=ax, ax_label=y_label, ax_type='y', data=y_data, num=5,
                    ticks=y1_ticks)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path + figname)
    if show==True:
        plt.show()

    # return fig, axes


def plot_twinx_scatter(x_data, x_label, y_data, y_label, y2_data=[], y2_label=None, save_path=None, figname=None,
               y_scale=['lin', 'lin'], title=None, curve_labels_y=[''], curve_labels_y2=[''], show=False, format=None):

    if type(y_data) is not list:
        y_data = [y_data]
    if type(y2_data) is not list:
        y2_data = [y2_data]

    colors = ['red', 'blue']

    fig, ax = plt.subplots(figsize=(10,8))
    # make a plot
    if y_scale[0] == 'log':
        ax.semilogy(x_data, y_data, color="red", marker="o", markersize=.5)
    else:
        for (y, cur_lab, c) in zip(y_data, curve_labels_y, colors):
            ax.scatter(x_data, y, color=c, marker="o", s=.4, label=cur_lab)
    set_ticks_label(ax=ax, ax_label=x_label, ax_type='x', valfmt="{x:.1f}", data=x_data, num=5)

    if len(y2_data) != 0:
        # twin object for two different y-axis on the sample plot
        ax2 = ax.twinx()
        # make a plot with different y-axis using second axis object
        if y_scale[1] == 'log':
            ax2.semilogy(x_data, y2_data, color="green", marker="o", markersize=.5)
        else:
            for (y, cur_lab) in zip(y2_data, curve_labels_y2):
                ax2.scatter(x_data, y, color="green", marker="o", markersize=.5)
        set_ticks_label(ax=ax2, ax_label=y2_label, ax_type='y', data=y2_data, num=5,
                        ticks=[np.min(y2_data), 0, np.max(y2_data)],
                        fontdict_label={'color':'green'})
    if title:
        plt.title(title)
    ax.grid()
    if len(y2_data) > 0:
        align_yaxis(ax, ax2)
    if len(y_data) > 1:
        set_legend(ax=ax)
        set_ticks_label(ax=ax, ax_label=y_label, ax_type='y', data=y_data, num=5,
                        ticks=[np.min(y_data), 0, np.max(y_data)], fontdict_label={'color': c})
    set_ticks_label(ax=ax, ax_label=y_label, ax_type='y', data=y_data, num=5,
                    ticks=[np.min(y_data), 0, np.max(y_data)])
    plt.tight_layout()

    if save_path:
        if format is 'svg':
            plt.savefig(save_path + figname + '.svg', format='svg', dpi=1200)
        else:
            plt.savefig(save_path + figname+'.png')
    if show==True:
        plt.show()
    plt.close()
