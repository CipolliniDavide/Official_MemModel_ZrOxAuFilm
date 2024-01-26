import networkx as nx
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import datetime
from Utils.utils import utils
#from .my_functions import calculate_Isource, calculate_network_resistance
from .visualize import visualize


def plot_H_evolution_multiproc(H_Horizon,
                     node_labels,
                     t_list_Horizon,
                     V_Horizon,
                     coordinates,
                     node_voltage_list,
                     src, eff_cond_list, desired_eff_conductance, save_path,
                     numeric_label=False,
                     y_label_2ndrow='Eff. Cond. [S]',
                     number_of_plots=10, alpha=.25, t1=None):

    save_path = save_path + '/H_Evolution/'
    utils.ensure_dir(save_path)
    print('Images saved to: \n\t{:s}'.format(save_path))
    # coordinates = [(node, (feat['coord'])) for node, feat in G.nodes(data=True)]

    edge_min = np.min([h.data.min() for h in H_Horizon])
    edge_max = np.max([h.data.max() for h in H_Horizon])

    from multiprocessing import Pool, cpu_count

    # for time_index in np.linspace(0, len(H_Horizon), endpoint=False, num=number_of_plots, dtype=int):#range(len(H_Horizon)):

    def make_plot(time_index):
        H = nx.DiGraph(H_Horizon[time_index])
        nx.set_node_attributes(H, dict(coordinates), 'coord')
        node_voltage = node_voltage_list[time_index]

        figsize=(14, 12)
        fig, ax = plt.subplots(figsize=figsize)
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

        # Create a gridspec for adding subplots of different sizes
        axgrid = fig.add_gridspec(8, 4)
        ax = fig.add_subplot(axgrid[0:4, :])
        ax1 = fig.add_subplot(axgrid[4:6, :])
        ax2 = fig.add_subplot(axgrid[6:, :])

        # node_voltage = [H.nodes[n]['V'] for n in H.nodes()]

        ax = visualize.plot_network(ax=ax, G=H, figsize=(8, 6), labels=node_labels,
                                    node_color=list(node_voltage), show=False,
                                    up_text=.3, hor_text=0,
                                    numeric_label=numeric_label,
                                    #save_fold='./{:s}'.format(save_name),
                                    vmin=np.min(node_voltage_list), vmax=np.max(node_voltage_list),
                                    edge_vmin=edge_min, edge_vmax=edge_max,
                                    nodes_cmap=plt.cm.Blues,
                                    weight_key='weight',)

        ax1 = plot_voltage_input_with_time_bar(ax=ax1, V_list=V_Horizon, t_list=t_list_Horizon,
                                               time_index=time_index, src=src,
                                               #fontdict=dict(fontdict),
                                               #fontdict_ticks_label=fontdict_ticks_label,
                                               #fontdict_leg=fontdict_leg, fontdict_leg_title=fontdict_leg_title
                                               )

        ax2 = plot_eff_conductance_with_time_bar(ax2, eff_cond=eff_cond_list, t_list=t_list_Horizon, time_index=time_index,
                                                 x_label='Time [s]',
                                                 y_label=y_label_2ndrow,
                                                 y_hline=desired_eff_conductance,
                                                 alpha=alpha,
                                                 #fontdict=fontdict, fontdict_ticks_label=fontdict_ticks_label,
                                                 #fontdict_leg=fontdict_leg, fontdict_leg_title=fontdict_leg_title)
                                                 )
        if t1:
            ax2.axvspan(0, t1, color='g', alpha=0.2, lw=0)
            ax2.axvspan(t1, np.max(t_list_Horizon), color='y', alpha=0.2, lw=0)
        plt.tight_layout()
        plt.savefig(save_path+'{:04d}.png'.format(time_index))
        plt.close('all')

    with Pool(processes=cpu_count()-2) as pool:
        pool.map(make_plot, np.linspace(0, len(H_Horizon), endpoint=False, num=number_of_plots, dtype=int))


def plot_H_evolution(H_Horizon,
                     node_labels,
                     t_list_Horizon,
                     V_Horizon,
                     coordinates,
                     node_voltage_list,
                     src, eff_cond_list, desired_eff_conductance, save_path,
                     numeric_label=False,
                     title='',
                     y_label_2ndrow=r'$\mathbf{G_{nw}}$ [S]',
                     number_of_plots=10, alpha=.25, t1=None):

    save_path = save_path + '/H_Evolution/'
    utils.ensure_dir(save_path)
    print('Images saved to: \n\t{:s}'.format(save_path))
    # coordinates = [(node, (feat['coord'])) for node, feat in G.nodes(data=True)]

    # edge_min = np.min([h.data.min() for h in H_Horizon])
    # edge_max = np.max([h.data.max() for h in H_Horizon])

    for time_index in np.linspace(0, len(H_Horizon), endpoint=False, num=number_of_plots, dtype=int):#range(len(H_Horizon)):
        H = nx.DiGraph(H_Horizon[time_index])
        nx.set_node_attributes(H, dict(coordinates), 'coord')
        node_voltage = node_voltage_list[time_index]

        figsize=(14, 12)
        fig, ax = plt.subplots(figsize=figsize)
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

        # Create a gridspec for adding subplots of different sizes
        axgrid = fig.add_gridspec(8, 4)
        ax = fig.add_subplot(axgrid[0:6, :])
        ax1 = fig.add_subplot(axgrid[6:7, :])
        ax2 = fig.add_subplot(axgrid[7:, :])

        # node_voltage = [H.nodes[n]['V'] for n in H.nodes()]
        edge_min = H_Horizon[time_index].data.min()
        edge_max = H_Horizon[time_index].data.max()

        ax = visualize.plot_network(ax=ax, G=H, figsize=(8, 6), labels=node_labels,
                                    node_color=list(node_voltage), show=False,
                                    up_text=.3, hor_text=0,
                                    numeric_label=numeric_label,
                                    #save_fold='./{:s}'.format(save_name),
                                    # vmin=np.min(node_voltage_list), vmax=np.max(node_voltage_list),
                                    vmin=np.min(node_voltage), vmax=np.max(node_voltage),
                                    edge_vmin=edge_min, edge_vmax=edge_max,
                                    nodes_cmap=plt.cm.Blues,
                                    weight_key='weight',)

        ax1 = plot_voltage_input_with_time_bar(ax=ax1, V_list=V_Horizon, t_list=t_list_Horizon,
                                               time_index=time_index, src=src,
                                               #fontdict=dict(fontdict),
                                               #fontdict_ticks_label=fontdict_ticks_label,
                                               #fontdict_leg=fontdict_leg, fontdict_leg_title=fontdict_leg_title
                                               )

        ax2 = plot_eff_conductance_with_time_bar(ax2, eff_cond=eff_cond_list, t_list=t_list_Horizon, time_index=time_index,
                                                 x_label='Time [s]',
                                                 y_label=y_label_2ndrow,
                                                 y_hline=desired_eff_conductance,
                                                 alpha=alpha,
                                                 #fontdict=fontdict, fontdict_ticks_label=fontdict_ticks_label,
                                                 #fontdict_leg=fontdict_leg, fontdict_leg_title=fontdict_leg_title)
                                                 )
        if t1:
            ax2.axvspan(0, t1, color='g', alpha=0.2, lw=0)
            ax2.axvspan(t1, np.max(t_list_Horizon), color='y', alpha=0.2, lw=0)
        plt.suptitle(title, weight='bold', size='xx-large')
        plt.tight_layout()
        plt.savefig(save_path+'{:04d}.png'.format(time_index))
        plt.close('all')




def plot_cAFM(H_Horizon,
              t_list_Horizon,
              Vbias,
              coordinates,
              cmap,
              source_loc,
              save_path,
              numeric_label=False,
              node_labels=None,
              number_of_plots=10,
              alpha=.25,
              t1=None):

    save_path = save_path #+ '/cAFM/'
    utils.ensure_dir(save_path)
    print('Images saved to: \n\t{:s}'.format(save_path))
    # coordinates = [(node, (feat['coord'])) for node, feat in G.nodes(data=True)]

    edge_min = np.min([h.data.min() for h in H_Horizon])
    edge_max = np.max([h.data.max() for h in H_Horizon])

    # for time_index in np.linspace(0, len(H_Horizon), endpoint=False, num=number_of_plots, dtype=int):#range(len(H_Horizon)):
    node_mean_curr = cmap[:, :np.shape(cmap)[1] // 2].mean(axis=1)
    node_color = np.empty(len(cmap))
    node_color[:] = np.NAN
    node_mean_curr[node_labels[0][0]] = np.nan
    vmin = np.min(node_mean_curr[~np.isnan(node_mean_curr)])
    vmax = np.max(node_mean_curr[~np.isnan(node_mean_curr)])
    for time_index, src in enumerate(np.array(source_loc).repeat(cmap.shape[1]//2)):
        H = nx.DiGraph(H_Horizon[time_index])
        nx.set_node_attributes(H, dict(coordinates), 'coord')
        node_color[src] = node_mean_curr[src]#cmap[src, :np.shape(cmap)[1]//2].mean()

        figsize=(10, 8)
        fig, ax = plt.subplots(figsize=figsize)
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

        # Create a gridspec for adding subplots of different sizes
        axgrid = fig.add_gridspec(8, 4)
        ax = fig.add_subplot(axgrid[0:4, :])
        ax1 = fig.add_subplot(axgrid[4:6, :])
        ax2 = fig.add_subplot(axgrid[6:, :])

        ax = visualize.plot_network(ax=ax, G=H, figsize=(8, 6), labels=node_labels + [(src, 'Src')],
                                    node_color=list(node_color), show=False,
                                    up_text=.3, hor_text=0,
                                    numeric_label=numeric_label,
                                    #save_fold='./{:s}'.format(save_name),
                                    vmin=vmin, vmax=vmax,
                                    edge_vmin=edge_min, edge_vmax=edge_max,
                                    nodes_cmap=plt.cm.viridis,
                                    weight_key='weight',)

        ax1 = plot_voltage_input_with_time_bar(ax=ax1, V_list=[[Vbias]*len(t_list_Horizon)], t_list=t_list_Horizon,
                                               time_index=time_index, src=src,
                                               #fontdict=dict(fontdict),
                                               #fontdict_ticks_label=fontdict_ticks_label,
                                               #fontdict_leg=fontdict_leg, fontdict_leg_title=fontdict_leg_title
                                               )

        ax2 = plot_eff_conductance_with_time_bar(ax2, eff_cond=cmap[np.arange(len(cmap))!=node_labels[0][0]].reshape(-1, order='C'),
                                                 t_list=t_list_Horizon, time_index=time_index,
                                                 x_label='Time [s]',
                                                 y_label='Current [A]',
                                                 y_hline=None,
                                                 alpha=alpha,
                                                 #fontdict=fontdict, fontdict_ticks_label=fontdict_ticks_label,
                                                 #fontdict_leg=fontdict_leg, fontdict_leg_title=fontdict_leg_title)
                                                 )
        if t1:
            ax2.axvspan(0, t1, color='g', alpha=0.2, lw=0)
            ax2.axvspan(t1, np.max(t_list_Horizon), color='y', alpha=0.2, lw=0)
        plt.tight_layout()
        plt.savefig(save_path+'{:04d}.png'.format(time_index))
        plt.close('all')
    a=0







def plot_animation(H_list, V_list, Iread_list, src, gnd, t_list, edge_min_max,
                   anim_save=0, save_path='./output/network_model/', loc_legend='upper right'):
    '''
    for s in src:
        H_list[0].nodes[s]['label'] = '\nSrc{:d}'.format(s)
    for s in gnd:
        H_list[0].nodes[s]['label'] = '\nGnd{:d}'.format(s)
    for s in Iread_list:
        H_list[0].nodes[s]['label'] = '\nNode{:d}'.format(s)
    '''
    labels = nx.get_node_attributes(H_list[0], "label")

    try:
        coordinates = [(node, (feat['coord'])) for node, feat in H_list[0].nodes(data=True)]
    except:
        coordinates = nx.spring_layout(H_list[0], seed=63)
    pos = dict(coordinates)

    frames_num = len(t_list)
    frames_interval = 100

    fig, ax = plt.subplots(figsize=(10, 14))
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(8, 4)
    ax0 = fig.add_subplot(axgrid[0:4, :])
    ax = fig.add_subplot(axgrid[4:6, :])
    ax1 = fig.add_subplot(axgrid[6:, :])

    I_output = [[]]*len(Iread_list)
    for H in H_list:
        for v in range(len(Iread_list)):
            I_output[v] = I_output[v] + [calculate_Isource(H=H, sourcenode=Iread_list[v])]

    def update(i):
        ax.cla()
        ax0.cla()
        ax1.cla()
        # pos = nx.get_node_attributes(H_list[i], 'pos')
        # pos = nx.spring_layout(H_list[i], seed=2)

        ax0.set_title('t = {:.5f} s'.format(t_list[i]), fontsize='large')  # + str(round(t_list[i], 4)) + ' s')

        nx.draw_networkx(H_list[i], pos,
                         # NODES
                         node_size=60,
                         node_color=[H_list[i].nodes[n]['V'] for n in H_list[i].nodes()],
                         cmap=plt.cm.Blues,
                         vmin=0,
                         vmax=np.max(V_list),
                         # EDGES
                         width=4,
                         edge_color=[H_list[i][u][v]['Y'] for u, v in H_list[i].edges()],
                         edge_cmap=plt.cm.Reds,
                         edge_vmin=edge_min_max[0],
                         edge_vmax=edge_min_max[1],
                         with_labels=True,  # Set TRUE to see node numbers
                         labels=labels,
                         font_size=18,
                         font_weight='bold', ax=ax0)

        nx.draw_networkx_nodes(H_list[i], pos, nodelist=src + gnd, node_size=100, node_color='k', ax=ax0)

        # Plot Voltage Source
        for v in range(len(V_list)):
            float_index = [j for j in range(len(V_list[v])) if V_list[v][j] == 'f']
            value_index = [j for j in range(len(V_list[v])) if V_list[v][j] != 'f']
            p = ax.plot([t_list[j] for j in value_index], [float(V_list[v][j]) for j in value_index],
                        label='Node' + str(src[v]), linewidth=2)
            color = p[0].get_color()
            ax.plot([t_list[j] for j in float_index], [0] * len(float_index), 'x', color=color, linewidth=2)
        y_max = np.max(V_list) + 1
        ax.vlines(x=[t_list[i]], ymin=0, ymax=y_max, colors='r', linewidth=4)
        #ax.set_ylim((0, y_max))
        #ax.set_xlabel('Time [s]', fontsize=20)
        ax.set_ylabel('Voltage [V]', fontsize=20)
        ax.tick_params(axis='both', labelsize='x-large')
        ax.set_xticklabels([])
        #ax.set_yticks(fontsize=15)
        ax.legend(fontsize='x-large', title="Input:", title_fontproperties={'weight': 'bold', 'size': 'x-large'}, loc=loc_legend)
        ax.grid()

        # Plot Output Current
        for outnode, I in zip(Iread_list, I_output):
            p = ax1.plot(t_list, I,
                    label='Node' + str(outnode), linewidth=2)
            #color = p[0].get_color()
            #ax.plot([t_list[j] for j in float_index], [0] * len(float_index), 'x', color=color, linewidth=2)
        y_max = np.max(I_output) + np.max(I_output)*3/10
        ax1.vlines(x=[t_list[i]], ymin=0, ymax=y_max, colors='r', linewidth=4)
        ax1.set_ylim((0, y_max))
        ax1.set_xlabel('Time [s]', fontsize=20)
        ax1.set_ylabel('Current [A]', fontsize=20)
        ax1.tick_params(axis='both', labelsize='x-large')
        # ax.set_yticks(fontsize=15)
        ax1.legend(fontsize='x-large', title="Output:", title_fontproperties={'weight': 'bold', 'size': 'x-large'}, loc=loc_legend)
        ax1.grid()

    anim = matplotlib.animation.FuncAnimation(fig, update, frames=frames_num, interval=frames_interval, blit=False,
                                              repeat=True)

    if anim_save == 1:
        print('\n')
        print('Animation Saving...')
        current_date_and_time = datetime.datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")
        current_date_and_time_string = str(current_date_and_time)
        file_name = current_date_and_time_string
        utils.ensure_dir(save_path)
        anim.save(save_path + file_name + '_animation.gif', writer='imagemagick')

    return anim
    #plt.show()


def plot_animation_CondEff(H_list, V_list,
                           EffCond_couples,
                           src, gnd, t_list, edge_min_max, figsize=(15, 13),
                           legend_title='',
                           fontdict={'fontsize': 28, 'fontweight': 'bold'}, labels=None,
                           fontdict_ticklabels={'fontsize': 22, 'fontweight': 'bold'},
                           fontdict_tick={'labelsize':'x-large'},
                           fontdict_leg={'weight': 'bold', 'size': 25},
                           fontdict_leg_title={'weight': 'bold', 'size': 28},

                           numeric_labels=False, up_text=1, node_size=100, y_hline=None,
                           title='',
                           anim_save=0, save_path='./output/network_model/', loc_legend='upper right'):
    '''
    for s in src:
        H_list[0].nodes[s]['label'] = '\nSrc{:d}'.format(s)
    for s in gnd:
        H_list[0].nodes[s]['label'] = '\nGnd{:d}'.format(s)
    for s in Iread_list:
        H_list[0].nodes[s]['label'] = '\nNode{:d}'.format(s)
    '''
    #labels = nx.get_node_attributes(H_list[0], "label")

    try:
        coordinates = [(node, (feat['coord'])) for node, feat in H_list[0].nodes(data=True)]
    except:
        coordinates = nx.spring_layout(H_list[0], seed=63)
    pos = dict(coordinates)

    frames_num = len(t_list)
    frames_interval = 100

    fig, ax = plt.subplots(figsize=figsize)
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(8, 4)
    ax0 = fig.add_subplot(axgrid[0:4, :])
    ax = fig.add_subplot(axgrid[4:6, :])
    ax1 = fig.add_subplot(axgrid[6:, :])

    #CondEff = [[]] * len(EffCond_couples)
    #for H in H_list:
    #    for i, (v, u) in enumerate(EffCond_couples):
    #        CondEff[i] = CondEff[i] + [1/calculate_network_resistance(H=H, sourcenode=u, groundnode=v)]

    def update(i):
        ax.cla()
        ax0.cla()
        ax1.cla()
        # pos = nx.get_node_attributes(H_list[i], 'pos')
        # pos = nx.spring_layout(H_list[i], seed=2)

        ax0.set_title(title, fontdict=fontdict)  # + str(round(t_list[i], 4)) + ' s')

        nx.draw_networkx(H_list[i], pos,
                         # NODES
                         node_size=node_size,
                         node_color=[H_list[i].nodes[n]['V'] for n in H_list[i].nodes()],
                         cmap=plt.cm.Blues,
                         vmin=0,
                         vmax=np.max(V_list),
                         # EDGES
                         width=4,
                         edge_color=[H_list[i][u][v]['Y'] for u, v in H_list[i].edges()],
                         edge_cmap=plt.cm.Reds,
                         edge_vmin=edge_min_max[0],
                         edge_vmax=edge_min_max[1],
                         with_labels=numeric_labels,  # Set TRUE to see node numbers
                         #labels=labels,
                         font_size=22,
                         font_weight='bold', ax=ax0)

        nx.draw_networkx_nodes(H_list[i], pos, nodelist=src + gnd, node_size=100, node_color='k', ax=ax0)

        if labels:
            for (n, lab) in labels:
                x, y = coordinates[n][1]
                plt.text(x, y + up_text, s=lab, fontdict=fontdict,
                         horizontalalignment='center')  # bbox=dict(facecolor='red', alpha=0.5),

        # Plot Voltage Source
        for v in range(len(V_list)):
            float_index = [j for j in range(len(V_list[v])) if V_list[v][j] == 'f']
            value_index = [j for j in range(len(V_list[v])) if V_list[v][j] != 'f']
            p = ax.plot([t_list[j] for j in value_index], [float(V_list[v][j]) for j in value_index],
                        label='Node' + str(src[v]), linewidth=2)
            color = p[0].get_color()
            ax.plot([t_list[j] for j in float_index], [0] * len(float_index), 'x', color=color, linewidth=2)
        y_max = np.max(V_list) + 1
        ax.vlines(x=[t_list[i]], ymin=0, ymax=y_max, colors='r', linewidth=4)
        ax.set_ylim((0, y_max))
        # ax.set_xlabel('Time [s]', fontsize=20)
        ax.set_ylabel('Voltage [V]', fontdict=fontdict)
        y_ticks = np.linspace(np.min(V_list), np.max(V_list), 3)
        y_ticks_label = ['{:.1f}'.format(t) for t in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks_label, fontdict=fontdict_ticklabels, minor=False)

        #ax.tick_params(axis='both', labelsize=fontdict_tick['labelsize'])
        ax.set_xticklabels([])
        # ax.set_yticks(fontsize=15)
        ax.legend(title="Input:", title_fontproperties=fontdict_leg_title, prop=fontdict_leg,
                  loc=loc_legend)
        ax.grid()

        # Plot Eff Res
        #for outnode, I in zip(EffCond_couples, CondEff):
        #    p = ax1.plot(t_list, I,
         #                label='Node' + str(outnode), linewidth=2)
            # color = p[0].get_color()
            # ax.plot([t_list[j] for j in float_index], [0] * len(float_index), 'x', color=color, linewidth=2)
        ax1.plot(t_list, effCond_list, label='Node' + str(src[v]), linewidth=2)

        y_max = np.max([np.max(effCond_list), y_hline])
        y_max = y_max + y_max * 3 / 10
        ax1.vlines(x=[t_list[i]], ymin=0, ymax=y_max, colors='r', linewidth=4)
        #ax1.set_ylim((0, y_max))
        ax1.set_xlabel('Time [s]', fontdict=fontdict)
        ax1.set_ylabel('Eff. Cond. [S]', fontdict=fontdict)
        x_ticks = np.linspace(0, t_list[-1], 3)
        x_ticks_label = ['{:.0e}'.format(t) for t in x_ticks]
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_ticks_label, fontdict=fontdict_ticklabels, minor=False)


        y_ticks = np.linspace(np.min([np.min(CondEff), y_hline]), y_max, 3)
        y_ticks_label = ['{:.1e}'.format(t) for t in y_ticks]
        ax1.set_yticks(y_ticks)
        ax1.set_yticklabels(y_ticks_label, fontdict=fontdict_ticklabels, minor=False)

        #ax1.tick_params(axis='both', labelsize=fontdict_tick['labelsize'])
        # ax.set_yticks(fontsize=15)
        if y_hline:
            ax1.hlines(y=y_hline, xmax=t_list[-1], label='Desired Eff.Cond.', xmin=t_list[0], color='purple',
                       linewidth=4,
                       linestyle='--', alpha=0.6)
        ax1.legend(title=legend_title, title_fontproperties=fontdict_leg_title, prop=fontdict_leg,
                   loc=loc_legend)

        ax1.grid()

        plt.tight_layout()

    anim = matplotlib.animation.FuncAnimation(fig, update, frames=frames_num, interval=frames_interval, blit=False,
                                              repeat=True)

    if anim_save == 1:
        print('\n')
        print('Animation Saving...')
        current_date_and_time = datetime.datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")
        current_date_and_time_string = str(current_date_and_time)
        file_name = current_date_and_time_string
        utils.ensure_dir(save_path)
        anim.save(save_path + file_name + '_animation.gif', writer='imagemagick')

    return anim


def plot_voltage_input_with_time_bar(ax, V_list, t_list, src, time_index,
                                     fontdict={'fontsize': 20, 'fontweight': 'bold'},
                                     fontdict_ticks_label={'fontsize': 18, 'fontweight': 'bold'},

                                     loc_legend='upper right',
                                     fontdict_leg={'weight': 'bold', 'size': 20},
                                     fontdict_leg_title={'weight': 'bold', 'size': 18},
                                     ):
    # VOLTAGE INPUT
    # Plot Voltage Source
    for i, v in enumerate(V_list):
        p = ax.plot(t_list, v, label='Src', linewidth=2)
        # color = p[0].get_color()
        # ax.plot([t_list[j] for j in float_index], [0] * len(float_index), 'x', color=color, linewidth=2)
    # y_max = np.max(V_list) + (np.max(V_list) - np.min(V_list))
    # ax.set_ylim((0, y_max))
    # ax.set_xlabel('Time [s]', fontsize=20)
    ax.set_ylabel('Volt. [V]', fontdict=fontdict)
    y_ticks = np.linspace(np.min(V_list), np.max(V_list), 3)
    y_ticks_label = ['{:.1f}'.format(t) for t in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks_label, fontdict=fontdict_ticks_label, minor=False)
    ax.vlines(x=t_list[time_index], ymin=0, ymax=y_ticks[-1], colors='r', linewidth=4)

    # ax.tick_params(axis='both', labelsize=fontdict_tick['labelsize'])
    ax.set_xticklabels([])
    # ax.set_yticks(fontsize=15)
    #ax.legend(title="Input:", title_fontproperties=fontdict_leg_title, prop=fontdict_leg,
    #           loc=loc_legend)
    ax.grid()
    return ax

def plot_eff_conductance_with_time_bar(ax, eff_cond, t_list, time_index,
                                       curve_label='',
                                       x_label='',
                                       y_label='Eff. Cond. [S]',
                                        y_hline=None,
                                        fontdict={'fontsize': 20, 'fontweight': 'bold'},
                                        fontdict_ticks_label={'fontsize': 18, 'fontweight': 'bold'},

                                        loc_legend='upper right',
                                        fontdict_leg={'weight': 'bold', 'size': 20},
                                        fontdict_leg_title={'weight': 'bold', 'size': 18},
                                        alpha=.25):
    ## PLOT EFFECTIVE CONDUCTANCE
    ax.plot(t_list, eff_cond, linewidth=2, label=curve_label)
    if y_hline:
        y_max = np.max([np.max(eff_cond), np.max(y_hline)])

    else:
        y_max = np.max(eff_cond)

    y_max = y_max #+ y_max * 3 / 10
    # ax1.set_ylim((0, y_max))
    ax.set_xlabel(x_label, fontdict=fontdict)
    ax.set_ylabel(y_label, fontdict=fontdict)
    x_ticks = np.linspace(0, t_list[-1], 3)
    x_ticks_label = ['{:.0e}'.format(t) for t in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks_label, fontdict=fontdict_ticks_label, minor=False)

    if y_hline:
        y_ticks = np.linspace(np.min([np.min(eff_cond), np.max(y_hline)]), y_max, 3)
    else:
        y_ticks = np.linspace(np.min(eff_cond), y_max, 3)

    y_ticks_label = ['{:.1e}'.format(t) for t in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks_label, fontdict=fontdict_ticks_label, minor=False)

    ax.vlines(x=t_list[time_index], ymin=y_ticks[0], ymax=y_ticks[-1], colors='r', linewidth=4)

    # ax1.tick_params(axis='both', labelsize=fontdict_tick['labelsize'])
    # ax.set_yticks(fontsize=15)
    if y_hline:
        if type(y_hline) == tuple:
            ax.fill_between(x_ticks, y_hline[0], y_hline[1], color='purple', alpha=alpha)
        else:
            ax.hlines(y=y_hline, xmax=t_list[-1], label='Desired Eff.Cond.', xmin=t_list[0], color='purple',
                       linewidth=4, linestyle='--', alpha=0.6)
    if curve_label != '':
        ax.legend(title='', title_fontproperties=fontdict_leg_title, prop=fontdict_leg,
                   loc=loc_legend)

    ax.grid()
    return ax



