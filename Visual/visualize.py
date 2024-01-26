#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 12:04:39 2021

@author: hp
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
#from mpl_toolkits.mplot3d import Axes3D
from Utils.utils import utils
from .visual_utils import create_colorbar, create_discrete_colorbar




class visualize:

    def plot_network(G,
                     ax=None,
                     node_color=list(),
                     figsize=(10, 14),
                     nodes_cmap=plt.cm.viridis,
                     edge_cmap=plt.cm.Reds,
                     weight_key='weight',
                     cb_nodes_lab='', cb_edge_lab='', edge_vmin=None, edge_vmax=None, vmin=None, vmax=None,
                     up_text=.3, hor_text=0,
                     node_size=100, numeric_label=False,
                     labels=None, save_fold=None, title='', save_name='', show=False):

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)  # Remove x and y ticks

        if labels is None:
            labels = [(node, '') for i, node in enumerate(G.nodes())]

        # colors = [G[u][v]['weight'] for u, v in G.edges]
        # edge_weight = [weight['weight'] for n1, n2, weight in list(G.edges(data=True))]
        try:
            coordinates = [(node, (feat['coord'])) for node, feat in G.nodes(data=True)]
        except:
            coordinates = nx.spring_layout(G, seed=63)

        fontdict_cb = {'fontsize': 'xx-large', 'fontweight': 'bold'}
        fontdict_cb_ticks_label = {'fontsize': 'large', 'fontweight': 'bold'}

        #fig, ax = plt.subplots(figsize=figsize)
        try:
            weights = np.array([G[u][v][weight_key] for u, v in G.edges])
        except:
            weights = np.ones(G.number_of_edges())

        if len(node_color) == 0:
            if np.min(weights) == np.max(weights):
                nx.draw_networkx(G,
                                 pos=dict(coordinates),
                                 with_labels=numeric_label,
                                 node_size=node_size,
                                 # labels=dict(labels),
                                 edge_cmap=edge_cmap,
                                 # cmap=nodes_cmap,
                                 width=4,
                                 # edge_color=weights,
                                 # edge_vmin=np.min(weights),
                                 # edge_vmax=np.max(weights)
                                 )
            else:
                nx.draw_networkx(G,
                                 pos=coordinates,
                                 with_labels=numeric_label,
                                 node_size=node_size,
                                 #labels=dict(labels),
                                 edge_cmap=edge_cmap,
                                 # cmap=nodes_cmap,
                                 width=4,
                                 edge_color=weights,
                                 edge_vmin=np.min(weights),
                                 edge_vmax=np.max(weights))
        else:
            if not edge_vmin:
                edge_vmin = np.min(weights)
            if not edge_vmax:
                edge_vmax = np.max(weights)
            if not vmin:
                vmin = np.min(node_color)

            if not vmax:
                vmax = np.max(node_color)
            nx.draw_networkx(G, pos=dict(coordinates),
                             with_labels=numeric_label,
                             node_size=node_size,
                             cmap=nodes_cmap,
                             node_color=node_color,
                             vmin=vmin,
                             vmax=vmax,
                             # labels=dict(labels),
                             edge_cmap=edge_cmap,
                             width=4,
                             edge_color=weights,
                             edge_vmin=edge_vmin,
                             edge_vmax=edge_vmax,
                             font_size='x-large',
                             font_weight='bold', ax=ax)

            shrink_bar = 0.65
            pad = .001
            ticks_num = 2
            if cb_nodes_lab != '':
                sm_nodes = plt.cm.ScalarMappable(cmap=nodes_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
                # cb_nodes = plt.colorbar(sm_nodes, ax=ax, shrink=shrink_bar, pad=pad)
                # divider = make_axes_locatable(ax)
                # cax = divider.new_horizontal(size="3%", pad=pad, pack_start=False)
                # fig.add_axes(cax)
                cb_nodes = plt.colorbar(sm_nodes, ax=ax, shrink=shrink_bar, pad=pad, orientation="vertical")

                cb_nodes.set_label(cb_nodes_lab, fontdict=fontdict_cb)
                cb_nodes_ticks = np.linspace(start=vmin, stop=vmax, num=ticks_num, endpoint=True)
                cb_tick_lab = ['{:.0e}'.format(i) for i in cb_nodes_ticks]
                # cb_tick_lab = ['$\mathregular{1 \\times 10^{-16}}$', '$\mathregular{1}$']
                cb_nodes.ax.set_yticks(cb_nodes_ticks)
                cb_nodes.ax.set_yticklabels(cb_tick_lab, fontdict=fontdict_cb_ticks_label)

            #############3
            if cb_edge_lab != '':
                sm_edges = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=edge_vmin, vmax=edge_vmax))
                # cb_edges = plt.colorbar(sm_edges, shrink=shrink_bar, orientation="horizontal", pad=pad)
                # divider = make_axes_locatable(ax)
                # cax_edge = divider.new_vertical(size="3%", pad=pad, pack_start=True)
                # fig.add_axes(cax_edge)
                cb_edges = plt.colorbar(sm_edges, ax=ax, shrink=shrink_bar, orientation="horizontal", pad=pad)

                cb_edges.set_label(cb_edge_lab, fontdict=fontdict_cb)
                cb_edge_ticks = np.linspace(start=edge_vmin, stop=edge_vmax, num=ticks_num, endpoint=True)
                cb_edge_tick_lab = ['{:.0e}'.format(i) for i in cb_edge_ticks]
                # cb_edge_tick_lab = ['$\mathregular{4 \\times 10^{-6}}$', '$\mathregular{3 \\times 10^{2}}$' ]
                cb_edges.ax.set_xticks(cb_edge_ticks)
                cb_edges.ax.set_xticklabels(cb_edge_tick_lab, fontdict=fontdict_cb_ticks_label)

        if labels:
            for (n, lab) in labels:
                x, y = coordinates[n][1]
                ax.text(x + hor_text, y + up_text, s=lab, fontdict=fontdict_cb,
                         horizontalalignment='center')  # bbox=dict(facecolor='red', alpha=0.5),

        plt.box(False)
        fontdict = {'fontsize': 'xx-large', 'fontweight': 'bold'}
        ax.set_title(title, fontdict=fontdict)
        #plt.tight_layout()
        if save_fold:
            plt.savefig(save_fold + save_name + '.png')
        #if show:
        #    plt.show()
        #else:
        #    plt.close()

        return ax

    def draw_degree_colormap(G, save_name=None, save_fold='./', title='' ):
        # Draw
        fig = plt.figure(figsize=(12,9))
        ax = fig.add_subplot(111)# Remove x and y ticks
        
        options = {"edgecolors": "tab:gray", "node_size": 100, "alpha": 0.7}
        #layout = nx.spring_layout(G, seed=10396953)
        #layout = nx.spectral_layout(G)
        layout = nx.kamada_kawai_layout(G)
        colors= [d for n, d in G.degree()]
        vmin = min(colors)
        vmax = max(colors)
        #print(vmin, vmax)
        cmap = plt.get_cmap('plasma', np.max(colors) - np.min(colors) + 1)
        nodes = nx.draw(G, pos=layout, ax=ax, node_color=colors, cmap=cmap, with_labels=False, vmin=vmin, vmax=vmax, **options)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])

        _, _, cbar, cmap = create_discrete_colorbar(fig=fig, ax=ax, array_of_values=colors, fontdict_cbar_label={'label': 'Node degree'})
        #plt.show()
        a=0
        #cbar = plt.colorbar(sm)
        # Create axis for colorbar
        #cbar_ax = make_axes_locatable(ax).append_axes(position='right', size='5%', pad=0.1)
        # Create colorbar
        #cbar = fig.colorbar(nodes, cax=cbar_ax)
        # Edit colorbar ticks and labels
        #cbar_ticks= np.arange(vmin, vmax+1, 1)
        #cbar.set_ticks(cbar_ticks)
        #cbar.ax.set_ylabel('Degree')
        #cbar_ticks_label= ['%.3f' %i for i in cbar_ticks]
        #cbar_ticks_label[-1]= cbar_ticks_label[-1:][0] + ' ' + '%s' %cbar_label 
        #cbar.ax.set_yticklabels(cbar_ticks_label)
        
        ax.set_title(title)
        ax.set_axis_off()
        plt.tight_layout()
        
        if save_name:
            plt.savefig(save_fold+save_name+'_probtheta_V.png') 
            plt.close()
        else: plt.show()
        
    def degree_analysis(G, save_name=None, save_fold='./', title='',
                        ):
        # Source code: 
        # https://networkx.org/documentation/latest/auto_examples/drawing/plot_degree.html#sphx-glr-auto-examples-drawing-plot-degree-py
        
        import networkx as nx
        import numpy as np
        import matplotlib.pyplot as plt

        #G = nx.gnp_random_graph(100, 0.02, seed=10374196)
        
        degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
        dmax = max(degree_sequence)
        
        fig = plt.figure("Degree analysis", figsize=(12, 8))
        # Create a gridspec for adding subplots of different sizes
        axgrid = fig.add_gridspec(5, 4)
        
        ax0 = fig.add_subplot(axgrid[0:3, :])
        Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
        
        #pos = nx.kamada_kawai_layout(G)
        #pos = [feat['coord'] for node, feat in G.nodes(data=True)]
        #pos = nx.spring_layout(Gcc, seed=10396953)
        colors = [d for n, d in G.degree()]
        vmin = min(colors)
        vmax = max(colors)
        #nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
        #nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
        ax0.set_title("Largest connected component")
        ax0.set_axis_off()

        cmap = plt.get_cmap('plasma', np.max(colors) - np.min(colors) + 1)
        options = {"edgecolors": "tab:gray", "node_size": 50, "alpha": 0.7}
        #nx.draw(Gcc, pos=pos, ax=ax0, node_color=colors, cmap=cmap, with_labels=False, vmin=vmin, vmax=vmax, **options)
        coordinates = [(node, (feat['coord'])) for node, feat in G.nodes(data=True)]
        try:
            edge_weight = [weight['weight'] for n1, n2, weight in list(G.edges(data=True))]
            nx.draw_networkx(G, pos=dict(coordinates), with_labels=False, cmap=cmap,
                             node_color=colors, vmin=vmin, vmax=vmax, #labels=dict(labels),
                             edge_color=edge_weight, edge_vmin=np.min(edge_weight), edge_vmax=np.max(edge_weight))
        except:
            nx.draw_networkx(G, pos=dict(coordinates), with_labels=False, cmap=cmap,
                             node_color=colors, vmin=vmin, vmax=vmax,  # labels=dict(labels),
                             )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar_ticks = np.arange(vmin, vmax+1, 1)
        cbar.set_ticks(cbar_ticks)
        cbar.ax.set_ylabel('Degree')
        
        ax1 = fig.add_subplot(axgrid[3:, :2])
        ax1.plot(degree_sequence, "b-", marker="o")
        ax1.set_title("Degree Rank Plot")
        ax1.set_ylabel("Degree")
        ax1.set_xlabel("Rank")
        
        ax2 = fig.add_subplot(axgrid[3:, 2:])
        ax2.bar(*np.unique(degree_sequence, return_counts=True))
        ax2.set_title("Degree histogram")
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("# of Nodes")
        plt.tight_layout()

        if save_name:
            plt.savefig(save_fold+save_name+'DegreeAnalysis.png') 
            plt.close()
        else: plt.show()
        
    def adjaciency_matx(G, save_name=None, save_fold='./', title='', cbar_label=''):
        #data= nx.adjacency_matrix(G)
        data = nx.to_numpy_array(G)
        
        # Create figure and add axis
        fig = plt.figure('Adjacency Matrix', figsize=(12,10))
        ax = fig.add_subplot(111)# Remove x and y ticks
        
        vmin=data.min(); vmax=data.max()
        img = ax.imshow(data, origin='lower', #cmap='YlGnBu_r', 
                        #extent=(0, data, 0, self.data.scan_size_micron), 
                        vmin=vmin, vmax=vmax)
        ticks= [int(0), int(G.number_of_nodes()-1), int(G.number_of_nodes()//2)]
        ax.set_xticks(ticks); ax.set_yticks(ticks[1:])
        ax.set_xticklabels(ticks, fontsize=14), ax.set_yticklabels(ticks[1:], fontsize=14)
        ax.set_xlabel('Node id', fontsize=15, fontweight='bold'); 
        ax.set_ylabel('Node id', fontsize=15, fontweight='bold')
        
        # Create axis for colorbar
        cbar_ax = make_axes_locatable(ax).append_axes(position='right', size='2%', pad=0.1)
        # Create colorbar
        cbar = fig.colorbar(mappable=img, cax=cbar_ax)
        # Edit colorbar ticks and labels
        cbar_ticks = np.linspace(vmin, vmax, 3)
        cbar.set_ticks(cbar_ticks)
        cbar_ticks_label = ['%.3f' %i for i in cbar_ticks]
        cbar_ticks_label[-1] = cbar_ticks_label[-1:][0] + ' ' + '%s' %cbar_label
        cbar.ax.set_yticklabels(cbar_ticks_label, fontsize=14)
        cbar.ax.set_ylabel('Weight', fontsize=15, fontweight='bold')
        
        ax.set_title(title, fontsize=20, fontweight='bold')
        plt.tight_layout()
        if save_name:
            plt.savefig(save_fold+save_name+'_Adjacency_Matrix.png')
            plt.close()
        else: plt.show()
        
    def edge_weight_distribution(G, save_name=None, save_fold='./', title='', cbar_label=''):
        import networkx as nx
        import numpy as np
        import matplotlib.pyplot as plt
        
        weights_sequence = sorted([d['weight'] for n1, n2, d in G.edges(data=True)], reverse=True)
        dmax = max(weights_sequence)
        
        fig = plt.figure("Edge weight analysis", figsize=(12, 8))
        
        ax0 = fig.add_subplot(121)
        Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
        
        pos = nx.kamada_kawai_layout(G)
        #pos = nx.spring_layout(Gcc, seed=10396953)
        colors= [d['weight'] for n1, n2, d in G.edges(data=True)]
        vmin = min(colors)
        vmax = max(colors)
        print(vmin, vmax)
        #nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
        #nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
                
        ax0.set_title("Largest connected component\nG has {} edges".format(G.number_of_edges()))
        ax0.set_axis_off()
        cmap=plt.cm.coolwarm
        options = {"node_size": 20, "alpha": 0.4}
        nx.draw(Gcc, pos=pos, ax=ax0, edge_color=colors, cmap=plt.cm.coolwarm, with_labels=False, vmin=vmin, vmax=vmax, **options)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar_ticks= np.linspace(vmin, vmax, 3, endpoint=True)
        cbar.set_ticks(cbar_ticks)
        cbar.ax.set_ylabel('Weight')
        
        ax1 = fig.add_subplot(122)
        pdf, _, bins =utils.empirical_pdf_and_cdf(weights_sequence)
        ax1.plot(bins[1:], pdf, "b-", marker="o")
        ax1.set_title("Weight histogram")
        ax1.set_ylabel("Prob")
        ax1.set_xlabel("Weight")
        
        plt.tight_layout()
        if save_name:
            plt.savefig(save_fold+save_name+'WeightAnalysis.png') 
            plt.close()
        else:
            plt.show()
        
    def histogram(bins, values, save_name=None, save_fold='./', title='', ylabel='', xlabel=''):
        fig = plt.figure(save_name, figsize=(12, 9))
                
        ax1 = fig.add_subplot(111)
        
        #ax1.plot(bins[1:], values, "b-", marker="o")
        ax1.hist(bins=bins[1:], x=values)#, "b-", marker="o")
        ax1.set_title(title, fontsize=24, fontweight='bold')
        ax1.set_ylabel(ylabel, fontsize=20, fontweight='bold')
        ax1.set_xlabel(xlabel, fontsize=20, fontweight='bold')
        ax1.xaxis.set_tick_params(labelsize=18)
        ax1.yaxis.set_tick_params(labelsize=18)

        plt.tight_layout()
        if save_name:
            plt.savefig(save_fold+save_name+'.png') 
            plt.close()
        else: plt.show()

    def plot_k_means_tsn_points(data, k=3, save_fold='/home/hp/Scrivania/', save_name='', random_state=0):
        #model.eval()
        #z = model(torch.arange(data.num_nodes, device=device))
        from sklearn.manifold import TSNE
        from scipy.cluster.vq import kmeans, vq
        from sklearn.preprocessing import StandardScaler
        #from sklearn.cluster import KMeans

        scaler = StandardScaler().fit(data)
        scaled_data= scaler.transform(data)
        z = TSNE(n_components=2, random_state=random_state).fit_transform(scaled_data)
        # Apply k-means on 2d embeddings
        # computing K-Means with K = 3 (3 clusters)

        # kmeans = KMeans(n_clusters=k, random_state=0).fit(scaler.transform(raw_embedding))
        # idx = kmeans.labels_
        centroids, _ = kmeans(scaled_data, k)
        idx, _ = vq(scaled_data, centroids)
        # draw the points
        alpha = 0.7
        label_map = {l: i for i, l in enumerate(np.unique(idx))}
        node_colors = [label_map[target] for target in idx]
        fig = plt.figure(save_name + '-embedding', figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.scatter(
            z[:, 0],
            z[:, 1],
            c=node_colors,
            cmap="jet",
            alpha=alpha, s=30
        )
        plt.axis('off')
        ax.set_title(save_name + ' - tSNE + KMeans', fontweight='bold', fontsize=15)
        utils.ensure_dir(save_fold)
        plt.savefig(save_fold + save_name + '-tSNE_2d-kmeans{:02d}'.format(k)+'.png')
        plt.close()

    def plot_k_means(data, k=3, save_fold='/home/hp/Scrivania/', save_name=''):
        #model.eval()
        #z = model(torch.arange(data.num_nodes, device=device))
        from scipy.cluster.vq import kmeans, vq
        from sklearn.preprocessing import StandardScaler
        #from sklearn.cluster import KMeans

        scaler = StandardScaler().fit(data)
        scaled_data= scaler.transform(data)
        #z = TSNE(n_components=2).fit_transform(scaled_data)
        # Apply k-means on 2d embeddings
        # computing K-Means with K = 3 (3 clusters)

        # kmeans = KMeans(n_clusters=k, random_state=0).fit(scaler.transform(raw_embedding))
        # idx = kmeans.labels_
        centroids, _ = kmeans(scaled_data, k)
        idx, _ = vq(scaled_data, centroids)
        # draw the points
        alpha = 0.7
        label_map = {l: i for i, l in enumerate(np.unique(idx))}
        node_colors = [label_map[target] for target in idx]
        fig = plt.figure(save_name + '-embedding', figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.scatter(
            scaled_data[:, 0],
            scaled_data[:, 1],
            c=node_colors,
            cmap="jet",
            alpha=alpha, s=30
        )
        plt.axis('off')
        ax.set_title(save_name + ' - KMeans', fontweight='bold', fontsize=15)
        utils.ensure_dir(save_fold)
        plt.savefig(save_fold + save_name + '-kmeans{:02d}'.format(k)+'.png')
        plt.close()


    def t_SNE_on_embeddings(node_embeddings, save_path='/home/hp/Scrivania/', save_name='', random_state=0):
        from sklearn.manifold import TSNE
        # Apply t-SNE transformation on node embeddings
        tsne = TSNE(n_components=2, random_state=random_state)
        node_embeddings_2d = tsne.fit_transform(node_embeddings)

        # draw the points
        alpha = 0.7
        # label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
        # node_colors = [label_map[target] for target in node_targets]

        fig = plt.figure(save_name + '-embedding', figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.scatter(
            node_embeddings_2d[:, 0],
            node_embeddings_2d[:, 1],
            # c=node_colors,
            cmap="jet",
            alpha=alpha,
        )
        ax.set_title(save_name + ' - tSNE', fontweight='bold', fontsize=15)
        plt.savefig(save_path + save_name + '- tSNE')

        return node_embeddings_2d

    def plot_superpix_image(image, segments, save_fold='./'):
        from skimage.segmentation import mark_boundaries
        # show the output of SLIC
        numSegments= len(np.unique(segments))
        fig = plt.figure("Superpixels -- %d segments" % (numSegments), figsize=(12,12))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(image, segments, color=(1,1,1)), origin='lower')
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_fold+"Superpixels -- %d segments" %(numSegments))
        plt.close()

    def plot_superpix_image_point_color(image, segments, G, attr_color, save_fold='./', save_name='', cbar_type=None, show=False):
        from skimage.segmentation import mark_boundaries
        import numpy as np
        #from skimage.measure import regionprops
        #regions = regionprops(segments, intensity_image=image)
        # show the output of SLIC
        numSegments= len(np.unique(segments))
        fig = plt.figure("Superpixels -- %d segments" % (numSegments), figsize=(15,12))
        ax = fig.add_subplot(1, 1, 1)
        font_size = 35
        ax.imshow(mark_boundaries(image, segments, color=(1,1,1)), origin='lower')
        # Scatter
        #cx = list(nx.get_node_attributes(G, 'cx').values())
        #cy = list(nx.get_node_attributes(G, 'cy').values())
        cx = [n[0] for n in list(G.nodes())]
        cy = [n[1] for n in list(G.nodes())]
        color = list(nx.get_node_attributes(G, attr_color).values())
        vmin = np.min(color)
        vmax = np.max(color)
        #cmap = plt.cm.jet
        if cbar_type == 'discrete':
            #cmap = plt.cm.get_cmap('PiYG', len(np.unique(color)))
            cmap = plt.cm.get_cmap('Paired', len(np.unique(color)))
            sc = ax.scatter(cx, cy, c=color, vmin=vmin, vmax=vmax, s=200, cmap=cmap)
            cbar_ax = make_axes_locatable(ax).append_axes(position='right', size='3%', pad=0.1)
            # Create colorbar
            print(np.unique(color))
            cbar = fig.colorbar(mappable=sc, cax=cbar_ax, ticks=np.arange(np.min(color), np.max(color) + 1))
            cbar.ax.set_ylabel(attr_color, fontsize=font_size, fontweight='bold')
            cbar.ax.tick_params(axis='y', labelsize=font_size)
            cbar_ticks = np.linspace(vmin, vmax, len(np.unique(color)))
            cbar.set_ticks(cbar_ticks)
        else:
            sc = ax.scatter(cx, cy, c=color, vmin=vmin, vmax=vmax, s=200)
            _, _, cbar = create_colorbar(fig=fig, ax=ax, mapp=sc, array_of_values=color,
                                         fontdict_cbar_label={'label': attr_color, 'fontsize': 50},
                                         fontdict_cbar_ticks={'axis': 'y', 'labelsize': 35})
        ax.axis("off")
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(save_fold+save_name+"-Superpixels-Scatter-%s-%d segments" %(attr_color, numSegments))
            plt.close()

    def plot_polar_plot_map(G, mov_average=3, save_fold='./', save_name=''):
        polar_features = [np.convolve(list(data), np.ones(mov_average) / mov_average, mode='valid') for data in polar_features]
        #polar_plot(polar_features[i], save_name='seg%d' % i, save_fold=self.save_fold)

        # self.bin_list = np.convolve(list(self.bin_list), np.ones(mov_average) / mov_average, mode='valid')

    def plot_superpix_image_kmeans_point_color(embedding,image, segments, G,  k=3, save_fold='./', save_name=''):
            #model.eval()
            #z = model(torch.arange(data.num_nodes, device=device))
            #from sklearn.manifold import TSNE
            from scipy.cluster.vq import kmeans, vq
            from sklearn.preprocessing import StandardScaler
            #from sklearn.cluster import KMeans
    
            scaler = StandardScaler().fit(embedding)
            scaled_data= scaler.transform(embedding)
            centroids, _ = kmeans(scaled_data, k)
            idx, _ = vq(scaled_data, centroids)
            
            from skimage.segmentation import mark_boundaries
            import numpy as np
            #from skimage.measure import regionprops
            #regions = regionprops(segments, intensity_image=image)
            # show the output of SLIC
            numSegments= len(np.unique(segments))
            fig = plt.figure("Superpixels -Kmeans%d - %d segments" % (k, numSegments), figsize=(18,12))
            ax = fig.add_subplot(1, 1, 1)
            font_size=40
            ax.imshow(mark_boundaries(image, segments))
            # Scatter
            cx = list(nx.get_node_attributes(G, 'cx').values())
            cy = list(nx.get_node_attributes(G, 'cy').values())
            color = idx
            #from random import randrange
            #color= [ randrange(k) for i in range(len(idx))]
            vmin = np.min(color)
            vmax = np.max(color)
            cmap = plt.cm.get_cmap("viridis", k)
            sc= ax.scatter(cy, cx, c=color, vmin= vmin, vmax=vmax, s=200, cmap=cmap)
            # Create axis for colorbar
            cbar_ax = make_axes_locatable(ax).append_axes(position='right', size='5%', pad=0.1)
            # Create colorbar
            cbar = fig.colorbar(mappable=sc, cax=cbar_ax)
            # Edit colorbar ticks and labels
            cbar_ticks = np.linspace(vmin, vmax, 3)
            cbar.set_ticks(cbar_ticks)
            cbar_ticks_label = ['%.0f' % i for i in cbar_ticks]
            cbar.ax.set_yticklabels(cbar_ticks_label)
            cbar.ax.set_ylabel('Clusters', fontsize=font_size+8, fontweight='bold')
            cbar.ax.tick_params(axis='y', labelsize= font_size)
    
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(save_fold+save_name+"-Superpixels-Scatter-KMeans{:02d}-{:04d}segments".format(k, numSegments)+'.png')
            plt.close()


    def show_rag(img, segments, g, save_fold='./', save_name='', attr_color=None):
        from skimage.future import graph
        import numpy as np

        font_size = 35
        fontdict_cbar= {'fontsize': font_size, 'fontweight': 'bold'}
        #fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))
        fig, ax = plt.subplots(nrows=1, figsize=(18, 16))
        # RAG
        lc = graph.show_rag(segments, g, img, border_color='yellow', ax=ax, edge_cmap='viridis')
        # Colorbar
        # specify the fraction of the plot area that will be used to draw the colorbar
        cbar_ax = make_axes_locatable(ax).append_axes(position='right', size='3%', pad=0.1)
        # Create colorbar
        cbar_edg = fig.colorbar(mappable=lc, cax=cbar_ax)
        #cbar= fig.colorbar(lc, cax=cbar_ax)
        cbar_edg.ax.set_ylabel('Edges', fontdict=fontdict_cbar)
        cbar_edg.ax.tick_params(axis='y', labelsize=font_size)
        color_edges = list(nx.get_edge_attributes(g, 'weight').values())
        vmin_edges = np.min(color_edges)
        vmax_edges = np.max(color_edges)
        cbar_edg_ticks = np.linspace(vmin_edges, vmax_edges, 3)
        cbar_edg.set_ticks(cbar_edg_ticks)
        cbar_edg_ticks_label = ['%.1f' % i for i in cbar_edg_ticks]
        cbar_edg.ax.set_yticklabels(cbar_edg_ticks_label)

        # ax[1].set_title('RAG drawn with grayscale image and viridis colormap')
        # lc = graph.show_rag(labels, g, img,
        #                    img_cmap='gray', edge_cmap='viridis', ax=ax[1])
        # fig.colorbar(lc, fraction=0.03, ax=ax[1])

        # Scatter
        cx = list(nx.get_node_attributes(g, 'cx').values())
        cy = list(nx.get_node_attributes(g, 'cy').values())
        print(len(cx))
        #try:
        k= g.nodes[1][attr_color]
        color = list(nx.get_node_attributes(g, attr_color).values())
        print(color)
        vmin = np.min(color)
        vmax = np.max(color)
        sc = ax.scatter(cy, cx, c=color, vmin=vmin, vmax=vmax, cmap='plasma', s=250)
        # Create axis for colorbar
        cbar_ax_sc = make_axes_locatable(ax).append_axes(position='bottom', size='3%', pad=0.1)
        # Create colorbar
        cbar_sc = fig.colorbar(mappable=sc, cax=cbar_ax_sc, orientation="horizontal")
        # Edit colorbar ticks and labels
        cbar_sc_ticks = np.linspace(vmin, vmax, 3)
        cbar_sc.set_ticks(cbar_sc_ticks)
        cbar_sc_ticks_label = ['%.4f' % i for i in cbar_sc_ticks]
        cbar_sc.ax.tick_params(axis='x', labelsize=font_size)
        #print(cbar_sc_ticks_label)
        # cbar_ticks_label[-1] = cbar_ticks_label[-1:][0] + ' ' + '%s' %attr_color
        cbar_sc.ax.set_xticklabels(cbar_sc_ticks_label)
        cbar_sc.ax.set_xlabel(attr_color, fontdict=fontdict_cbar)
        #except AttributeError:
        #    sc = ax.scatter(cy, cx, s=200)

        #for a in ax:
        ax.axis('off')
        ax.set_title(save_name, fontdict=fontdict_cbar)
        plt.tight_layout()
        plt.savefig(save_fold+save_name+'-RAG')
        plt.close()


    def plot_scatter(x_values, y_values, x_label='', y_label='', save_fold='./', save_name=''):
        '''x-values: 1d-list or array'''
        import numpy as np
        x_values = np.asarray(x_values)
        y_values = np.asarray(y_values)
        font_size = 35
        fontdict_cbar = {'fontsize': font_size, 'fontweight': 'bold'}
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x_values, y_values)
        y_ticks = np.linspace(y_values.min(), y_values.max(), 3)
        ax.set_yticks(y_ticks)
        y_ticks_label = ['%.3f' % i for i in y_ticks]
        ax.set_yticklabels(y_ticks_label)
        ax.set_ylabel(y_label, fontdict=fontdict_cbar)

        ax.set_xlabel(x_label, fontdict=fontdict_cbar)
        x_ticks = np.linspace(x_values.min(), x_values.max(), 3)
        ax.set_xticks(x_ticks);
        x_ticks_label = ['%.2f' % i for i in x_ticks]
        ax.set_xticklabels(x_ticks_label)
        ax.set_xlabel(y_label, fontdict=fontdict_cbar)
        ax.set_xlabel(x_label, fontdict=fontdict_cbar)
        ax.tick_params(axis='both', labelsize=font_size)
        plt.tight_layout()
        plt.savefig(save_fold + save_name)
        plt.close()

    def plot_graph_over_image(G, segments, image, save_fold='./', lw=2.5):
        from skimage.segmentation import mark_boundaries
        from .visual_utils import multiline
        # Plot graph on image
        w_list = [attr['weight'] for n1, n2, attr in G.edges(data=True)]
        x_coord_edge = np.asarray([[n1[0], n2[0]]  for n1, n2, attr in G.edges(data=True)])
        y_coord_edge = np.asarray([[n1[1], n2[1]] for n1, n2, attr in G.edges(data=True)])

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)
        # ax.imshow(image, 'gray', origin='lower')
        ax.imshow(mark_boundaries(image, segments, color=(1,1,1)), origin='lower')
        lc = multiline(x_coord_edge, y_coord_edge, c=w_list, cmap='jet', lw=lw, ax=ax)
        ax.scatter(x_coord_edge, y_coord_edge, marker="o", c='black')
        fig, ax, cbar = create_colorbar(fig=fig, ax=ax, mapp=lc, array_of_values=w_list, fontdict_cbar_label={'label':'Weights'})
        plt.tight_layout()
        plt.savefig(save_fold + 'graph_detect.png', fig=fig)
        plt.close(fig)

        #plt.hist(w_list);
        #plt.savefig(save_fold+'weighs.png');
        #plt.close()


def plot_experimental_data(time, V, I_exp, G_exp, title=''):
    plt.figure()
    plt.suptitle('Experimental data - ' + title)
    plt.subplot(211)
    ax1 = plt.gca()
    plt.grid()
    # ax1.set_xlabel('time [s]')
    ax1.set_ylabel('Voltage [V]', color='blue')
    plt.plot(time, V, 'b')
    ax1.set_xticklabels([])
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Current [A]', color='red')
    plt.plot(time, I_exp, 'r--')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.subplot(212)
    plt.grid()
    plt.plot(time, G_exp, 'b')
    plt.xlabel('time [s]')
    plt.ylabel('Conductance [S]')
    plt.show()
