import networkx as nx
import numpy as np
from .utils import utils
import copy

def set_new_weights(G, new_weights):
    edge_list = [(n1, n2) for n1, n2, weight in list(G.edges(data=True))]
    #edge_weight_list = [weight['weight'] for n1, n2, weight in list(G.edges(data=True))]
    edge_list_with_attr = [(edge[0], edge[1], {'weight': w}) for (edge, w) in zip(edge_list, new_weights)]
    G.add_edges_from(edge_list_with_attr)

def pruning_of_edges(G, p, save_fold='./'):

    def eps_edges(edge_weight_list, p=.25):
        pdf, cdf, bins = utils.empirical_pdf_and_cdf(edge_weight_list, bins=100)
        try:
            return bins[cdf < p][-1]
        except: return 0

    weights = [w['weight'] for u, v, w in G.edges(data=True)]
    # Find weights to keep: weights > epsilon
    epsilon = eps_edges(p=p, edge_weight_list=weights)
    # plot cdf
    pdf, cdf, bins = utils.empirical_pdf_and_cdf(weights, bins=100)
    #plt.close()
    #plt.plot(bins, cdf)
    #plt.axvline(epsilon, c='red')
    #plt.xlabel('Edge weight')
    #plt.ylabel('CDF')
    #plt.savefig(save_fold + 'CDF_edges.png')
    #plt.close()
    #
    return [(u, v, {'weight': w['weight']}) for u, v, w in G.edges(data=True) if w['weight'] > epsilon]


def rescale_weights(G, scale=(0.0001,1), method='MinMax'):
    from .utils import utils
    edge_list = [(n1, n2) for n1, n2, weight in list(G.edges(data=True))]
    edge_weight_list = [weight['weight'] for n1, n2, weight in list(G.edges(data=True))]
    if method == 'MinMax':
        edge_weight_list = utils.scale(edge_weight_list, scale)
    elif method == 'box-cox':
        from sklearn.preprocessing import PowerTransformer
        #print('Rescale method ', method)
        power = PowerTransformer(method='box-cox', standardize=True)
        data_trans = power.fit_transform(np.reshape(utils.scale(edge_weight_list, scale), newshape=(-1, 1)))
        edge_weight_list = utils.scale(data_trans.reshape(-1), scale)

    edge_list_with_attr = [(edge[0], edge[1], {'weight': w}) for (edge, w) in zip(edge_list, edge_weight_list)]
    G.add_edges_from(edge_list_with_attr)

def connected_components(G):
    # Return subgraphs from largest to smaller
    import networkx as nx
    subgraphs = [G.subgraph(c) for c in nx.connected_components(G) if len(c) > 1]
    sorted_subgraphs = sorted(subgraphs, key=len)
    return sorted_subgraphs[::-1]

def relative_connection_density(G, nodes):
    subG = G.subgraph(nodes).copy()
    density = nx.density(subG)
    return density

def average_weighted_degree(G, key_='weight'):
    '''Average weighted degree of a graph'''
    edgesdict = G.edges
    total = 0
    for node_adjacency_dict in edgesdict.values():
	    total += sum([adjacency.get(key_, 0) for adjacency in node_adjacency_dict.values()])
    return total

def average_degree(G):
    "Mean number of edges for a node in the network"
    degrees = G.degree()
    mean_num_of_edges = sum(dict(degrees).values()) / G.number_of_nodes()
    return mean_num_of_edges

def filter_nodes_by_attr(G, key_, key_value):
    "Returns the list of node indexs filtered by some value for the attribute key_"
    return [idx for idx, (x, y) in enumerate(G.nodes(data=True)) if y[key_] == key_value]

def add_ground_node(G, GROUND_NODE_X = 480, GROUND_NODE_Y = 250):
    ground_node = ('Ground', {'coord': (GROUND_NODE_X + 100, GROUND_NODE_Y)})
    connected_to_electrode = [(node, ground_node[0]) for node, feat in G.nodes(data=True) if
                              feat['coord'][0] > GROUND_NODE_X]
    G.add_nodes_from([ground_node])
    weights = [w['weight'] for u, v, w in G.edges(data=True)]
    w_gnd = np.max(weights)
    G.add_edges_from(connected_to_electrode, weight=w_gnd)


def effective_resistence(G):
    from networkx.algorithms.distance_measures import resistance_distance
    G_grounded = copy.deepcopy(G)
    add_ground_node(G_grounded)
    eff_res = [resistance_distance(G=G_grounded, nodeA=node_name, nodeB='Ground', weight='weight', invert_weight=True)
               for node_name in list(G.nodes()) if node_name != 'Ground']
    return eff_res, 0 #node_list
