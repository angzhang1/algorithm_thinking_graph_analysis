# general imports
import urllib2
import random
import time
import math
import bfs_visited

# Desktop imports
import matplotlib.pyplot as plt
import upa_trail
import Project1_Graph as simple_graph
import numpy as np


############################################
# Provided code

def copy_graph(graph):
    """
    Make a copy of a graph
    """
    new_graph = {}
    for node in graph:
        new_graph[node] = set(graph[node])
    return new_graph


def delete_node(ugraph, node):
    """
    Delete a node from an undirected graph
    """
    neighbors = ugraph[node]
    ugraph.pop(node)
    for neighbor in neighbors:
        ugraph[neighbor].remove(node)


def targeted_order(ugraph):
    """
    Compute a targeted attack order consisting
    of nodes of maximal degree

    Returns:
    A list of nodes
    """
    # copy the graph
    new_graph = copy_graph(ugraph)

    order = []
    while len(new_graph) > 0:
        max_degree = -1
        for node in new_graph:
            if len(new_graph[node]) > max_degree:
                max_degree = len(new_graph[node])
                max_degree_node = node

        neighbors = new_graph[max_degree_node]
        new_graph.pop(max_degree_node)
        for neighbor in neighbors:
            new_graph[neighbor].remove(max_degree_node)

        order.append(max_degree_node)
    return order


##########################################################
# Code for loading computer network graph

NETWORK_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_rf7.txt"


def load_graph(graph_url):
    """
    Function that loads a graph given the URL
    for a text representation of the graph

    Returns a dictionary that models a graph
    """
    graph_file = urllib2.urlopen(graph_url)
    graph_text = graph_file.read()
    graph_lines = graph_text.split('\n')
    graph_lines = graph_lines[: -1]

    print "Loaded graph with", len(graph_lines), "nodes"

    answer_graph = {}
    for line in graph_lines:
        neighbors = line.split(' ')
        node = int(neighbors[0])
        answer_graph[node] = set([])
        for neighbor in neighbors[1: -1]:
            answer_graph[node].add(int(neighbor))

    return answer_graph


# Fast algorithm for generating targeted order based on max degree
def fast_targeted_order(ugraph):
    """
    compute a attach order based on node degree
    :param ugraph: input graph, need to be undirected
    :return: a list of the nodes in the graph in decreasing order of their degrees
    """
    new_graph = copy_graph(ugraph)
    num_nodes = len(new_graph)
    # Init to empty sets for each degree 0 to n-1
    degree_set = {i: set() for i in xrange(num_nodes)}

    for node in new_graph:
        degree_set[len(new_graph[node])].add(node)

    l_attack = []
    for degree in xrange(num_nodes - 1, -1, -1):
        while degree_set[degree]:
            target_node = degree_set[degree].pop()
            for neigh in new_graph[target_node]:
                deg_neighbor = len(new_graph[neigh])
                degree_set[deg_neighbor].remove(neigh)
                degree_set[deg_neighbor - 1].add(neigh)
            l_attack.append(target_node)
            delete_node(new_graph, target_node)
    return l_attack


##########################################################
# Code for generating random graph
def er_ugraph(num_nodes, p):
    """
    create a random generated graph
    :param num_nodes:
    :param p:
    :return: a dictionay representing a undirected graph
    """
    if num_nodes == 0:
        return {}

    random.seed()

    ugraph = {i: set([]) for i in xrange(num_nodes)}

    for i in xrange(num_nodes):
        for x in xrange(i + 1, num_nodes):
            if random.random() < p:
                ugraph[i].add(x)
                ugraph[x].add(i)

    return ugraph


def random_order(ugraph, num_nodes):
    """
    return a list of nodes in ugraph in random order
    :param ugraph:
    :param num_nodes:
    :return:
    """
    if len(ugraph) is 0 or num_nodes is 0:
        return []

    random.seed()

    return random.sample(ugraph.keys(), num_nodes)


def num_edges(ugraph):
    """
    compute total number of edges for a undirected graph
    :param ugraph:
    :return:
    """
    total_edges = 0
    for edges in ugraph.values():
        total_edges += len(edges)
    return total_edges / 2


def upa_graph(num_nodes, ave_out_degree):
    """
    generate upa graph
    :param num_nodes: total number of nodes
    :param ave_out_degree: this controls the edges
    :return:
    """
    trail = upa_trail.UPATrial(ave_out_degree)
    ugraph = simple_graph.make_complete_graph(ave_out_degree)

    for i in xrange(ave_out_degree, num_nodes):
        ugraph[i] = trail.run_trial(ave_out_degree)
        for neigh in ugraph[i]:
            ugraph[neigh].add(i)

    return ugraph


if __name__ == '__main__':
    network_graph = load_graph(NETWORK_URL)

    num_nodes_total = len(network_graph.keys())
    num_edges_network = num_edges(network_graph)

    # Probability
    er_prob = num_edges_network * 2.0 / (num_nodes_total * num_nodes_total)
    ugraph_er = er_ugraph(num_nodes_total, er_prob)

    average_out_degree = num_edges_network / num_nodes_total
    ugraph_upa = upa_graph(num_nodes_total, average_out_degree)

    resilience_network = np.zeros(1 + num_nodes_total)
    resilience_er = np.zeros(1 + num_nodes_total)
    resilience_upa = np.zeros(1 + num_nodes_total)

    print "ER probability", er_prob
    print "UPA m", average_out_degree

    ####################################################
    # Resilience under random attack order

    # Take the average of 3 resilience
    for dummy_idx in range(3):
        cloned_network = copy_graph(network_graph)
        resilience_network = resilience_network + \
                             np.array(bfs_visited.compute_resilience(cloned_network,
                                                                     random_order(cloned_network, num_nodes_total)))
        cloned_er = copy_graph(ugraph_er)
        resilience_er = resilience_er + \
                        np.array(bfs_visited.compute_resilience(cloned_er, random_order(cloned_er, num_nodes_total)))
        cloned_upa = copy_graph(ugraph_upa)
        resilience_upa = resilience_upa + \
                         np.array(bfs_visited.compute_resilience(cloned_upa, random_order(cloned_upa, num_nodes_total)))

    resilience_network /= 3.0
    resilience_er /= 3.0
    resilience_upa /= 3.0

    xvals = range(num_nodes_total + 1)
    plt.figure()
    plt.plot(xvals, resilience_network, '-b', label='network graph')
    plt.plot(xvals, resilience_er, '-r', label='er graph, p=' + str(er_prob))
    plt.plot(xvals, resilience_upa, '-m', label='UPA graph, m=' + str(average_out_degree))
    plt.legend(loc='upper right')
    plt.xlabel("Number of nodes removed")
    plt.ylabel("Largest size of connected components")

    ####################################################
    # Resilience under targeted attack order

    resilience_network = np.zeros(1 + num_nodes_total)
    resilience_er = np.zeros(1 + num_nodes_total)
    resilience_upa = np.zeros(1 + num_nodes_total)

    for dummy_idx in range(3):
        cloned_network = copy_graph(network_graph)
        resilience_network = resilience_network + \
                             np.array(bfs_visited.compute_resilience(cloned_network,
                                                                     fast_targeted_order(cloned_network)))
        cloned_er = copy_graph(ugraph_er)
        resilience_er = resilience_er + \
                        np.array(bfs_visited.compute_resilience(cloned_er, fast_targeted_order(cloned_er)))
        cloned_upa = copy_graph(ugraph_upa)
        resilience_upa = resilience_upa + \
                         np.array(bfs_visited.compute_resilience(cloned_upa, fast_targeted_order(cloned_upa)))

    resilience_network /= 3.0
    resilience_er /= 3.0
    resilience_upa /= 3.0

    plt.figure()
    plt.plot(xvals, resilience_network, '-b', label='network graph')
    plt.plot(xvals, resilience_er, '-r', label='er graph, p=' + str(er_prob))
    plt.plot(xvals, resilience_upa, '-m', label='UPA graph, m=' + str(average_out_degree))
    plt.legend(loc='upper right')
    plt.xlabel("Number of nodes removed")
    plt.ylabel("Largest size of connected components")
    plt.title("Graph resilience comparison under targeted attack")

    # Show all plots
    plt.show()
