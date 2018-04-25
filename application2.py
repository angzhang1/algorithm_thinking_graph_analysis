"""
Provided code for Application portion of Module 2
"""

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
    graph_lines = graph_lines[ : -1]

    print "Loaded graph with", len(graph_lines), "nodes"

    answer_graph = {}
    for line in graph_lines:
        neighbors = line.split(' ')
        node = int(neighbors[0])
        answer_graph[node] = set([])
        for neighbor in neighbors[1 : -1]:
            answer_graph[node].add(int(neighbor))

    return answer_graph


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

# Generate the random graphs
network_graph = load_graph(NETWORK_URL)

num_nodes = len(network_graph.keys())
num_edges_network = num_edges(network_graph)

# Probability
er_prob = num_edges_network * 2.0 / (num_nodes * num_nodes)
ugraph_er = er_ugraph(num_nodes, er_prob)

ave_out_degree = num_edges_network / num_nodes

print ave_out_degree

trail = upa_trail.UPATrial(ave_out_degree)
upa_graph = simple_graph.make_complete_graph(ave_out_degree)

print upa_graph

for i in xrange(ave_out_degree, num_nodes):
    upa_graph[i] = trail.run_trial(ave_out_degree)
    for neigh in upa_graph[i]:
        upa_graph[neigh].add(i)

print num_edges(ugraph_er)
print len(ugraph_er.keys())

print num_edges(upa_graph)