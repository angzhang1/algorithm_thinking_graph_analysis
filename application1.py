"""
Provided code for Application portion of Module 1

Imports physics citation graph
"""

# general imports
import urllib2
import Project1_Graph as graph
import DPATrail as dpa
import matplotlib.pyplot as plt
import math

# Set timeout for CodeSkulptor if necessary
#import codeskulptor
#codeskulptor.set_timeout(20)


###################################
# Code for loading citation graph

CITATION_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_phys-cite.txt"


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


def normalized_in_degree_distribution(digraph):
    """

    :param digraph:
    :return: a dict of nomalized in degree distribution {degree, probability}
    """
    in_degree_distribution = graph.in_degree_distribution(digraph)
    sum_degrees = sum(in_degree_distribution.values())
    print sum_degrees
    for degree in in_degree_distribution.keys():
        in_degree_distribution[degree] /= (1.0 * sum_degrees)
    return in_degree_distribution


citation_graph = load_graph(CITATION_URL)

# Calculate average out degree
num_nodes = len(citation_graph)
num_edges = 0
for edgs in citation_graph.values():
    num_edges += len(edgs)

average_out_degree = int(math.ceil(num_edges / 1.0 / num_nodes))
# 12.703
# 27770
# 352768

print num_edges
print average_out_degree

############
# Generate DPA graph
trail = dpa.DPATrial(average_out_degree)

dpa_graph = graph.make_complete_graph(average_out_degree)

for i in xrange(average_out_degree, num_nodes):
    dpa_graph[i] = trail.run_trial(average_out_degree)

in_degree_distribution = normalized_in_degree_distribution(citation_graph)
distribution_dpa = normalized_in_degree_distribution(dpa_graph)

plt.figure()
plt.plot([math.log10(k) for k in in_degree_distribution.keys()[1:]],
         [math.log10(dist) for dist in in_degree_distribution.values()[1:]],"b.")
# plt.plot(in_degree_distribution.keys()[1:], in_degree_distribution.values()[1:])
plt.xlabel("log10(in_degree)")
plt.ylabel("log10(probability)")
plt.title("In-degree distribution of the citation graph")

# plt.figure()
# plt.plot(in_degree_distribution.keys(), in_degree_distribution.values(), "b.")

plt.figure()
plt.plot([math.log10(k) for k in distribution_dpa.keys()[1:]],
         [math.log10(dist) for dist in distribution_dpa.values()[1:]],"c.")
# plt.plot(in_degree_distribution.keys()[1:], in_degree_distribution.values()[1:])
plt.xlabel("log10(in_degree)")
plt.ylabel("log10(probability)")
plt.title("In-degree distribution of the DPA graph")

plt.show()
