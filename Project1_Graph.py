"""
Project 1
"""
#
# EX_GRAPH0 = {0: {1, 2}, 1: set(), 2: set()}
#
# EX_GRAPH1 = {0: {1, 4, 5}, 1: {2, 6}, 2: {3}, 3: {0}, 4: {1}, 5: {2}, 6: set()}
#
# EX_GRAPH2 = {0: {1, 4, 5}, 1: {2, 6}, 2: {3, 7}, 3: {7}, 4: {1}, 5: {2}, 6: set(), 7: {3}, 8: {1, 2}, 9: {0, 3, 4, 5, 6, 7}}


def make_complete_graph(num_nodes):
    """
    :param num_nodes:
    :return: dictionary to represent a complete graph
    """
    if num_nodes == 0:
        return {}

    return {i: set([x for x in range(num_nodes) if x is not i]) for i in xrange(num_nodes)}


def compute_in_degrees(digraph):
    """
    Compute the in degrees of a diagraph
    :param digraph: a dictionary representing directed graph
    :return: a dictionary which is the in degrees
    """
    # Initialize as in-degrees of 0 for all nodes
    in_degrees = {i: 0 for i in digraph.keys()}

    for edge_lists in digraph.values():
        for vert in edge_lists:
            in_degrees[vert] += 1

    return in_degrees


def in_degree_distribution(digraph):
    """
    Compute the in degree distribution of a DAG
    :param digraph:
    :return:
    """
    in_degree = compute_in_degrees(digraph)
    distribution = {}

    for degree in in_degree.values():
        if distribution.has_key(degree):
            distribution[degree] += 1
        else:
            distribution[degree] = 1

    return distribution
