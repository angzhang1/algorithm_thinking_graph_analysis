"""
Project 2
"""

from collections import deque


def bfs_visited(ugraph, start_node):
    """
    bfs on undirected graph
    :param ugraph: a undirected graph
    :param start_node:
    :return: the set of all visited node
    """
    to_visit = deque([start_node])
    visited = set([start_node])

    while to_visit:
        next_node = to_visit.pop()
        for neighbor in ugraph[next_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                to_visit.appendleft(neighbor)

    return visited


def cc_visited(ugraph):
    """
    Compute connected components of a undirected graph
    :param ugraph: input undirected graph as a dictionary
    :return: list of sets representing the connected components of the graph
    """
    remaining_nodes = set(ugraph.keys())
    connected_components = []

    while remaining_nodes:
        any_node = remaining_nodes.pop()
        all_visited = bfs_visited(ugraph, any_node)
        connected_components.append(all_visited)
        # remove visited nodes from remaining nodes set
        remaining_nodes -= all_visited

    return connected_components


def largest_cc_size(ugraph):
    """
    return the size of the largest connected components
    :param ugraph: undirected graph
    :return: an int: the size of the largest connected component
    """
    remaining_nodes = set(ugraph.keys())
    largest_size = 0

    while len(remaining_nodes) > largest_size:
        any_node = remaining_nodes.pop()
        all_visited = bfs_visited(ugraph, any_node)
        if len(all_visited) > largest_size:
            largest_size = len(all_visited)
        # remove visited nodes from remaining nodes set
        remaining_nodes -= all_visited

    return largest_size


def remove_node(ugraph, node):
    """
    Remove a node from the graph
    :param ugraph: will be modified if the node is in the graph
    :param node: the node to remove
    :return: none
    """
    if node not in ugraph.keys():
        return

    neighbors = ugraph[node]
    ugraph.pop(node)

    for neigh in neighbors:
        ugraph[neigh].remove(node)


def compute_resilience(ugraph, attack_order):
    """
    compute the resilience of the graph removing nodes in attack order
    :param ugraph:
    :param attack_order: the node to be removed in this order
    :return: list of the largest connected components after removing each node
    """
    resilience = [largest_cc_size(ugraph)]
    for attacked_node in attack_order:
        remove_node(ugraph, attacked_node)
        resilience.append(largest_cc_size(ugraph))

    return resilience


# EX_GRAPH1 = {0: {1, 4, 5}, 1: {2, 6}, 2: {3}, 3: {0}, 4: {1}, 5: {2}, 6: set()}
# remove_node(EX_GRAPH1, 6)
#
# print largest_cc_size(EX_GRAPH1)
#
# print EX_GRAPH1
#
# print compute_resilience(EX_GRAPH1, [0, 5, 6])
