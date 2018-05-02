"""
Implement weighted union find with path compression
"""

import bfs_visited as bfs
import application2 as ap


class UnionFind(object):
    """
    Weighted union find class with path compression
    For computing the largest connect component's size efficiently
    """

    def __init__(self, graph):
        """
        Constructor
        :param n:
        :return:
        """
        self.id = {i: -1 for i in graph.keys()}
        self.sz = {i: 0 for i in graph.keys()}
        self.largest_cc_size = 0
        self.count = 0

    def add_node(self, p):
        """
        Add a node to the set
        :param p:
        :return:
        """
        if self.id[p] == -1:
            self.id[p] = p
            self.sz[p] = 1
            self.count += 1
            if self.largest_cc_size == 0:
                self.largest_cc_size = 1

    def resilience(self):
        return self.largest_cc_size

    def is_connected(self, p, q):
        """

        :param p:
        :param q:
        :return: true if connected
        """
        return self.find(p) == self.find(q)

    def find(self, p):
        """

        :param p:
        :return: void
        """
        while self.id[p] != p:
            self.id[p] = self.id[self.id[p]]
            p = self.id[p]
        return self.id[p]

    def union(self, p, q):
        """
        Union node p and q
        :param p:
        :param q:
        :return:
        """
        # If the node is not added, yet, return
        if self.id[p] == -1 or self.id[q] == -1:
            return

        i = self.find(p)
        j = self.find(q)
        if i != j:
            if self.sz[i] < self.sz[j]:
                self.id[i] = j
                self.sz[j] += self.sz[i]
                if self.sz[j] > self.largest_cc_size:
                    self.largest_cc_size = self.sz[j]
            else:
                self.id[j] = i
                self.sz[i] += self.sz[j]
                if self.sz[i] > self.largest_cc_size:
                    self.largest_cc_size = self.sz[i]
            self.count -= 1


def compute_resilience_uf_full_list(ugraph, attack_order):
    """
    resilience under attack order using union find algorithm
    :param ugraph:
    :param attack_order: the attack order with lengh equal to the number of nodes or ugraph
    :return: list of largest cc size with 0 .. n nodes removed
    """
    resilience = [0]
    uf = UnionFind(ugraph)
    for attacked_node in reversed(attack_order):
        uf.add_node(attacked_node)
        for neighbor in ugraph[attacked_node]:
            uf.union(attacked_node, neighbor)
        resilience.append(uf.resilience())

    resilience.reverse()
    return resilience


def compute_resilience_uf(ugraph, attack_order):
    """

    :param ugraph:
    :param attack_order:
    :return:
    """
    # used in the return statement
    len_attack = len(attack_order)

    if len(ugraph) != len(attack_order):
        attack_set = set(attack_order)
        for node in ugraph:
            if node not in attack_set:
                attack_order.append(node)

    # your union find implementation
    sizes = compute_resilience_uf_full_list(ugraph, attack_order)

    # sizes is a list of CC sizes
    return sizes[:len_attack + 1]

if __name__ == '__main__':
    # Unit test
    GRAPH4 = {0: set([1, 2, 3, 4]),
              1: set([0]),
              2: set([0]),
              3: set([0]),
              4: set([0]),
              5: set([6, 7]),
              6: set([5]),
              7: set([5])}

    print len(GRAPH4)

    attack = ap.random_order(GRAPH4, 2)
    attack = []
    print attack
    print compute_resilience_uf(GRAPH4, attack)
    print bfs.compute_resilience(GRAPH4, attack)
