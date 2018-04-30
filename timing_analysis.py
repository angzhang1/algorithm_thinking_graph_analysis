import time
import application2 as ap
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x_nodes = range(10, 1000, 10)
    y_slow = []
    y_fast = []

    for num_nodes in x_nodes:
        ugraph = ap.upa_graph(num_nodes, 5)

        start = time.time()
        ap.targeted_order(ugraph)
        end = time.time()

        y_slow.append((end - start) * 1000)

        start = time.time()
        ap.fast_targeted_order(ugraph)
        end = time.time()

        y_fast.append((end - start) * 1000)

    plt.plot(x_nodes, y_slow, '-b', label='slow targeted order')
    plt.plot(x_nodes, y_fast, '-r', label='fast targeted order')
    plt.legend(loc='upper left')
    plt.xlabel("Number of nodes in the undirected graph")
    plt.ylabel("Running time in millisecond")
    plt.title("Desktop Python 2.7, OSX, 2.5 GHz Intel Core i5")
    plt.show()
