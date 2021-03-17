# issue 壊れてる
"""
TSP:
頂点数Nの重み付き有向グラフの距離行列が与えられる.
頂点0からスタートしてすべての頂点をちょうど一度ずつめぐって帰る閉路のうち,
重みの総和が最小のものを求める.
ついでに経路復元もする.
"""

import networkx as nx
import matplotlib.pyplot as plt


def TSP(n: "頂点数", dist: "距離行列"):
    # dist[i][j] = i->j の距離
    INF = 1_000_000_000

    # dp[S][v] = 集合Sの要素をすべて通って頂点vに行く最短経路長
    # prev[S][v] = 集合Sの要素をすべて通って頂点vに行ったとき, 直前に通った頂点
    dp = [[INF] * n for _ in range(1 << n)]
    prev = [[-1] * n for _ in range(1 << n)]
    dp[0][0] = 0

    for S in range(1 << n):
        for u in range(n):
            if S >> u & 1:
                continue
            for v in range(n):
                if dp[S][u] + dist[u][v] < dp[S + (1 << u)][v]:
                    dp[S + (1 << u)][v] = dp[S][u] + dist[u][v]
                    prev[S + (1 << u)][v] = u

    if dp[(1 << n) - 1][0] == INF:
        return float("inf"), "There is no pathway that meets the requirements."

    # 経路復元
    path = [0]
    now, state = 0, (1 << n) - 1
    while state:
        now = prev[state][now]
        state -= 1 << now
        path.append(now)
    path.reverse()

    path = " -> ".join(map(str, path))
    length = dp[(1 << n) - 1][0]
    return length, path


if __name__ == '__main__':
    INF = 1_000_000_000

    dist = [[INF, 2, INF, INF],
            [INF, INF, 3, 9],
            [1, INF, INF, 6],
            [INF, INF, 4, INF]]

    ans, path = TSP(4, dist)
    print("経路長:", ans)
    print("経路:", path)

    # # グラフの表示
    # G = nx.DiGraph()
    # G.add_nodes_from(range(4))
    # for i in range(4):
    #     for j in range(4):
    #         if dist[i][j] < INF:
    #             G.add_edge(i, j, weight=dist[i][j])

    # pos = nx.spring_layout(G, k=0.7)
    # edge_labels = {(i, j): w['weight'] for i, j, w in G.edges(data=True)}
    # print(edge_labels)
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    # nx.draw_networkx(G, pos, with_labels=True)
    # plt.show()

    dist = [[INF, 1, 1],
            [INF, INF, 1],
            [INF, INF, INF]]

    ans, path = TSP(3, dist)
    print("経路長:", ans)
    print("経路:", path)
