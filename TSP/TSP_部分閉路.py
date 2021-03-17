# issue 壊れてる
"""
TSP:
頂点数Nの重み付き有向グラフの距離行列が与えられる.
頂点sからスタートしてk-1個の頂点をちょうど一度ずつめぐってtに行く経路のうち,
重みの総和が最小のものを求める.
ついでに経路復元もする.
"""

import networkx as nx
import matplotlib.pyplot as plt


def TSP(n: "頂点数", dist: "距離行列", s: "始点", t: "終点", length):
    # dist[i][j] = i->j の距離
    # length:通る頂点の数
    INF = 1_000_000_000

    # dp[S][v] = 集合Sの要素をすべて通って頂点vに行く最短経路長
    # prev[S][v] = 集合Sの要素をすべて通って頂点vに行ったとき, 直前に通った頂点
    dp = [[INF] * n for _ in range(1 << n)]
    prev = [[-1] * n for _ in range(1 << n)]
    dp[0][s] = 0

    for S in range(1 << n):
        for u in range(n):
            if S >> u & 1:
                continue
            for v in range(n):
                if dp[S][u] + dist[u][v] < dp[S + (1 << u)][v]:
                    dp[S + (1 << u)][v] = dp[S][u] + dist[u][v]
                    prev[S + (1 << u)][v] = u

    candidates = next_combination(n, length)
    d = INF
    use = 0

    for S in candidates:
        if dp[S][t] < d:
            d = dp[S][t]
            use = S

    if d == INF:
        return float("inf"), "There is no pathway that meets the requirements."

    # 経路復元
    path = [t]
    now, state = t, use
    while state:
        now = prev[state][now]
        state -= 1 << now
        path.append(now)
    path.reverse()

    path = " -> ".join(map(str, path))
    length = d
    return length, path


def next_combination(n, k):
    """{0, 1, 2, ..., n - 1}に含まれるサイズkの部分集合の列挙"""
    comb = (1 << k) - 1
    res = []
    while comb < 1 << n:
        res.append(comb)
        x = comb & -comb
        y = comb + x
        comb = ((comb & ~y) // x >> 1) | y
    return res


if __name__ == '__main__':
    INF = 1_000_000_000

    dist = [[INF, 2, INF, INF],
            [INF, INF, 3, 9],
            [1, INF, INF, 6],
            [INF, INF, 4, INF]]

    ans, path = TSP(4, dist, 0, 0, 3)
    print("経路長:", ans)
    print("経路:", path)

    dist = [[INF, 1, 1],
            [INF, INF, 1],
            [INF, INF, INF]]

    ans, path = TSP(3, dist, 0, 0, 3)
    print("経路長:", ans)
    print("経路:", path)

    dist = [[INF, 1, INF, 1],
            [INF, INF, INF, 2],
            [4, INF, INF, 6],
            [10, INF, 3, INF]]

    ans, path = TSP(4, dist, 1, 1, 3)
    print("経路長:", ans)
    print("経路:", path)
