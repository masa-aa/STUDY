# issue 壊れてる
"""
TSP:
頂点数Nの重み付き有向グラフの距離行列が与えられる.
頂点sからスタートしてk-1個の頂点をちょうど一度ずつめぐってtに行く経路のうち,
重みの総和が最小のものを求める.
ついでに経路復元もする.
"""
from get_data import get_spots


def TSP(n: "頂点数", dist: "距離行列"):
    # dist[i][j] = i->j の距離
    # length:通る頂点の数
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

    # 経路復元
    path = [n - 1]
    now, state = n - 1, (1 << n) - 1
    while state > 1:
        state -= 1 << now
        now = prev[state][now]
        path.append(now)

    path.reverse()

    return path


def convert(dist, nodes):
    new_dist = [[0] * len(nodes) for _ in range(len(nodes))]
    new_n = len(nodes)
    for i, e1 in enumerate(nodes):
        for j, e2 in enumerate(nodes):
            new_dist[i][j] = dist[e1][e2]
    tsp = TSP(new_n, new_dist)
    spots = get_spots()
    return tuple(map(lambda x: spots[nodes[x]], tsp))


if __name__ == '__main__':
    from get_data import get_time, get_spots
    d = get_time(stay=30)
    spots = get_spots()
    routes = (12, 2, 22, 4, 10, 6)
    new_routes = convert(d, routes)
    print(list(map(lambda x: spots[x], routes)))
    print(new_routes)
