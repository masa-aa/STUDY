import xlrd
import pprint
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from math import sqrt, pi, cos, sin, acos


def distance(From, To):
    """東経と北緯が与えられて距離を返す."""
    x0, y0 = From
    x1, y1 = To
    x0 *= pi / 180
    x1 *= pi / 180
    y = pi * (y1 - y0) / 180
    return 6378137 * acos(sin(x0) * sin(x1) + cos(x0) * cos(x1) * cos(y))


def get_list_2d(sheet, start_row, end_row, start_col, end_col):
    """[start_row, end_row)*[start_col, end_col)の行列を取得"""
    return [list(map(int, sheet.row_values(row, start_col, end_col)))
            for row in range(start_row, end_row)]


def get_time(*go, stay=0, edge_limit=6):
    """距離行列(time)を返す"""
    """
    　stay:滞在時間
    　edge_limit:自身を含む辺をつなぐ数
    """
    wb = xlrd.open_workbook('data/time.xlsx')
    sheet = wb.sheet_by_name('data_only')
    d = np.array(get_list_2d(sheet, 0, 25, 0, 25))
    d += stay
    spots = get_spots()
    go = {spots.index(spot) for spot in go}
    compress = np.argsort(np.argsort(d))
    for i in range(25):
        if i != 12:
            d[i][11] = 100_000_000

    for i in range(25):
        for j in range(25):
            if compress[i][j] > edge_limit and i not in go and j not in go:
                d[i][j] = 100_000_000
    return d


def get_happiness(country, *go):
    """幸福度を返す"""
    """warning:各幸福度が100以上になってはならない"""
    wb = xlrd.open_workbook('data/AHP.xlsx')
    sheet = wb.sheet_by_name(country)
    col = ord("K") - ord("A")
    spots = get_spots()
    happiness = np.array([sheet.cell_value(row, col) for row in range(3, 28)]) * 100
    s = 2 * happiness.max()
    for spot in go:
        happiness[spots.index(spot)] = s
    return np.array(happiness)


def get_spots():
    """都市を返す"""
    wb = xlrd.open_workbook('data/time.xlsx')
    sheet = wb.sheet_by_name('name_and_data')
    spots = [sheet.cell_value(row, 0) for row in range(6, 31)]
    return spots


def get_distance():
    """緯度, 経度情報から距離行列を返す"""
    pos = \
        {'清水寺': (34.994856, 135.785046),
         '二条城': (35.01423, 135.748218),
         '伏見稲荷': (34.96714, 135.772672),
         '金閣寺': (35.03937, 135.729243),
         'ギオンコーナー': (35.001551, 135.775822),
         '嵐山': (35.009449, 135.666773),
         '祇園': (35.003782, 135.777245),
         '八坂神社': (35.003656, 135.778553),
         '京都御所': (35.025414, 135.762125),
         '銀閣寺': (35.027021, 135.798206),
         '錦市場': (35.005008, 135.764902),
         '京都タワー': (34.987531, 135.759324),
         '京都駅': (34.985849, 135.758767),
         '龍安寺': (35.034494, 135.718263),
         '伏見': (34.932416, 135.771056),
         '東寺': (34.980598, 135.747786),
         '高台寺': (35.00051, 135.781218),
         '南禅寺': (35.011414, 135.794484),
         '東福寺': (34.976064, 135.773777),
         '平安神宮': (35.015982, 135.782426),
         '嵐山モンキーパーク': (35.011408, 135.676206),
         '東山': (34.992396, 135.775797),
         '河原町': (35.002111, 135.769279),
         '三十三間堂': (34.987885, 135.771713),
         '下鴨神社': (35.038037, 135.772773)}
    pos = list(pos.values())
    dist = np.zeros((25, 25))
    for i, e in enumerate(pos):
        for j, f in enumerate(pos):
            dist[i, j] = distance(e, f)
    return dist


if __name__ == '__main__':
    # for v in get_time(edge_limit=10):
    #     print(" ".join(map(lambda x: str(x) if x < 10000 else "INF", v)))
    # print(get_time(edge_limit=10))
    print(get_happiness("中国", "清水寺"))
    # print(get_spots())
    # print(get_distance())
    # print(get_time("嵐山", stay=30)[5])
