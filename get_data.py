import xlrd
import pprint
import os
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def get_list_2d(sheet, start_row, end_row, start_col, end_col):
    """[start_row, end_row)*[start_col, end_col)の行列を取得"""
    return [list(map(int, sheet.row_values(row, start_col, end_col)))
            for row in range(start_row, end_row)]


def get_time(stay=0):
    """距離行列(time)を返す"""
    wb = xlrd.open_workbook('data/time.xlsx')
    sheet = wb.sheet_by_name('data_only')
    d = np.array(get_list_2d(sheet, 0, 25, 0, 25))
    return d + stay


def get_happiness(country):
    """幸福度を返す"""
    wb = xlrd.open_workbook('data/AHP.xlsx')
    sheet = wb.sheet_by_name(country)
    col = ord("R") - ord("A")
    happiness = [sheet.cell_value(row, col) for row in range(3, 28)]
    return np.array(happiness)


def get_spots():
    """都市を返す"""
    wb = xlrd.open_workbook('data/time.xlsx')
    sheet = wb.sheet_by_name('name_and_data')
    spots = [sheet.cell_value(row, 0) for row in range(6, 31)]
    return spots


if __name__ == '__main__':
    print(get_time())
    print(get_happiness("中国"))
    print(get_spots())
