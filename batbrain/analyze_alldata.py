import os
import sys
import numpy as np
from natsort import natsorted
import glob
import pandas as pd
import matplotlib.pyplot as plt


def read_excel(target_csv):
    print("target_csv:", os.path.basename(target_csv))
    data = pd.ExcelFile(target_csv)
    input_sheet_name = data.sheet_names
    print(input_sheet_name)
    sheet_df = data.parse('data_all', skiprows=2)

    return sheet_df

def read_echo_data(csv_list):
    


def main(csv_list, all_data):
    sheet_df = read_excel(all_data)
    echo_list = read_echo_data(csv_list)
    bat_list = []
    for i in range(7):
        j = i * 7
        bat_list.append(np.array(sheet_df.iloc[:, [j+1, j+2, j+3, j+4, j+5, j+6, j+7]]))
    # batD_1st = np.array(sheet_df.iloc[:, [1, 2, 3, 4, 5, 6, 7]])
    # batF_1st = np.array(sheet_df.iloc[:, [8, 9, 10, 11, 12, 13, 14]])
    # batG_1st = np.array(sheet_df.iloc[:, [15, 16, 17, 18, 19, 20, 21]])
    # batH_1st = np.array(sheet_df.iloc[:, [22, 23, 24, 25, 26, 27, 28]])
    # batI_1st = np.array(sheet_df.iloc[:, [29, 30, 31, 32, 33, 34, 35]])
    # batMF_1st = np.array(sheet_df.iloc[:, [36, 37, 38, 39, 40, 41, 42]])
    # batMH_1st = np.array(sheet_df.iloc[:, [43, 44, 45, 46, 47, 48, 49]])
    plt.plot(batD_1st[:, 1], batD_1st[:, 5], marker='D', markersize=5,
             markeredgewidth=3, color="b", markeredgecolor="b", markerfacecolor="w")
    plt.plot(batD_1st[:, 1], batD_1st[:, 6], marker='D', markersize=5,
             markeredgewidth=3, color="g", markeredgecolor="g", markerfacecolor="w")
    plt.show()
    for csv_data in csv_list:
        splist_csv_data = csv_data.split("_")
        print(splist_csv_data)
        index = splist_csv_data[-3]
        if splist_csv_data[0].split("/")[-1] == "2100":
            index = splist_csv_data.pop(5)
            splist_csv_data.insert(-3, f"{int(index)+5000}")
        elif splist_csv_data[0].split("/")[-1] == "4100":
            index = splist_csv_data.pop(5)
            splist_csv_data.insert(-3, f"{int(index)+3000}")
        newname = "_".join(splist_csv_data)
        print(csv_data)
        print(newname)
        os.rename(csv_data, newname)


if __name__ == "__main__":
    argvs = sys.argv

    if len(argvs) < 2:
        print(
            "Usage: python {} [folder of echo csv] [data csv]".format(argvs[0]))
        exit()
    csv_list = natsorted(glob.glob("{}/*.csv".format(argvs[1])))
    all_data = argvs[2]

    main(csv_list, all_data)
