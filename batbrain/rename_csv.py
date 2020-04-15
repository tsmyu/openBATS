import os
import sys
from natsort import natsorted
import glob

def main(csv_list):
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

    main(csv_list)
