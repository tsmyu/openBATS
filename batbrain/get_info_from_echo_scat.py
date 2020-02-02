import os
import sys
import glob
import numpy as np
from scipy import signal
import itertools
import math
import re
import csv

from cross_correlation import DataField as CrossCorrDataField
from scat import DataField as ScatDataField


def main(csv_list, emit_csv_list):
    """
    main for detect point from echo
    """
    for idx, csv in enumerate(csv_list):
        print("analyze target:{}".format(csv))
        # cross_corr_field = CrossCorrDataField(csv, emit_csv)
        # cross_corr_field.preprocessing()
        # cross_corr_field.get_info()
        # cross_corr_field.save()
        scat_field = ScatDataField(csv, emit_csv_list)
        emit_spike_list, echo_right_spike_list, echo_left_spike_list = scat_field.calc_cochlear_block()
        scat_field.calc_temporal_block(emit_spike_list, echo_right_spike_list, echo_left_spike_list)

        print("-------finish:{}th/{}-----------".format(idx + 1, len(csv_list)))


if __name__ == "__main__":
    argvs = sys.argv
    if len(argvs) < 2:
        print(
            "Usage: python {} [folder of csv] [emit_pulse csv]".format(argvs[0]))
        exit()
    csv_list = sorted(glob.glob("{}/*.csv".format(argvs[1])))
    if not os.path.exists("./corr_{}".format(os.path.basename(argvs[1]))):
        os.makedirs("./corr_{}".format(os.path.basename(argvs[1])))
    if not os.path.exists("./echo_point_{}".format(os.path.basename(argvs[1]))):
        os.makedirs("./echo_point_{}".format(os.path.basename(argvs[1])))
    print("csv_list:{}".format(csv_list))
    emit_csv_list = sorted(glob.glob("{}/*.csv".format(argvs[2])))
    main(csv_list, emit_csv_list)
