
import os
import sys
import argparse
from typing_extensions import Required

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--n_Gors', type=int, required=True)
    parser.add_argument('--n_roles', type=int, required=True)
    parser.add_argument('--val_devide', type=int, default=10)
    parser.add_argument('--hmm_iter', type=int, default=500)
    parser.add_argument('--t_step', '--totalTimeSteps', type=int, default=80)
    parser.add_argument('--overlap', type=int, default=40)
    parser.add_argument('-k', '--k_nearest', type=int, default=0)
    parser.add_argument('--batchsize', type=int, required=True)
    parser.add_argument('--n_epoch', type=int, required=True)
    parser.add_argument('--attention', type=int, default=-1)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--model', tyoe=str, required=True)
    parser.add_argument('-ev_th', '--event_threshold', type=int, required=True, help='event with frames less than the threshold will be removed')
    parser.add_argument('--fs', tyoe=int, default=10)
    parser.add_argument('--body', action='store_true')
    parser.add_argument('--acc', tyoe=int, default=2)
    parser.add_argument('--vel_in', action='store_true')
    parser.add_argument('--in_out', action='store_true')
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--numProcess', tyoe=int, default=16)
    parser.add_argument('--TEST', action='store_true')
    parser.add_argument('--Sanity', action='store_true')
    parser.add_argument('--hard_only', action='srore_true')
    parser.add_argument('--wo_macro', action='store_true')
    parser.add_argument('--res', action='store_ture')
    parser.add_argument('--jrk', tyoe=float, defaut=0)
    parser.add_argument('--lam_acc', type=float, default=0)
    parser.add_argument('--pretrain', type=int, default=0)
    parser.add_argument('--pretrain2', type=int, default=0)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--drop_ind', action='store_true')
    args, _ = parser.parse_known_args()

    # directories
    main_dir = '../'
    game_dir = main_dir+'data_'+args.data+'/'
    Data = LoadData(main_dir, game_dir, args.data)
    path_init = './weights/'

    numProcess = args.numProcess
    os.environ["OMP_NUM_THREADS"] = str(numProcess)
    TEST = args.TEST

    # pre-process
    args.meanHMM = True # sorting sequences using meanHMM
    args.in_sma = True # small multi-agent data
    acc = args.acc # output: 0: vel, 1: pos+vel, 2:vel+acc, 3: pos+vel+acc
    args.vel_in = 1 if args.vel_in else 2 # input 1: vel 2: vel+acc
    if acc == -1:
        args.vel_in = -1 # position only
    elif acc == 0 or acc == 1:
        args.vel_in = 1
    vel_in = args.vel_in
    args.velocity = args.vel_in
    args.filter = True
    assert not (args.in_out and args.in_sma)
    assert not (args.vel_in == 1 and acc >= 2)
    


if __name__=="__main__":
    main()