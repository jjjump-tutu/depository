import argparse
import os
import numpy as np
#python3 train.py --dataset_dir '/media/hypo/Hypo/physionet_org_train' --dataset_name cc2018 --signal_name 'C4-M1' --sample_num 10 --model_name lstm --batchsize 64 --network_save_freq 5 --epochs 20 --lr 0.0005 --BID 5_95_th --select_sleep_time --cross_validation subject
# python3 train.py --dataset_dir './datasets/sleep-edfx/' --dataset_name sleep-edfx --signal_name 'EEG Fpz-Cz' --sample_num 10 --model_name lstm --batchsize 64 --network_save_freq 5 --epochs 20 --lr 0.0005 --BID 5_95_th --select_sleep_time --cross_validation subject

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # self.parser.add_argument('--dataset_dir', type=str, default='./datasets/sleep-edfx/',
        #                         help='your dataset path')
        # self.parser.add_argument('--dataset_name', type=str, default='sleep-edfx',help='Choose dataset sleep-edfx | sleep-edf | cc2018')
        # self.parser.add_argument('--model_name', type=str, default='lstm',help='Choose model  lstm | multi_scale_resnet_1d | resnet18 |...')
        self.parser.add_argument('-d', '--dataset', type=str, default='1.npz')
        self.parser.add_argument('-te', '--train_epoch', type=int, default=70)
        self.parser.add_argument('-re', '--rebalanced_epoch', type=int, default=70)
        self.parser.add_argument('-f', '--frozen', default=False)
        self.parser.add_argument('-rm', '--rebalanced_mode', type=str,default='avg')
        self.parser.add_argument('-r', '--rebalanced', default=False)

        self.initialized = True

    def getparse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt

