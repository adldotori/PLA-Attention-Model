import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from scapy.all import rdpcap
import time
import csv

class NetworkFlowDataset(data.Dataset):
    def __init__(self, pcapname, csvname):
        super().__init__()
        self.pcap_data = rdpcap(osp.join('data',pcapname))
        self.label_data = pd.read_csv(osp.join('data', csvname), encoding='latin1')

    def __len__(self):
        return len(self.pcap_data)
    
    def __getitem__(self, idx):
        sample_raw = self.pcap_data[idx].load
        sample = np.frombuffer(sample_raw, dtype=np.uint8)
        
        label_raw = self.label_data[idx]['Label']
        return sample, label_raw

class NetworkFlowDataloader(object):
    def __init__(self, opt, dataset):
        super().__init__()
        use_cuda = not torch.cuda.is_available()
        kwargs = {'num_workers': opt.num_workers} if use_cuda else {}

        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=True, **kwargs)

        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-workers', type=int, default = 4)
    parser.add_argument('-e', '--epoch', type=int, default=10)
    parser.add_argument('-b', '--batch-size', type=int, default = 4)
    parser.add_argument('-d', '--display-step', type=int, default = 600)
    opt = parser.parse_args()

    start = time.time()
    dataset = NetworkFlowDataset('Thursday-WorkingHours.pcap', 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv')
    data_loader = NetworkFlowDataloader(opt, dataset)

    print('[+] Size of the dataset: %05d, dataloader: %03d' \
        % (len(dataset), len(data_loader.data_loader)))