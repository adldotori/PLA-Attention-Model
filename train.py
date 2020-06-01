import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.optim import lr_scheduler

from model import *
from loss import *
from dataloader import *

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-workers', type=int, default = 4)
    parser.add_argument('-e', '--epoch', type=int, default=10)
    parser.add_argument('-b', '--batch-size', type=int, default = 128)
    parser.add_argument('-t', '--test-batch-size', type=int, default = 1)
    parser.add_argument('-d', '--display-step', type=int, default = 600)
    opt = parser.parse_args()
    return opt

def train(opt):
    model = PLA_Attention_Model(100, 100, 50).cuda()
    # model.load_state_dict(torch.load('checkpoint.pt'))
    model.train()

    train_dataset = NetworkFlowDataset('Thursday-WorkingHours.pcap', 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv')
    train_data_loader = NetworkFlowDataloader(opt, train_dataset)

    test_dataset = NetworkFlowDataset('Friday-WorkingHours.pcap', 'Friday-WorkingHours-Morning.pcap_ISCX.csv')
    test_data_loader = NetworkFlowDataloader(opt, test_dataset)

    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = Loss()

    writer = SummaryWriter()

    for epoch in range(opt.epoch):
        for i in range(len(train_data_loader.data_loader)):
            step = epoch * len(train_data_loader.data_loader) + i + 1

            # load data
            flow, label = train_data_loader.next_batch()
            flow = flow.cuda()
            label = label.cuda()

            # train model
            optim.zero_grad()
            result = model(flow)
            loss = criterion(result, label)
            loss.backward()
            optim.step()

            writer.add_scalar('loss', loss, step)
            
            writer.close()

            if step % opt.display_step == 0:
                _, predicted = torch.max(result, 1)
                total = label.size(0)
                correct = (predicted == label).sum().item()
                total_test = 0
                correct_test = 0
                for i in range(len(test_data_loader.data_loader)):
                    flow, label = test_data_loader.next_batch()
                    flow = flow.cuda()
                    label = label.cuda()
                    result = model(flow)
                    _, predicted = torch.max(result, 1)
                    total_test += label.size(0)
                    correct_test += (predicted == label).sum().item()
                print('[Epoch {}] Loss : {:.2}, train_acc : {:.2}, test_acc : {:.2}'.format(epoch, loss, correct/total, correct_test/total_test))
        
        torch.save(model.state_dict(), 'checkpoint.pt')

if __name__ == '__main__':
    opt = get_opt()
    train(opt)