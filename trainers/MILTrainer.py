#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Copyright (c) 2019 gyfastas
'''
from __future__ import absolute_import
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.utility as utility
from utils.logger import Logger
import argparse
from importlib import import_module
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score
# for debug
import torch.nn.functional as F
import utils.utility as utility
from utils.logger import Logger
import argparse
from importlib import import_module

class MILTrainer(object):
    def __init__(self, net, optimizer, lrsch, loss, train_loader, val_loader, logger, start_epoch,
                 save_interval=1):
        '''
        mode:   0: only single task--combine 
                1: multi task added, three losses are simply added together.
                2: Ldiff between two extracted features
        
        '''
        self.net = net
        self.optimizer = optimizer
        self.lrsch = lrsch
        self.loss = loss
        self.train_loader = train_loader
        self.test_loader = val_loader
        self.logger = logger
        self.logger.global_step = start_epoch
        self.save_interval = save_interval
            
    def train(self):
        self.net.eval()
        self.logger.update_step()
        train_loss = 0.
        prob = []
        pred = []
        target = []
        for data, img_info, label, idx_list in (tqdm(self.train_loader, ascii=True, ncols=60)):
            # reset gradients
            self.optimizer.zero_grad()
            data = data.cuda()
            bag_label = label.cuda()
            img_info = img_info.cuda()
            #idx_list = idx_list.cuda()
            prob_label, predicted_label, loss, weights = self.net(data, img_info, bag_label, idx_list)
            train_loss += loss.item()
            target.append(bag_label.cpu().detach().numpy().ravel()) # bag_label or label??
            pred.append(predicted_label.cpu().detach().numpy().ravel())
            prob.append(prob_label.cpu().detach().numpy().ravel())

            # backward pass
            loss.backward()
            # step
            self.optimizer.step()
            self.lrsch.step()

             # log
            '''
            target.append(bag_label.cpu().detach().float().tolist()[0])
            pred.append(predicted_label.cpu().detach().float().tolist()[0])
            prob.append(prob_label.cpu().detach().float().tolist()[0])
            '''
            
        # calculate loss and error for epoch
        '''
        print('target is {}'.format(target))
        print('pred is {}'.format(pred))
        print('prob is {}'.format(prob))
        '''
        train_loss /= len(self.train_loader)
        self.log_metric("Train", target, prob, pred)

        if not (self.logger.global_step % self.save_interval):
            self.logger.save(self.net, self.optimizer, self.lrsch, self.loss)

        #print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(self.epoch, train_loss.cpu().numpy()[0], train_error))


    def test(self):
        self.net.eval()
        test_loss = 0.
        target = []
        pred = []
        prob = []
        for data, img_info, label, idx_list in tqdm(self.test_loader, ascii=True, ncols = 60):
            data = data.cuda()
            bag_label = label.cuda()
            img_info = img_info.cuda()
        
            prob_label, predicted_label, loss, weights = self.net(data, img_info, bag_label,idx_list)
            test_loss += loss.item()
        # label or bag label?
            target.append(bag_label.cpu().detach().numpy().ravel())
            pred.append(predicted_label.cpu().detach().numpy().ravel())
            prob.append(prob_label.cpu().detach().numpy().ravel())

            
        '''
        test_error /= len(self.test_loader)
        test_loss /= len(self.test_loader)
        '''
        # target prob pred?
        self.log_metric("Test", target, prob, pred)

        

        
    def log_metric(self, prefix, target, prob, pred):
        pred_list = np.concatenate(pred)
        prob_list = np.concatenate(prob)
        target_list = np.concatenate(target)
        cls_report = classification_report(target_list, pred_list, output_dict=True, zero_division=0)
        acc = accuracy_score(target_list, pred_list)
        #print ('acc is {}'.format(acc))
        auc_score = roc_auc_score(target_list, prob_list)
        print('auc is {}'.format(auc_score))
        #print(cls_report)

        self.logger.log_scalar(prefix+'/'+'AUC', auc_score, print=True)
        self.logger.log_scalar(prefix+'/'+'Acc', acc, print= True)
        self.logger.log_scalar(prefix+'/'+'Malignant_precision', cls_report['1']['precision'], print= True)
        self.logger.log_scalar(prefix+'/'+'Benign_precision', cls_report['0']['precision'], print= True)
        self.logger.log_scalar(prefix+'/'+'Malignant_recall', cls_report['1']['recall'], print= True)
        self.logger.log_scalar(prefix+'/'+'Benign_recall', cls_report['0']['recall'], print= True)
        self.logger.log_scalar(prefix+'/'+'Malignant_F1', cls_report['1']['f1-score'], print= True)


        
        '''
        self.logger.log_scalar(prefix+'/'+'Accuracy', acc, print=True)
        self.logger.log_scalar(prefix+'/'+'Precision', cls_report['1.0']['precision'], print=True)
        self.logger.log_scalar(prefix+'/'+'Recall', cls_report['1.0']['recall'], print=True)
        self.logger.log_scalar(prefix+'/'+'F1', cls_report['1.0']['f1-score'], print=True)
        self.logger.log_scalar(prefix+'/'+'Specificity', cls_report['0.0']['recall'], print=True)
        '''
        

if __name__ == '__main__':
    # for debugging training function

    parser = argparse.ArgumentParser(description='Ultrasound CV Framework')
    parser.add_argument('--config',type=str,default='grey_SWE')

    args = parser.parse_args()
    configs = getattr(import_module('configs.'+args.config),'Config')()
    configs = configs.__dict__
    logger = configs['logger']
    logger.auto_backup('./')
    logger.backup_files([os.path.join('./configs',args.config+'.py')])
    trainer = configs['trainer']
    for epoch in range(logger.global_step, configs['epoch']):
        trainer.train()
        trainer.test()



