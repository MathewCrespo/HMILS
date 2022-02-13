from PIL import Image
import torch
import os
import sys
from torch.utils.data import Dataset
from torchvision import transforms
import xml.etree.cElementTree as ET
from tqdm import tqdm
import random
import pandas as pd
from random import randint,sample
import cv2
import selectivesearch

'''
DirectBags -- do selective search in advance
Two tasks available: Benign and Malignant classifcation and axillary lymph node metastasis (ALNM)
'''

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

def is_xml_file(filename):
    return filename.endswith('xml')

class HieBags(Dataset):
    '''
    This Dataset is for doing benign/maglignant classification with US or SWE or two modal together

    Args:
        root: (str) data root
        sub_list: (int) sub path list [0, 1, 2, 3, 4] indicating which subset to use
        pre_transform: (torchvision.transform) the transform used
        modality: (int) indicate which modality to use. 0: US 1: SWE 2: US & SWE
    '''
    def __init__(self, root, pre_transform=None, sub_list = [0,1,2,3,4], task = 'BM'):
        self.root = root
        self.table = pd.read_excel(os.path.join(self.root,'label_info.xlsx'))
        self.img_list = [] ## a list of tuple (grey_filename, swe_filename)
        self.pre_transform = pre_transform
        self.patient_dict = [] ## a dict of information: {name: {label:, images: {grey:, pure:}}} 
        self.label_list = []


        self.sub_list = sub_list
        self.task = task
        for i in self.sub_list:
            self.new_scan(i)    
        #print(self.patient_dict)

    def parse_img_info(self, img_info_path):
        f = open(img_info_path,'r')
        line = f.readline().strip('\n')
        img_w = int(line.split(',')[0])
        img_h = int(line.split(',')[1])
        f.close()
        return img_w, img_h
    
    def parse_ins_info(self,log_path):
        f = open(log_path,'r')
        line = f.readline().strip('\n')
        x = int(line.split(',')[0])
        y = int(line.split(',')[1])
        w = int(line.split(',')[2])
        h = int(line.split(',')[3])
        f.close()
        return x,y,w,h

    def __getitem__(self, idx):  # for a single patient
        ### start from patient_dict and create a dict
        #print(idx)
        now_patient = self.patient_dict[idx]
        
        label = now_patient['label']
        grey_img_path = now_patient['images']
        grey_imgs = []
        img_info = []
        idx_list = [0]
        idx_temp = 0
        path_list = []
        for path in grey_img_path:   # different ultrasound image in a single patient
            path_list.append(path)
            img_path = path.split('.')[0]
            ins_list = os.listdir(img_path)
            lesion_list = []
            pos_list = []
            region_num = 0
            img_info_path = os.path.join(img_path,'original.txt')
            img_w, img_h = self.parse_img_info(img_info_path)            
            for ins in ins_list:     # different lesion area in a single ultrasound image
                if is_image_file(ins):
                    region_num  += 1
                    ins_info = ins.split('.')[0]+'.txt'
                    ins_path = os.path.join(img_path,ins)
                    log_path = os.path.join(img_path,ins_info)
                    # read image 
                    ins_img = Image.open(ins_path)
                    ins_img = self.pre_transform(ins_img)
                    lesion_list.append(ins_img)
                    # parse bbox info
                    x,y,w,h = self.parse_ins_info(log_path)
                    pos_info = torch.Tensor([x,y,w,h,img_w,img_h])
                    pos_list.append(pos_info)
                    
            ins_stack = torch.stack([x for x in lesion_list], dim=0)
            pos_stack = torch.stack([x for x in pos_list], dim = 0)

            grey_imgs.append(ins_stack)
            img_info.append(pos_stack)
            idx_temp += region_num
            idx_list.append(idx_temp)
        bag_imgs = torch.cat([x for x in grey_imgs], dim=0)
        bag_pos = torch.cat([x for x in img_info], dim=0)
        return bag_imgs, bag_pos, label, idx_list #, path_list


    def new_scan(self,fold):
        fold_table = self.table[self.table['{}_fold'.format(self.task)]==fold].reset_index(drop=True)
        #print(fold_table)
        for k in range(len(fold_table)):
            
            id = fold_table.loc[k,'name']
            p_path = os.path.join(self.root,str(id))
            p_label = fold_table.loc[k,self.task]
            now_patient = {}
            now_patient['label'] = p_label
            img_dir = p_path+'/grey'
            now_patient['images'] = [os.path.join(img_dir,x) for x in os.listdir(img_dir) if is_image_file(x)]
            self.patient_dict.append(now_patient)
            

    def scan(self, now_root):
        # 1- malignant  0-benign
        self.M_path = os.path.join(now_root, "Malignant")
        self.B_path = os.path.join(now_root, "Benign")
        # number of M
        ##scan benign path
        idx = 0
        for path in [self.B_path, self.M_path]:
            for patient_dir in os.listdir(path):
                patient_path = os.path.join(path, patient_dir)
                if os.path.isdir(patient_path):
                    ##assign labels
                    patient_info = {}
                    if path == self.M_path:
                        label = torch.Tensor([1, self.get_ALNM(patient_path)])
                    else:
                        label = torch.Tensor([0, 0])
                    
                    patient_info['dir'] = patient_path
                    patient_info['label'] = label
                    patient_info['images'] = {"grey": [], "pure": []}
                    grey_path = os.path.join(patient_path, "grey")
                    swe_path = os.path.join(patient_path, "pure")
                    ##scan grey folder as the reference (assume that grey file exists for sure)
                    for file_name in os.listdir(grey_path):
                        if is_image_file(file_name):
                            grey_file = os.path.join(grey_path, file_name)
                            swe_file = os.path.join(swe_path, file_name)
                            self.label_list.append(label)
                            
                            if os.path.exists(swe_file):
                                self.img_list.append([grey_file, swe_file])
                                patient_info["images"]["grey"].append(grey_file)
                                patient_info["images"]["pure"].append(swe_file)
                            else:
                                self.img_list.append([grey_file, None])
                                patient_info["images"]["grey"].append(grey_file)
                                patient_info["images"]["pure"].append(None)
                    ##update patient dict
                    self.patient_dict.append(patient_info)
                    idx += 1 
    
    def __len__(self):
        return len(self.patient_dict)

##Test Code
if __name__=="__main__":

    root =  '/media/hhy/data/USdata/MergePhase1/BUSSH_final'
    pre = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    ])

    trainset = HieBags(root, pre_transform=pre,sub_list=[0],task='BM')
    m = 0
    b = 0
    for i in range(len(trainset)):
        img,pos, label,idx,path = trainset[i]
        if label == 1:
            m+=1
        else :
            b+=1
    print(m)
    print(b)
    '''
    print(trainset[0][0].shape)
    print(trainset[0][1].shape)
    print(trainset[0][2])
    print(trainset[0][3])
    
    bag_state = []
    
    # statistics
    for i in range(len(trainset)):
        bag_label = trainset[i][1].int().tolist()
        bag_state.append(bag_label)
    maglinant_num_train = bag_state.count(1)

    testset = PatientBags(root+'/test', pre_transform=pre)
    bag_state = []
    for i in range(len(testset)):
        bag_label = testset[i][1].int().tolist()
        bag_state.append(bag_label)
    maglinant_num_test = bag_state.count(1)
    print('{} maglinant bags out of {} in trainset'.format(maglinant_num_train, len(trainset)))
    #print('{} maglinant bags out of {} in testset'.format(maglinant_num_test, len(testset)))
    '''

    
    

