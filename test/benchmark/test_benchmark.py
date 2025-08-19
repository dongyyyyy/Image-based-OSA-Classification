from torchvision import models
import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import random
import time

import sys
module_path = os.path.abspath(os.path.join('./'))
sys.path.append(module_path)


from model.Backbone.ResNet import *
from model.Backbone.Comparison_models import models_comparison_single_view

import logging

from torchsummary import summary
from torch.optim.lr_scheduler import LambdaLR
# from utils.dataloader import AHI_classification
from utils.dataloader import AHI_classification_folder
from torch.utils.data import DataLoader, SequentialSampler
import torch.optim as optim

from utils.Logger import Logger, Logger_filename

# from config import get_parser
from config.config_test import get_parser

from tqdm import tqdm

import math

logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def search_folders_name(dirname):
    filelist = []
    for (path, dir, files) in os.walk(dirname):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                if (path + '\\') not in filelist:
                    filelist.append("%s\\"%(path))
    return filelist

def search_folders(dirname):
    filelist = []
    for (path, dir, files) in os.walk(dirname):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                filelist.append("%s\\%s"%(path,filename))
    return filelist

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def read_file_list(file_path):
    with open(file_path, 'r') as f:
        file_list = f.read().splitlines()
    return file_list


if __name__ == '__main__':
    args = get_parser()
    # exit(1)
    if args.classification_mode == 'binary':
        args.num_classes = 2
    elif args.classification_mode in ['two_severe', 'normal_moderate_severe']:
        args.num_classes = 3
    elif args.classification_mode == 'normal_mild_moderate_severe':
        args.num_classes = 4
    else:
        raise ValueError("Unsupported classification mode.")
    
    set_seed(seed = args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    title = 'ResNet50-AHI'
    
    confusion_matrix = np.zeros((args.num_classes, args.num_classes))
    for fold in range(args.k_fold):
        print(f'start fold {fold + 1} / {args.k_fold}')

        # logger.warning( 
        #     f"Process rank: {args.local_rank}, "
        #     f"device: {args.device}, "
        #     f"n_gpu: {args.n_gpu}, "
        #     f"distributed training: {bool(args.local_rank != -1)}",)


        print(f'load weight fold {fold + 1}')
        
        model = models_comparison_single_view(model_architecture=args.backbone_architecture,num_classes=args.num_classes,pretrained=True,image_size=args.image_size)
        

        current_fold_path = args.out + f'/fold_{fold+1}/'

        model_weight_path = os.path.join(current_fold_path,'best_model.pth.tar')

        # model weight load
        model.load_state_dict(torch.load(model_weight_path,weights_only=True)['model_state_dict'])

        model = model.cuda()
        # summary(model, input_size=(3, 448, 448))

        test_file_path = os.path.join(current_fold_path,'test_filename.txt')
        print(f'test_file_path = {test_file_path}')
        # test_file_path = os.path.join(f'/data1/osh_sleep/AHI_classification/result_2024_06_19/Training_view_1/k_fold_10_num_fold_{fold}_seed_0_rotationDegree_10_img_size_448', 'test_filename.txt')
        image_validation_list = read_file_list(test_file_path)
        
        image_validation_list = [item.strip().replace('\t', '') for item in image_validation_list if not item.strip().startswith('filename')]
        
        validation_set = set(image_validation_list)

        print('len(image_validation_list) : ', len(image_validation_list))
    

        ##################################################
        # 이거 하나만써도 문제 없음
        # image_validation_list = [item.strip().replace('\t', '') for item in image_validation_list if not item.strip().startswith('filename')]
        
        # validation_set = set(image_validation_list)

        # print('len(image_validation_list) : ', len(image_validation_list))
        ##################################################
        # print(f'args.dataset_info_path = {args.dataset_info_path}')
        # test_file_path = os.path.join(args.dataset_info_path, f'seed_{args.seed}/k_fold_num_{args.k_fold}/fold_{fold+1}/test_filename.txt')
        # print(f'test_file_path = {test_file_path}')
        # image_validation_list = read_file_list(test_file_path)
        



        # image_validation_list = [item.strip().replace('\t', '') for item in image_validation_list if not item.strip().startswith('filename')]
        
        # image_validation_list = [item.strip().replace('/data/', '/home/oem/') for item in image_validation_list]


        # validation_set = set(image_validation_list)
        ##################################################
        data = pd.read_csv(args.dataset_path + 'Image_label/label.csv')
        label_list = data.values.tolist()
        data_info = ['index','name','id','cpap','sex','age','BMI','ESS','height','weight','AHI','ODI','Arrhythmia ','expert']

        image_validation_list_with_label = []
                
                
        
        for index in range(len(image_validation_list)):
            path_parts = image_validation_list[index].strip().split('/')
            last_part = path_parts[-2] if len(path_parts) > 1 else None

            if last_part:
                name_parts = last_part.split(' ')
                if len(name_parts) == 3:
                    name = name_parts[1]
                    file_index = int(name_parts[2])
                elif len(name_parts) == 4:
                    name = name_parts[1]
                    file_index = int(name_parts[3])

                if label_list[file_index][1] == name:
                    if args.classification_mode == 'binary':
                        if int(label_list[file_index][10]) >= 30:
                            AHI_count = 1
                        else:
                            AHI_count = 0
                        
                    elif args.classification_mode == 'two_severe':
                        if int(label_list[file_index][10]) >= args.severe_range:
                            AHI_count = 2
                        elif int(label_list[file_index][10]) >= 30:
                            AHI_count = 1
                        else:
                            AHI_count = 0    
                        
                    elif args.classification_mode == 'normal_moderate_severe':
                        if int(label_list[file_index][10]) >= 30:
                            AHI_count = 2
                        elif int(label_list[file_index][10]) >= 15:
                            AHI_count = 1
                        else:
                            AHI_count = 0   
                        
                    elif args.classification_mode == 'normal_mild_moderate_severe':
                        if int(label_list[file_index][10]) >= 30:
                            AHI_count = 3
                        elif int(label_list[file_index][10]) >= 15:
                            AHI_count = 2
                        elif int(label_list[file_index][10]) >= 7:
                            AHI_count = 1
                        else:
                            AHI_count = 0   
                        
                    image_validation_list_with_label.append([image_validation_list[index],AHI_count])
                
                # print('image_validation_list_with_label : ', image_validation_list_with_label[:5])

        val_dataset = AHI_classification_folder(args=args, dataset_list = image_validation_list_with_label, image_view = args.image_view,  size=args.image_size,is_train=False)

        print('len(image_validation_list) : ', len(image_validation_list))
        
        validationloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # num_workers=0,
        shuffle=False,
        drop_last=False)

        print('len(val_dataset.label_list) : ', len(val_dataset.label_list))
    


        start_time = time.time()
        model.eval()
        
        val_total_loss = 0.
        val_total_count = 0
        val_total_data = 0

        
        with tqdm(validationloader,desc='Val',unit='batch') as tepoch:
            for index,(batch_image, batch_label,_) in enumerate(tepoch):
                # batch_image = batch_image.cuda()
                # batch_label = batch_label.cuda()
                batch_image = batch_image.cuda()
                
                
                batch_label = batch_label.cuda()
                
                with torch.no_grad():
                    # logit_output = model(batch_image)
                    logit_output = model(batch_image)
                    
                    
                    _,predict = torch.max(logit_output,1)
                    
                    for i in range(len(predict)):
                        confusion_matrix[batch_label[i]][predict[i]] += 1
                
                    check_count = (predict == batch_label).sum().item()
                    
                    val_total_count += check_count
                    # val_total_data += len(batch_image)
                    val_total_data += len(batch_label)
                    accuracy = val_total_count / val_total_data
                    tepoch.set_postfix(accuracy=100.*accuracy)

        val_accuracy = val_total_count / val_total_data * 100


        output_str = 'val dataset : %d/%d epochs spend time : %.4f sec  / correct : %d/%d -> %.4f%%\n' \
                    % (fold + 1, args.k_fold, time.time() - start_time,
                        val_total_count, val_total_data, val_accuracy)
        sys.stdout.write(output_str)
            
    print(confusion_matrix)

    accuracy = np.diag(confusion_matrix,k=0).sum() / np.sum(confusion_matrix)

    confusion_percent = np.zeros((args.num_classes, args.num_classes))
    confusion_percent_recall = np.zeros((args.num_classes, args.num_classes))
    sensitivity = np.zeros(args.num_classes)
    specificity = np.zeros(args.num_classes)
    recall = np.zeros(args.num_classes)
    precision = np.zeros(args.num_classes)
    
    tp = np.zeros(args.num_classes)
    tn = np.zeros(args.num_classes)
    tp_fp = confusion_matrix.sum(axis=0) # X, 1
    tp_fn = confusion_matrix.sum(axis=1) # 1, X
    tn_fp = np.zeros(args.num_classes)

    for i in range(args.num_classes):
        for z in range(args.num_classes):
            if z != i:
                tn_fp[i] += tp_fn[z]
            else:
                tp[i] += confusion_matrix[i][i]
    for index in range(args.num_classes):
        confusion_percent[:,index] = confusion_matrix[:,index] / tp_fp[index]
        confusion_percent_recall[index,:] = confusion_matrix[index,:] / tp_fn[index]
        tn[index] = confusion_matrix.sum() - confusion_matrix.sum(axis=0)[index] - confusion_matrix.sum(axis=1)[index] + confusion_matrix[index][index]
        sensitivity[index] = confusion_matrix[index][index] / float(tp_fn[index])
        specificity[index] = tn[index] / float(tn[index]+confusion_matrix.sum(axis=1)[index]-tp[index])
        
        precision[index] = confusion_matrix[index][index] / float(tp_fp[index])

        recall = sensitivity
        
    f1_score = 2 * (precision * recall) / (precision + recall)
    p_a = tp.sum() / confusion_matrix.sum()
    p_c = 0
    for i in range(args.num_classes):
        p_c += (confusion_matrix.sum(axis=0)[i] / confusion_matrix.sum() * confusion_matrix.sum(axis=1)[i] / confusion_matrix.sum())
    kappa = (p_a-p_c) / (1-p_c)
    
    balanced_accuracy = 0.
    for index in range(len(recall)):
        balanced_accuracy += recall[index]
    
    balanced_accuracy /= len(recall)
    
    confusion_percent = np.round(np.array(confusion_percent),4)
    confusion_percent_recall = np.round(np.array(confusion_percent_recall),4)
    
    print(f'Acc ==> {accuracy*100:.2f} // Balanced Acc. ==> {balanced_accuracy*100:.2f}')
    # print(f'Precision ==> {confusion_percent}')
    # print(f'Recall ==> {confusion_percent_recall}')
    
    precision = np.round(np.array(precision),4)
    recall = np.round(np.array(recall),4)
    f1_score = np.round(np.array(f1_score),4)
    
    print('='*100)
    print('precision : ', precision)
    print('recall : ', recall)
    print('f1_score : ', f1_score)