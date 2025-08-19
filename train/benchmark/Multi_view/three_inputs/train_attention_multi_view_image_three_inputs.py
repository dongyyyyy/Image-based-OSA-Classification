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

from model.Backbone.Comparison_models import models_comparison_single_view

from model.Attention.backbone_load import backbone_without_GAP
from model.Attention.three_inputs.Attention import TF_Attention, MHCA_Attention, MHDA_Attention

import logging

from torchsummary import summary
from torch.optim.lr_scheduler import LambdaLR
# from utils.dataloader import AHI_classification
from utils.dataloader import AHI_multi_three_inputs
from torch.utils.data import DataLoader, SequentialSampler
import torch.optim as optim

from utils.Logger import Logger, Logger_filename
from utils.function import *


# from config import get_parser
from config.Multi_view.config import get_parser

from tqdm import tqdm

import math

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    args = get_parser()
    if args.classification_mode == 'binary':
        args.num_classes = 2
    elif args.classification_mode in ['two_severe', 'normal_moderate_severe']:
        args.num_classes = 3
    elif args.classification_mode == 'normal_mild_moderate_severe':
        args.num_classes = 4
    else:
        raise ValueError("Unsupported classification mode.")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.makedirs(args.out,exist_ok=True)    
    # print(len(args.gpu_id.split(',')))
    # exit(1)
    title = 'ResNet50-AHI'
    
    
    for fold in range(args.k_fold):
        set_seed(args = args)
        print(f'start fold {fold + 1} / {args.k_fold}')
        if args.pretrained_model_weight_path != 'None':
            fold_dir = os.path.join(args.out, f'fold_{fold + 1}')
        else:
            fold_dir = os.path.join(args.out,'from_scratch', f'fold_{fold + 1}')
        os.makedirs(fold_dir, exist_ok=True)
        print(f'fold_dir = {fold_dir}')
        args.logger = Logger(os.path.join(fold_dir, 'log.txt'), title=title)
        args.logger.set_names(['Train Acc','Validation Acc', 'Best Top1 Acc', 'Best epoch'])

        args.logger_train = Logger_filename(os.path.join(fold_dir, 'train_filename.txt'), title=title)
        args.logger_train.set_names(['filename'])
        
        args.logger_test = Logger_filename(os.path.join(fold_dir, 'test_filename.txt'), title=title)
        args.logger_test.set_names(['filename'])
        
        args.logger_intersection = Logger_filename(os.path.join(fold_dir, 'intersection_filename.txt'), title=title)
        args.logger_intersection.set_names(['filename'])
            
        streamHandler = logging.StreamHandler()
        fileHandler = logging.FileHandler(os.path.join(fold_dir, 'information.log'),mode='w')
        logger.addHandler(streamHandler)
        logger.addHandler(fileHandler)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO)



        logger.info(dict(args._get_kwargs()))
        print(dict(args._get_kwargs()))
        # image_view=args.image_view.split(',')
        print(f'load weight fold {fold + 1}')
        if args.pretrained_model_weight_path != 'None':
            model_weight_path_1 = f'{args.pretrained_model_weight_path}/image_view_{args.image_view[0]}/seed_{args.seed}/fold_{fold + 1}/best_model.pth.tar'
            model_weight_path_2 = f'{args.pretrained_model_weight_path}/image_view_{args.image_view[1]}/seed_{args.seed}/fold_{fold + 1}/best_model.pth.tar'
            model_weight_path_3 = f'{args.pretrained_model_weight_path}/image_view_{args.image_view[2]}/seed_{args.seed}/fold_{fold + 1}/best_model.pth.tar'
        else:
            model_weight_path_1=None
            model_weight_path_2=None
            model_weight_path_3=None
        
        backbone1, output_dim1 = backbone_without_GAP(model_architecture=args.backbone_architecture,weight_files=model_weight_path_1,pretrained=True,num_classes=args.num_classes)
        backbone2, output_dim2 = backbone_without_GAP(model_architecture=args.backbone_architecture,weight_files=model_weight_path_2,pretrained=True,num_classes=args.num_classes)
        backbone3, output_dim3 = backbone_without_GAP(model_architecture=args.backbone_architecture,weight_files=model_weight_path_3,pretrained=True,num_classes=args.num_classes)
        
        num_patches = args.image_size // 32 * args.image_size // 32
        if args.MHDA:
            print('Training MHDA!!')
            model = MHDA_Attention(backbone1=backbone1, backbone2=backbone2,backbone3=backbone3, backbone_freezing=args.backbone_freezing,shared_attention=args.shared_attention,
                                    backbone_output_dim=output_dim1,emb_size=args.emb_size, num_patches=num_patches,depth=args.depth,
                                    positional_embedding=args.positional_embedding,cls_token=args.cls_token,last_gelu=args.last_gelu,
                                    heads=args.num_heads,dropout=args.dropout,emb_dropout=args.emb_dropout, class_num=args.num_classes
                                    ,last_mlps=args.last_mlps)
        else:
            if args.cross_attention:
                print('Training MHCA!!')
                model = MHCA_Attention(backbone1=backbone1, backbone2=backbone2,backbone3=backbone3, backbone_freezing=args.backbone_freezing,shared_attention=args.shared_attention,
                                    backbone_output_dim=output_dim1,emb_size=args.emb_size, num_patches=num_patches,depth=args.depth,
                                    positional_embedding=args.positional_embedding,cls_token=args.cls_token,last_gelu=args.last_gelu,
                                    heads=args.num_heads,dropout=args.dropout,emb_dropout=args.emb_dropout, class_num=args.num_classes)
            else:
                print('Training MHSA!!')
                model = TF_Attention(backbone1=backbone1, backbone2=backbone2,backbone3=backbone3, backbone_freezing=args.backbone_freezing,shared_attention=args.shared_attention,
                                    backbone_output_dim=output_dim1,emb_size=args.emb_size, num_patches=num_patches,depth=args.depth,
                                    positional_embedding=args.positional_embedding,cls_token=args.cls_token,last_gelu=args.last_gelu,
                                    heads=args.num_heads,dropout=args.dropout,emb_dropout=args.emb_dropout, class_num=args.num_classes)
        
        # backbone1 = nn.DataParallel(model.backbone1.cuda(0))
        if len(args.gpu_id.split(',')) == 0:
            model = model.cuda()
        else:
            # model.backbone1 = nn.DataParallel(model.backbone1.cuda())
            # model.backbone2 = nn.DataParallel(model.backbone2.cuda())
            model = nn.DataParallel(model.cuda())

        # print('model : ', model)
        
        # summary(model, input_size=[(3, 448, 448), (3, 448, 448)])


        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, nesterov=args.nesterov)
        criterion = nn.CrossEntropyLoss()
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 0, args.epochs)

        best_accuracy = 0
        best_epoch = 0
        print(f'args.dataset_info_path = {args.dataset_info_path}')
        train_file_path = os.path.join(args.dataset_info_path, f'seed_{args.seed}/k_fold_num_{args.k_fold}/fold_{fold+1}/train_filename.txt')
        test_file_path = os.path.join(args.dataset_info_path, f'seed_{args.seed}/k_fold_num_{args.k_fold}/fold_{fold+1}/test_filename.txt')
        print(f'train_file_path = {train_file_path}')
        print(f'test_file_path = {test_file_path}')
        image_train_list = read_file_list(train_file_path)
        image_validation_list = read_file_list(test_file_path)
        
        
        image_train_list = [item.strip().replace('\t', '') for item in image_train_list if not item.strip().startswith('filename')]
        image_validation_list = [item.strip().replace('\t', '') for item in image_validation_list if not item.strip().startswith('filename')]
        
        train_set = set(image_train_list)
        validation_set = set(image_validation_list)
        inter = train_set.intersection(validation_set)
    
        print('len(image_train_list) : ', len(image_train_list))
        print('len(image_validation_list) : ', len(image_validation_list))
        print('len(inter) : ', len(inter))
    
        
        for filename in image_train_list:
            args.logger_train.append([filename])
            
        for filename in image_validation_list:
            args.logger_test.append([filename])
            
        for filename in inter:
            args.logger_intersection.append([filename])
            
        args.logger_train.close()
        args.logger_test.close()

        data = pd.read_csv(args.dataset_path + 'Image_label/label.csv')
        label_list = data.values.tolist()
        data_info = ['index','name','id','cpap','sex','age','BMI','ESS','height','weight','AHI','ODI','Arrhythmia ','expert']

        image_train_list_with_label = []
        image_validation_list_with_label = []


        for index in range(len(image_train_list)):
            path_parts = image_train_list[index].strip().split('/')
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
                        args.num_classes = 2
                    elif args.classification_mode == 'two_severe':
                        if int(label_list[file_index][10]) >= args.severe_range:
                            AHI_count = 2
                        elif int(label_list[file_index][10]) >= 30:
                            AHI_count = 1
                        else:
                            AHI_count = 0    
                        args.num_classes = 3
                    elif args.classification_mode == 'normal_moderate_severe':
                        if int(label_list[file_index][10]) >= 30:
                            AHI_count = 2
                        elif int(label_list[file_index][10]) >= 15:
                            AHI_count = 1
                        else:
                            AHI_count = 0   
                        args.num_classes = 3
                    elif args.classification_mode == 'normal_mild_moderate_severe':
                        if int(label_list[file_index][10]) >= 30:
                            AHI_count = 3
                        elif int(label_list[file_index][10]) >= 15:
                            AHI_count = 2
                        elif int(label_list[file_index][10]) >= 7:
                            AHI_count = 1
                        else:
                            AHI_count = 0   
                        args.num_classes = 4  
                    # 3class classification
                    
                    
                    image_train_list_with_label.append([image_train_list[index],AHI_count])
                
                
        
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

        train_dataset = AHI_multi_three_inputs(args=args, folder_list = image_train_list_with_label, folder_label = image_train_list_with_label, image_view = args.image_view, size=args.image_size,is_train=True)
        val_dataset = AHI_multi_three_inputs(args=args, folder_list = image_validation_list_with_label, folder_label = image_validation_list_with_label, image_view = args.image_view,  size=args.image_size,is_train=False)
        
        trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # num_workers=0,
        shuffle=True,
        drop_last=True)

        validationloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # num_workers=0,
        shuffle=False,
        drop_last=False)
    

        print('len(train_dataset.label_list) : ', len(train_dataset.label_list))
        print('len(val_dataset.label_list) : ', len(val_dataset.label_list))
    

        for epoch in range(0, args.epochs):
            train_total_loss = 0.
            train_total_count = 0.
            train_total_data = 0
            batch_image = 0
            
            model.train()
            
            output_str = 'current epoch : %d/%d / current_lr : %f \n' % (epoch+1,args.epochs,optimizer.state_dict()['param_groups'][0]['lr'])
            logger.info(output_str)
            
            start_time = time.time()
            
            with tqdm(trainloader,desc='Train',unit='batch') as tepoch:
                for index,(batch_image, batch_label,_) in enumerate(tepoch):
                    # batch_image = batch_image.cuda()
                    
                    batch_image_1 = batch_image[0].cuda()
                    batch_image_2 = batch_image[1].cuda()
                    batch_image_3 = batch_image[2].cuda()
                    

                    # print(f'batch_image_1 shape = {batch_image_1.shape}')
                    # print(f'batch_image_2 shape = {batch_image_2.shape}')
                    # print(f'batch_image_3 shape = {batch_image_3.shape}')
                    batch_label = batch_label.cuda()
                    optimizer.zero_grad()
                    logit_output = model(batch_image_1,batch_image_2,batch_image_3)
                
                    loss = criterion(logit_output, batch_label)
                    
                    _,predict = torch.max(logit_output, 1)
                    
                    check_count = (predict == batch_label).sum().item()
                    
                    train_total_loss += loss.item()
                    train_total_count += check_count
                    # train_total_data += len(batch_image)
                    train_total_data += len(batch_label)
                    
                    
                    loss.backward()
                    optimizer.step()
                    model.zero_grad()
                    
                    accuracy = train_total_count / train_total_data
                    tepoch.set_postfix(loss=train_total_loss/(index+1),accuracy=100.*accuracy)
            
            train_total_loss /= index
            train_accuracy = train_total_count / train_total_data * 100
                        
            output_str = 'train dataset : %d/%d epochs spend time : %.4f sec / total_loss : %.4f correct : %d/%d -> %.4f%%\n' \
                        % (epoch + 1, args.epochs, time.time() - start_time, train_total_loss,
                            train_total_count, train_total_data, train_accuracy)
            # sys.stdout.write(output_str)
            logger.info(output_str)
            scheduler.step()
            start_time = time.time()
            model.eval()
            
            val_total_loss = 0.
            val_total_count = 0
            val_total_data = 0

            confusion_matrix = np.zeros((args.num_classes, args.num_classes))
            
            with tqdm(validationloader,desc='Val',unit='batch') as tepoch:
                for index,(batch_image, batch_label,_) in enumerate(tepoch):
                    # batch_image = batch_image.cuda()
                    # batch_label = batch_label.cuda()
                    batch_image_1 = batch_image[0].cuda()
                    batch_image_2 = batch_image[1].cuda()
                    batch_image_3 = batch_image[2].cuda()
                    
                    batch_label = batch_label.cuda()
                    

                    
                    with torch.no_grad():
                        # logit_output = model(batch_image)
                        logit_output = model(batch_image_1,batch_image_2,batch_image_3)
                        
                        loss = criterion(logit_output,batch_label)
                        
                        _,predict = torch.max(logit_output,1)
                        
                        for i in range(len(predict)):
                            confusion_matrix[batch_label[i]][predict[i]] += 1
                    
                        check_count = (predict == batch_label).sum().item()
                        
                        val_total_loss += loss.item()
                        val_total_count += check_count
                        # val_total_data += len(batch_image)
                        val_total_data += len(batch_label)
                        accuracy = val_total_count / val_total_data
                        tepoch.set_postfix(loss=val_total_loss/(index+1),accuracy=100.*accuracy)

            val_total_loss /= index
            val_accuracy = val_total_count / val_total_data * 100

            print('validation_confusion_matrix : ', confusion_matrix)

            output_str = 'val dataset : %d/%d epochs spend time : %.4f sec  / total_loss : %.4f correct : %d/%d -> %.4f%%\n' \
                        % (epoch + 1, args.epochs, time.time() - start_time, val_total_loss,
                            val_total_count, val_total_data, val_accuracy)
            # sys.stdout.write(output_str)
            logger.info(output_str)
        
            
            save_file = os.path.join(fold_dir, 'best_model.pth.tar')
            if epoch == 0:
                best_accuracy = val_accuracy
                best_epoch = epoch
                torch.save({
                    'model_state_dict': model.state_dict() if len(args.gpu_id.split(',')) == 0 else model.module.state_dict(),
                    'epoch' : epoch,
                    'optimizer_sate_dict': optimizer.state_dict(),
                    'best_accuracy': best_accuracy,
                    'learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
                    'scheduler': scheduler.state_dict()
                }, save_file)
                stop_count = 0
            else:
                if best_accuracy < val_accuracy:
                    best_accuracy = val_accuracy
                    best_epoch = epoch
                    stop_count = 0
                    torch.save({
                        'model_state_dict': model.state_dict() if len(args.gpu_id.split(',')) == 0 else model.module.state_dict(),
                        'epoch' : epoch,
                        'optimizer_sate_dict': optimizer.state_dict(),
                        'best_accuracy': best_accuracy,
                        'learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
                        'scheduler': scheduler.state_dict()
                    }, save_file)
                else:
                    stop_count += 1

            args.logger.append([train_accuracy, val_accuracy, best_accuracy,best_epoch])


        output_str = 'best epoch : %d/%d / best accuracy : %f%%\n' \
                    % (best_epoch + 1, args.epochs, best_accuracy)
        sys.stdout.write(output_str)
        logger.info(output_str)
        print('=' * 30)
        
        streamHandler.close()
        fileHandler.close()
        logger.removeHandler(streamHandler)
        logger.removeHandler(fileHandler)