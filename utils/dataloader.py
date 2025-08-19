import torch
import torch.utils.data as data
import os

# for using Albumentations library!
# pip install -U albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from .randaugment import RandAugmentMC

from torchvision import transforms

# ing...
# For single-view classification
class AHI_classification_folder(object):
    def read_dataset(self):
        image_extensions = ['.jpg', '.jpeg', '.jfif', '.pjpeg', '.pjp','.png', '.avif', '.gif'] 

        all_images_files = []
        all_labels = []

        for index in range(len(self.dataset_list)):
            dataset_path = self.dataset_list[index][0]
            image_list = os.listdir(dataset_path)
            for image_name in image_list:
                ext = os.path.splitext(image_name)[-1]
                if ext in image_extensions: 
                    if self.image_view == -1: 
                        all_images_files.append(dataset_path+image_name)
                        all_labels.append(int(self.dataset_list[index][1]))
                    else: 
                        if image_name.split('.')[0] in self.image_view:
                            all_images_files.append(dataset_path+image_name)
                            all_labels.append(int(self.dataset_list[index][1]))
                else:
                    print(dataset_path)
        return all_images_files, all_labels, len(all_images_files)

    def __init__(self,args, dataset_list, image_view=-1,size=224,is_train=False):
        self.dataset_list = dataset_list
        super(AHI_classification_folder, self).__init__()
        self.image_view = image_view
        image, label, total_len = self.read_dataset()
        
        if is_train: # 학습용 loader의 경우
            self.transform = A.Compose(
            [
                A.Resize(height=size,width=size), 
                A.HorizontalFlip(p=0.5), 
                A.Rotate(limit=args.rotate_degree,p=0.8), 
                A.CoarseDropout(always_apply=False, p=args.CoarseDropout_prob, max_holes=args.CoarseDropout_max_holes, max_height=args.CoarseDropout_max_height,
                                max_width=args.CoarseDropout_max_width, min_holes=args.CoarseDropout_min_holes, min_height=args.CoarseDropout_min_height, min_width=args.CoarseDropout_min_width),
                A.ColorJitter(brightness=(0,0.2), contrast=(0,0.2), saturation=(0,0.2), hue=(0,0.2),p=0.8), 
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
                ToTensorV2(),
            ]
            )
        else: # validation과 test용
            self.transform = A.Compose(
                [
                    A.Resize(height=size,width=size),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        self.image_list, self.label_list, self.total_len = image,label,total_len
        
        for index in range(len(self.image_list)):
            if self.image_list[index].split('/')[-1].split('.')[0].isalpha():
                print(self.image_list[index].split('/')[-1].split('.')[0])
       
        
    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index]) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image = self.transform(image=image)["image"]
        
        return image, self.label_list[index], int(self.image_list[index].split('/')[-1].split('.')[0])
        
    def __len__(self):
        return len(self.image_list)
    
    
    
# Current main dataloader function for loading train, validation(test)
# For multi-view image classification 
class AHI_multi(object): 
    def read_dataset(self):
        image_extensions = ['.jpg', '.jpeg', '.jfif', '.pjpeg', '.pjp','.png', '.avif', '.gif']
        # Using view index (1:Frontal | 2:Frontal with chin up | 3:Side view | 4:Mouth)
        all_images_files = []
        all_labels = []

        for index in range(len(self.folder_list)):
            dataset_path = self.folder_list[index]

            if isinstance(dataset_path, list):
                dataset_path = dataset_path[0]

            # To check the image inside the folder (To confirm which image view will be used)
            image_list = os.listdir(dataset_path)
            
            # For checking count 
            # if you want to use 2 image the return count will be 2
            # if the count is not correct, that patient will be not used.
            count = 0 
            
            for image_name in image_list:
                ext = os.path.splitext(image_name)[-1]

                if ext in image_extensions:
                    if os.path.splitext(image_name)[0] in self.image_view:
                        count += 1
            if count == 2: # will be used
                all_images_files.append(self.folder_list[index])
                all_labels.append(int(self.label_list[index][1]))
            
        return all_images_files, all_labels, len(all_images_files)


    # initialization
    def __init__(self,args, folder_list, folder_label, image_view=[1,2], size=224, is_train=False):
        super(AHI_multi, self).__init__()
        self.image_view = image_view # -1
        
        if is_train:
            self.transform = A.Compose(
            [
                A.Resize(height=size,width=size), 
                A.HorizontalFlip(p=0.5), #
                A.Rotate(limit=args.rotate_degree,p=0.8),
                A.CoarseDropout(always_apply=False, p=args.CoarseDropout_prob, max_holes=args.CoarseDropout_max_holes, max_height=args.CoarseDropout_max_height,
                                max_width=args.CoarseDropout_max_width, min_holes=args.CoarseDropout_min_holes, min_height=args.CoarseDropout_min_height, min_width=args.CoarseDropout_min_width), #
                A.ColorJitter(brightness=(0,0.2), contrast=(0,0.2), saturation=(0,0.2), hue=(0,0.2),p=0.8), 
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(height=size,width=size),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        self.folder_list, self.label_list = folder_list, folder_label
        
        self.all_folder_list, self.all_folder_label, self.length = self.read_dataset()
        
    def __getitem__(self, index):
        directory_path = self.all_folder_list[index]

        if isinstance(directory_path, list):
            directory_path = directory_path[0] # path information
        
        file_list = os.listdir(directory_path) # view information

        if len(self.image_view) == 2:
            for filename in file_list:
                if filename.split('.')[0] == self.image_view[0]:
                    image_1_filename = filename
                if filename.split('.')[0] == self.image_view[1]:
                    image_2_filename = filename
        
        # read image values
        image_1_path = os.path.join(directory_path, image_1_filename)
        image_1 = cv2.imread(image_1_path)

        # BGR to RGB
        image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
        # applying the data augmentation on image of view 1
        image_1 = self.transform(image=image_1)["image"]

        # read image values
        image_2_path = os.path.join(directory_path, image_2_filename)
        image_2 = cv2.imread(image_2_path)
        
        # BGR to RGB
        image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
        # applying the data augmentation on image of view 2
        image_2 = self.transform(image=image_2)["image"]

        return (image_1, image_2), self.all_folder_label[index], index

    def __len__(self):
        return len(self.all_folder_list)

class AHI_multi_three_inputs(object):
    def read_dataset(self):
        image_extensions = ['.jpg', '.jpeg', '.jfif', '.pjpeg', '.pjp','.png', '.avif', '.gif']
        # Using view index (1:Frontal | 2:Frontal with chin up | 3:Side view | 4:Mouth)
        all_images_files = []
        all_labels = []

        for index in range(len(self.folder_list)):
            dataset_path = self.folder_list[index]

            if isinstance(dataset_path, list):
                dataset_path = dataset_path[0]

            # To check the image inside the folder (To confirm which image view will be used)
            image_list = os.listdir(dataset_path)
            
            # For checking count 
            # if you want to use 2 image the return count will be 2
            # if the count is not correct, that patient will be not used.
            count = 0 
            
            for image_name in image_list:
                ext = os.path.splitext(image_name)[-1]

                if ext in image_extensions:
                    if os.path.splitext(image_name)[0] in self.image_view:
                        count += 1
            if count == 3: # will be used
                all_images_files.append(self.folder_list[index])
                all_labels.append(int(self.label_list[index][1]))
            
        return all_images_files, all_labels, len(all_images_files)


    # initialization
    def __init__(self,args, folder_list, folder_label, image_view=[1,2,3], size=224, is_train=False):
        super(AHI_multi_three_inputs, self).__init__()
        self.image_view = image_view # -1
        
        if is_train:
            self.transform = A.Compose(
            [
                A.Resize(height=size,width=size), 
                A.HorizontalFlip(p=0.5), 
                A.Rotate(limit=args.rotate_degree,p=0.8), 
                A.CoarseDropout(always_apply=False, p=args.CoarseDropout_prob, max_holes=args.CoarseDropout_max_holes, max_height=args.CoarseDropout_max_height,
                                max_width=args.CoarseDropout_max_width, min_holes=args.CoarseDropout_min_holes, min_height=args.CoarseDropout_min_height, min_width=args.CoarseDropout_min_width), 
                A.ColorJitter(brightness=(0,0.2), contrast=(0,0.2), saturation=(0,0.2), hue=(0,0.2),p=0.8), # color jittering
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Normalization
                ToTensorV2(),
            ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(height=size,width=size),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        self.folder_list, self.label_list = folder_list, folder_label
        
        self.all_folder_list, self.all_folder_label, self.length = self.read_dataset()
        
    def __getitem__(self, index):
        directory_path = self.all_folder_list[index]

        if isinstance(directory_path, list):
            directory_path = directory_path[0] # path information
        
        file_list = os.listdir(directory_path) # view information

        if len(self.image_view) == 3:
            for filename in file_list:
                if filename.split('.')[0] == self.image_view[0]:
                    image_1_filename = filename
                if filename.split('.')[0] == self.image_view[1]:
                    image_2_filename = filename
                if filename.split('.')[0] == self.image_view[2]:
                    image_3_filename = filename
        
        # read image values
        image_1_path = os.path.join(directory_path, image_1_filename)
        image_1 = cv2.imread(image_1_path)

        # BGR to RGB
        image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
        # applying the data augmentation on image of view 1
        image_1 = self.transform(image=image_1)["image"]

        # read image values
        image_2_path = os.path.join(directory_path, image_2_filename)
        image_2 = cv2.imread(image_2_path)
        
        # BGR to RGB
        image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
        # applying the data augmentation on image of view 2
        image_2 = self.transform(image=image_2)["image"]

        # read image values
        image_3_path = os.path.join(directory_path, image_3_filename)
        image_3 = cv2.imread(image_3_path)
        
        # BGR to RGB
        image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2RGB)
        # applying the data augmentation on image of view 2
        image_3 = self.transform(image=image_3)["image"]

        return (image_1, image_2, image_3), self.all_folder_label[index], index

    def __len__(self):
        return len(self.all_folder_list)

    
    