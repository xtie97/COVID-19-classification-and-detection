import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np
import torchvision.transforms.functional as TF
import glob
def adjust_contrast(image,c1):
    image = image.convert('L')
    image = np.array(image).astype(float)
    image_new = 255./(1+1*np.exp(-c1*(image-127.5)))
    image_new = Image.fromarray(image_new)
    image_new = image_new.convert('RGB')
    return image_new


class CV19DataSet(Dataset):
    def __init__(self, df, base_folder, transform):
        
        labels_pos = df.label_positive.tolist()
        labels_neg = df.label_negative.tolist()
        filenames = df.Filename.tolist()
        self.labels_pos = labels_pos
        self.labels_neg = labels_neg
        self.filenames = filenames
        self.transform = transform
        
        self.data_folder_covid = base_folder + 'Covid/'
        self.data_folder_noncovid = base_folder + 'NonCovid/'
        #self.covid_file = glob.glob(self.data_folder_covid+'*.png')
        #self.noncovid_file = glob.glob(self.data_folder_noncovid+'*.png')

    def __getitem__(self, index):
        label = [self.labels_pos[index], self.labels_neg[index]]
        fn = self.filenames[index]
        fn = fn[fn.find('Covid')+6:]
        
        if self.labels_pos[index] == 1:
            img = Image.open(self.data_folder_covid + fn).convert('RGB')
        else:
            img = Image.open(self.data_folder_noncovid + fn).convert('RGB')
        img = img.resize((224, 224), resample=Image.BILINEAR)
        
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.FloatTensor(label)
    
    def __len__(self):
        return len(self.filenames)


class CV19DataSet_crop(Dataset):
    def __init__(self, df, base_folder, transform, rfactor, cfactor, marker_size):
        
        labels_pos = df.label_positive.tolist()
        labels_neg = df.label_negative.tolist()
        filenames = df.Filename.tolist()
        resolution = df.resolution.tolist()
        contrast = df.contrast.tolist()
        marker = df.marker.tolist()
        self.rfactor = rfactor
        self.cfactor = cfactor
        self.marker_size = marker_size
        self.labels_pos = labels_pos
        self.labels_neg = labels_neg
        self.filenames = filenames
        self.transform = transform
        self.base_folder = base_folder
        self.resolution = resolution
        self.contrast = contrast
        self.marker = marker
        cols = ['x1', 'y1', 'x2', 'y2']
        self.coords = df[cols].values.tolist()
        self.roi_area = df['roi_area'].values.tolist()


    def __getitem__(self, index):
        label = [self.labels_pos[index], self.labels_neg[index]]
        fn = self.filenames[index]
        img = Image.open(self.base_folder + fn).convert('RGB')
        box = tuple(self.coords[index])
        img = img.crop(box)

        img = img.resize((224, 224), resample=Image.BILINEAR)
        
        if self.resolution[index] == 1:
            img = TF.adjust_sharpness(img,self.rfactor)
        if self.contrast[index] == 1:
            img = adjust_contrast(img,self.cfactor)
        if self.marker[index] == 1:
            img = np.array(img)
            r2 = random.randint(*random.choice([(5, 45), (180, 205)]))
            r1 = random.randint(0, 50)
            img[r1:r1+self.marker_size, r2:r2+self.marker_size] = 255
            img = Image.fromarray(img).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.FloatTensor(label)
    
    def __len__(self):
        return len(self.filenames)
