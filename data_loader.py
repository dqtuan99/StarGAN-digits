import torch
from torchvision import datasets
from torchvision import transforms
import numpy as np
import os

# def get_loader(ds_path='./data', batch_size=128, image_size=32):

    # transform = transforms.Compose([transforms.Resize([image_size, image_size]),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize([0.5], [0.5])])

#     ds = datasets.ImageFolder(ds_path, transform=transform)

#     ds_loader = torch.utils.data.DataLoader(dataset=ds,
#                                               batch_size=batch_size,
#                                               shuffle=True,
#                                               num_workers=4,
#                                               drop_last=True)
#     return ds_loader


MNIST_path = './dataset/MNIST_train.pt'
MNISTM_path = './dataset/MNISTM_train.pt'
SVHN_path = './dataset/SVHN_train.pt'
SynDigits_path = './dataset/SynDigits_train.pt'
USPS_path = './dataset/USPS_train.pt'


class ImgDomainAdaptationData(torch.utils.data.Dataset):
    def __init__(self, w, h):
        
        self.transform = transforms.Compose([transforms.Resize([w, h]),
                                            transforms.Normalize([0.5], [0.5])])
            
        self.MNIST_data = torch.load(MNIST_path)
        self.MNISTM_data = torch.load(MNISTM_path)
        self.SVHN_data = torch.load(SVHN_path)
        self.SynDigits_data = torch.load(SynDigits_path)
        self.USPS_data = torch.load(USPS_path)
        
        self.MNIST_img = self.transform(self.MNIST_data[0])
        self.MNISTM_img = self.transform(self.MNISTM_data[0])
        self.SVHN_img = self.transform(self.SVHN_data[0])
        self.SynDigits_img = self.transform(self.SynDigits_data[0])
        self.USPS_img = self.transform(self.USPS_data[0])
        
        self.MNIST_label = self.MNIST_data[1]
        self.MNISTM_label = self.MNISTM_data[1]
        self.SVHN_label = self.SVHN_data[1]
        self.SynDigits_label = self.SynDigits_data[1]
        self.USPS_label = self.USPS_data[1]
        
        self.MNIST_img, self.MNIST_domain = self.pre_processing(self.MNIST_img, 0)
        self.MNISTM_img, self.MNISTM_domain = self.pre_processing(self.MNISTM_img, 1)
        self.SVHN_img, self.SVHN_domain = self.pre_processing(self.SVHN_img, 2)
        self.SynDigits_img, self.SynDigits_domain = self.pre_processing(self.SynDigits_img, 3)
        self.USPS_img, self.USPS_domain = self.pre_processing(self.USPS_img, 4)
        
        self.img = torch.vstack((self.MNIST_img, 
                                  self.MNISTM_img, 
                                  self.SVHN_img,
                                  self.SynDigits_img,
                                  self.USPS_img))
        
        self.label = torch.hstack((self.MNIST_label, 
                                    self.MNISTM_label,
                                    self.SVHN_label,
                                    self.SynDigits_label,
                                    self.USPS_label))
                                 
        self.domain = np.hstack((self.MNIST_domain, 
                                    self.MNISTM_domain, 
                                    self.SVHN_domain,
                                    self.SynDigits_domain,
                                    self.USPS_domain))
        
    
    def pre_processing(self, img, domain):
        num_img = img.shape[0]
        
        if len(img.shape) < 4:
            img = img.unsqueeze(1).repeat(1, 3, 1, 1)
        
        domain_label = np.zeros(num_img, dtype=int) + domain
        
        return img, domain_label
        
    def __len__(self):
        return self.label.shape[0]
    
    def __getitem__(self, index):
        return self.img[index], self.label[index].item(), self.domain[index]