# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:24:06 2024

@author: Tuan
"""


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Generator, Discriminator


class Net(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, conv_features=[32, 64]):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, conv_features[0], kernel_size=5)
        self.conv2 = nn.Conv2d(conv_features[0], conv_features[1], kernel_size=5)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        # Temporarily create a dummy input to pass through the conv layers to calculate output size
        dummy_input = torch.autograd.Variable(torch.zeros(1, in_channels, 28, 28))
        dummy_output = self._forward_conv(dummy_input)
        self.conv_output_size = dummy_output.view(-1).shape[0]

        self.fc1 = nn.Linear(self.conv_output_size, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def _forward_conv(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(-1, self.conv_output_size)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


MNIST_domain_id = 0
MNISTM_domain_id = 1
SVHN_domain_id = 2
SynDigits_domain_id = 3
USPS_domain_id = 4

MNIST_path = './dataset/MNIST_test.pt'
MNISTM_path = './dataset/MNISTM_train.pt'
SVHN_path = './dataset/SVHN_train.pt'
SynDigits_path = './dataset/SynDigits_train.pt'
USPS_path = './dataset/USPS_train.pt'

class ImgDomainAdaptationTestData(torch.utils.data.Dataset):
    def __init__(self, path, domain_id, w, h):

        self.transform = transforms.Compose([transforms.Resize([w, h]),
                                            transforms.Normalize([0.5], [0.5])])

        self.data = torch.load(path)

        self.img = self.transform(self.data[0])

        self.label = self.data[1]

        self.img, self.domain = self.pre_processing(self.img, domain_id)


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

BATCH_SIZE = 128

img_data = ImgDomainAdaptationTestData(USPS_path, USPS_domain_id, 32, 32)
ds_loader = torch.utils.data.DataLoader(img_data,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False,
                                        drop_last=True)
tests_bar = tqdm(ds_loader)


# Define the device to run the classifier on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



LOAD_MODEL = True
IMAGE_SIZE = 32
NUM_DOMAINS = 5

gen = Generator(num_domains=NUM_DOMAINS, image_size=IMAGE_SIZE)
# dis = Discriminator(num_domains=NUM_DOMAINS, image_size=IMAGE_SIZE)

model_path = './model'

if LOAD_MODEL:
    gen.load_state_dict(torch.load(os.path.join(model_path, 'gen.pkl')))
    # dis.load_state_dict(torch.load(os.path.join(model_path, 'dis.pkl')))

gen.to(device)
gen.eval()
# dis.to(device)

classifier = Net()
classifier.load_state_dict(torch.load(os.path.join(model_path, 'MNIST_classifier2.pt')))
classifier.to(device)

# Star-GAN Testing

resize_fn = transforms.Resize([28, 28])

ground_truth_label = np.array([], dtype=int)
without_DA_pred_label = np.array([], dtype=int)
with_DA_pred_label = np.array([], dtype=int)

for i, data in enumerate(tests_bar):

  real, label, dm = data

  real, dm = real.to(device), dm.long().to(device)

  ground_truth_label = np.hstack([ground_truth_label, label.detach().cpu().numpy()])

  # Origin Images
  grayscale = real.mean(axis=1)
  grayscale = resize_fn(grayscale)

  without_DA_pred = torch.argmax(classifier(grayscale.view(BATCH_SIZE, 1, 28, 28)), axis=1).detach().cpu().numpy()
  without_DA_pred_label = np.hstack([without_DA_pred_label, without_DA_pred])

  # Fake Images
  target_dm = torch.zeros(BATCH_SIZE).long().to(device)
  fake = gen(real, target_dm)

  fake_grayscale = fake.mean(axis=1)
  fake_grayscale = resize_fn(fake_grayscale)

  with_DA_pred = torch.argmax(classifier(fake_grayscale.view(BATCH_SIZE, 1, 28, 28)), axis=1).detach().cpu().numpy()
  with_DA_pred_label = np.hstack([with_DA_pred_label, with_DA_pred])


from sklearn.metrics import accuracy_score, confusion_matrix

print("prediction accuracy without domain adaptation", accuracy_score(ground_truth_label, without_DA_pred_label))
print()
print("prediction accuracy with domain adaptation", accuracy_score(ground_truth_label, with_DA_pred_label))
print("\nconfusion matrix without domain adaptation")
print(confusion_matrix(ground_truth_label, without_DA_pred_label))
print("\nconfusion matrix with domain adaptation")
print(confusion_matrix(ground_truth_label, with_DA_pred_label))
