from model import Discriminator, Generator
# from data_loader import get_loader
from data_loader import ImgDomainAdaptationData
from utils import generate_imgs, gradient_penalty
from torch import optim
import torch
import torch.nn as nn
import os
import datetime
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from torchvision.models import vgg19
from torch.nn.functional import cross_entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    """Early stops the training if reconstruction loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, epsilon=1.04):
        """
        Args:
            patience (int): How long to wait after last time reconstruction loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.reconstruction_loss_min = np.Inf
        self.epsilon = epsilon

    def __call__(self, reconstruction_loss, gen, dis):

        score = -reconstruction_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(reconstruction_loss, gen, dis)
        elif score < self.best_score / self.epsilon:
            self.counter += 1
            print(f'\nCurrent reconstruction loss {reconstruction_loss:.6f} > {self.reconstruction_loss_min:.6f}/{self.epsilon} = {self.reconstruction_loss_min/self.epsilon:.6f}')
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(reconstruction_loss, gen, dis)
            self.counter = 0

    def save_checkpoint(self, reconstruction_loss, gen, dis):
        '''Saves model when the reconstruction loss decrease.'''
        if self.verbose:
            print(f'\nReconstruction loss decreased ({self.reconstruction_loss_min:.6f} --> {reconstruction_loss:.6f}).  Saving model ...\n')
        # Note: Here you should define how you want to save your model. For example:
        torch.save(gen.state_dict(), os.path.join(model_path, 'gen.pkl'))
        torch.save(dis.state_dict(), os.path.join(model_path, 'dis.pkl'))
        self.reconstruction_loss_min = reconstruction_loss


# Load a pre-trained VGG19 model, remove its final linear layer, and set to evaluation mode
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg_features = vgg19(pretrained=True).features
        self.vgg_slice = nn.Sequential(*list(vgg_features.children())[:36])  # Example: up to the second-last conv layer

    def forward(self, x):
        return self.vgg_slice(x)

vgg_extractor = VGGFeatureExtractor().to(device).eval()

def perceptual_loss(generated, target):
    gen_features = vgg_extractor(generated)
    target_features = vgg_extractor(target)
    return (gen_features - target_features).abs().mean()

def identity_loss(generator, real, domain):
    # Identity mapping loss
    same_domain_img = generator(real, domain)
    return (real - same_domain_img).abs().mean()


is_cuda = torch.cuda.is_available()

EPOCHS = 300  # 50-300
BATCH_SIZE = 128
LOAD_MODEL = False

IMAGE_SIZE = 32
NUM_DOMAINS = 4

N_CRITIC = 5
GRADIENT_PENALTY = 10

# Directories for storing model and output samples
model_path = './model'
os.makedirs(model_path, exist_ok=True)
samples_path = './samples'
os.makedirs(samples_path, exist_ok=True)

CONV_DIM = 12
# Networks
gen = Generator(num_domains=NUM_DOMAINS, image_size=IMAGE_SIZE, conv_dim=CONV_DIM)
dis = Discriminator(num_domains=NUM_DOMAINS, image_size=IMAGE_SIZE, conv_dim=CONV_DIM)

# print(gen)
# print(dis)

# Load previous model
# if LOAD_MODEL:
#     gen.load_state_dict(torch.load(os.path.join(model_path, 'gen.pkl')))
#     dis.load_state_dict(torch.load(os.path.join(model_path, 'dis.pkl')))

gen.to(device)
dis.to(device)

# Define Optimizers
g_opt = optim.Adam(gen.parameters(), lr=0.0001, betas=(0.5, 0.999))
d_opt = optim.Adam(dis.parameters(), lr=0.0001, betas=(0.5, 0.999))

# Define Loss
ce = nn.CrossEntropyLoss()

# Data loaders
IMGS_TO_DISPLAY = 25
dataset = ImgDomainAdaptationData(IMAGE_SIZE, IMAGE_SIZE)
ds_loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True)
iters_per_epoch = len(ds_loader)

# Fix images for viz
loader_iter = iter(ds_loader)
img_fixed = next(loader_iter)[0][:IMGS_TO_DISPLAY].to(device)

# GPU Compatibility
# is_cuda = torch.cuda.is_available()
# if is_cuda:
# 	gen, dis = gen.cuda(), dis.cuda()
# 	img_fixed = img_fixed.cuda()

total_iter = 0
g_adv_loss = g_clf_loss = g_rec_loss = torch.Tensor([0])

w_adv = 1
w_clf = 1
w_rec = 1

max_w_rec = 2
w_rec_inc = (max_w_rec - w_rec)/100

# experiment
w_vgg = 1
w_ide = 1

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
current_setting = f'experiment_vgg_{w_vgg}_ide_{w_ide}_adv{w_adv}_clf{w_clf}_rec{w_rec}_{max_w_rec}_conv{CONV_DIM}_batch{BATCH_SIZE}_ncritic_{N_CRITIC}'
# current_setting = f'adv{w_adv}_clf{w_clf}_rec{w_rec}_{max_w_rec}_conv{CONV_DIM}_batch{BATCH_SIZE}_ncritic_{N_CRITIC}'

model_path = os.path.join(model_path, current_setting, current_time)
os.makedirs(model_path, exist_ok=True)
samples_path = os.path.join(samples_path, current_setting, current_time)
os.makedirs(samples_path, exist_ok=True)

early_stopping = EarlyStopping(patience=40, verbose=True)

g_adv_loss_per_ep = []
g_clf_loss_per_ep = []
g_rec_loss_per_ep = []
g_loss_per_ep = []

# experiment
g_vgg_loss_per_ep = []
g_ide_loss_per_ep = []

d_adv_loss_per_ep = []
d_clf_loss_per_ep = []
d_loss_per_ep = []

train_info = []

for epoch in range(EPOCHS):
    gen.train()
    dis.train()

    ep_g_adv_losses = []
    ep_g_clf_losses = []
    ep_g_rec_losses = []
    ep_g_losses = []

    # experiment
    ep_g_vgg_losses = []
    ep_g_ide_losses = []

    ep_d_adv_losses = []
    ep_d_clf_losses = []
    ep_d_losses = []

    for batch_idx, batch_data in tqdm(enumerate(ds_loader), total=iters_per_epoch, desc=f'Epoch {epoch+1}'):
        total_iter += 1

        # Loading batch_data
        real, _, dm = batch_data

        real, dm = real.to(device), dm.long().to(device)

        target_dm = dm[torch.randperm(dm.size(0))]

        # Fake Images
        fake = gen(real, target_dm)

        # Training discriminator
        real_gan_out, real_cls_out = dis(real)
        fake_gan_out, fake_cls_out = dis(fake.detach())

        d_adv_loss = -(real_gan_out.mean() - fake_gan_out.mean()) + gradient_penalty(real, fake, dis, is_cuda) * GRADIENT_PENALTY
        d_clf_loss = ce(real_cls_out, dm)

        d_opt.zero_grad()
        d_loss = w_adv * d_adv_loss + w_clf * d_clf_loss
        d_loss.backward()
        d_opt.step()

        ep_d_adv_losses.append(d_adv_loss.item())
        ep_d_clf_losses.append(d_clf_loss.item())
        ep_d_losses.append(d_loss.item())

        # Training Generator
        if total_iter % N_CRITIC == 0:
            fake = gen(real, target_dm)
            fake_gan_out, fake_cls_out = dis(fake)

            g_adv_loss = - fake_gan_out.mean()
            g_clf_loss = ce(fake_cls_out, target_dm)
            g_rec_loss = (real - gen(fake, dm)).abs().mean()

            # experiment
            g_vgg_loss = perceptual_loss(fake, real)
            g_ide_loss = identity_loss(gen, real, dm)

            g_opt.zero_grad()
            g_loss = w_adv * g_adv_loss + w_clf * g_clf_loss + w_rec * g_rec_loss

            # experiment
            # g_loss += w_vgg * g_vgg_loss + w_ide * g_ide_loss
            g_loss += w_vgg * g_vgg_loss
            g_loss += w_ide * g_ide_loss

            g_loss.backward()
            g_opt.step()

            ep_g_adv_losses.append(g_adv_loss.item())
            ep_g_clf_losses.append(g_clf_loss.item())
            ep_g_rec_losses.append(g_rec_loss.item())
            ep_g_losses.append(g_loss.item())

            # experiment
            ep_g_vgg_losses.append(g_vgg_loss.item())
            ep_g_ide_losses.append(g_ide_loss.item())

        # if batch_idx % 100 == 99:
        #     print("\n==============================")
        #     print("Epoch: " + str(epoch + 1) + "/" + str(EPOCHS)
        #           + " iter: " + str(batch_idx+1) + "/" + str(iters_per_epoch)
        #           + " total_iters: " + str(total_iter)
        #           + "\n------------------------------"
        #           + "\ng_adv_loss:" + str(round(g_adv_loss.item(), 4))
        #           + "\tg_clf_loss:" + str(round(g_clf_loss.item(), 4))
        #           + "\tg_rec_loss:" + str(round(g_rec_loss.item(), 4))
        #           + "\td_adv_loss:" + str(round(d_adv_loss.item(), 4))
        #           + "\td_clf_loss:" + str(round(d_clf_loss.item(), 4)))

        #     print("------------------------------")
        #     print(f'Gen loss: {g_loss}, Dis loss: {d_loss}\n')

    w_rec = min(max_w_rec, w_rec + w_rec_inc)

    generate_imgs(img_fixed, NUM_DOMAINS, gen, samples_path, device, epoch+1)

    # Compute the average reconstruction loss for the epoch
    avg_g_adv_loss = np.mean(ep_g_adv_losses)
    avg_g_clf_loss = np.mean(ep_g_clf_losses)
    avg_g_rec_loss = np.mean(ep_g_rec_losses)
    avg_g_loss = np.mean(ep_g_losses)

    # experiment
    avg_g_vgg_loss = np.mean(ep_g_vgg_losses)
    avg_g_ide_loss = np.mean(ep_g_ide_losses)

    avg_d_adv_loss = np.mean(ep_d_adv_losses)
    avg_d_clf_loss = np.mean(ep_d_clf_losses)
    avg_d_loss = np.mean(ep_d_losses)

    # train_info.append([avg_g_adv_loss, avg_g_clf_loss, avg_g_rec_loss, avg_g_loss, avg_d_adv_loss, avg_d_clf_loss, avg_d_loss])
    train_info.append([avg_g_adv_loss, avg_g_clf_loss, avg_g_rec_loss,
                        avg_g_vgg_loss,
                        avg_g_ide_loss,
                       avg_g_loss,
                       avg_d_adv_loss, avg_d_clf_loss,
                       avg_d_loss])

    g_adv_loss_per_ep.append(avg_g_adv_loss)
    g_clf_loss_per_ep.append(avg_g_clf_loss)
    g_rec_loss_per_ep.append(avg_g_rec_loss)
    g_loss_per_ep.append(avg_g_loss)

    # Experiment
    g_vgg_loss_per_ep.append(avg_g_vgg_loss)
    g_ide_loss_per_ep.append(avg_g_ide_loss)

    d_adv_loss_per_ep.append(avg_d_adv_loss)
    d_clf_loss_per_ep.append(avg_d_clf_loss)
    d_loss_per_ep.append(avg_d_loss)

    print(f'\nGenerator:\nadv_loss: {avg_g_adv_loss:.4f}, clf_loss: {avg_g_clf_loss:.4f}, rec_loss: {avg_g_rec_loss:.4f}')
    # print(f'vgg_loss: {g_vgg_loss:.4f}, ide_loss: {g_ide_loss:.4f}')
    print(f'vgg_loss: {g_vgg_loss:.4f}')
    print(f'ide_loss: {g_ide_loss:.4f}')
    print(f'Total loss with weight ({w_adv}, {w_clf}, {w_rec:.4f}, {w_vgg}, {w_ide}): {avg_g_loss:.4f}')
    # print(f'Total loss with weight ({w_adv}, {w_clf}, {w_rec:.4f}): {avg_g_loss:.4f}')
    print("------------------------------")
    print(f'Discriminator:\nadv_loss: {avg_d_adv_loss:.4f}, clf_loss: {avg_d_clf_loss:.4f}')
    print(f'Total loss with weight ({w_adv}, {w_clf}): {avg_d_loss:.4f}')
    print("==============================")


    # Early stopping check at the end of the epoch
    early_stopping(avg_g_rec_loss, gen, dis)
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch}.")
        break


generate_imgs(img_fixed, NUM_DOMAINS, gen, samples_path, device, -1)

import pandas as pd
df = pd.DataFrame(train_info, columns=['Gen Adv Loss', 'Gen Clf Loss', 'Gen Rec Loss',
                                       'Gen VGG Loss',
                                       'Gen Ide Loss',
                                       'Gen Total Loss',
                                       'Dis Adv Loss', 'Dis Clf Loss',
                                       'Dis Total Loss'])

train_info_path = os.path.join('.', 'train_info', current_setting, current_time)
os.makedirs(train_info_path, exist_ok=True)
train_info_path = os.path.join(train_info_path, 'train_info.csv')
df.to_csv(train_info_path, index=True)

print(f'Saving train info to {train_info_path}')
print('All done')
