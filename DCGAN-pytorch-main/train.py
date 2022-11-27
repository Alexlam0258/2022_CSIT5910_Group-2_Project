import os
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torchvision

from configs.parameters import *
from datasets.flower import flower_dataloader as dataloader
from dcgan import Discriminator128 as Discriminator
from dcgan import Generator128 as Generator
from dcgan import weights_init
from utils import add_frame_to_gif, denorm

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)



class DCGANTrain:
    def __init__(self, dataloader, dataset_name, ngpu=1):
        self.dataloader = dataloader
        self.dataset_name = dataset_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.netG = Generator(ngpu).to(device)
        self.netG.apply(weights_init)

        self.netD = Discriminator(ngpu).to(device)
        self.netD.apply(weights_init)

        print(self.netG)
        print(self.netD)

        # Initialize BCELoss function
        self.criterion = torch.nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.fixed_noise = torch.randn(image_size, nz, 1, 1, device=device)

        # Establish convention for real and fake labels during training
        self.real_label = 1.
        self.fake_label = 0.

        # Setup Adam optimizers for both G and D
        self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        self.gif_frames = []

        self.G_losses = []
        self.D_losses = []

        self.gif_generater = partial(add_frame_to_gif, self.dataset_name, self.gif_frames)


    def one_epoch(self, epoch):
        D_x = D_G_z1 =  D_G_z2 = 0.0
         # For each batch in the dataloader
        for i, data in enumerate(self.dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            self.netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(self.device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
            # Forward pass real batch through D
            output = self.netD(real_cpu).view(-1)
            # Calculate loss on all-real batch

            errD_real = self.criterion(output, label)
            
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=self.device)
            # Generate fake image batch with G
            fake = self.netG(noise)
            label.fill_(self.fake_label)
            # Classify all fake batch with D
            output = self.netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = self.criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            self.optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.netG.zero_grad()
            label.fill_(self.real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = self.netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = self.criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            self.optimizerG.step()

            # Save Losses for plotting later
            self.G_losses.append(errG.item())
            self.D_losses.append(errD.item())

        # print training state at end of epoch
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
            % (epoch, num_epochs, i, len(self.dataloader),
                errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save sampled images
        fake = self.netG(self.fixed_noise).detach().cpu()
        fake_images = fake.reshape(fake.size(0), nc, image_size, image_size)
        fake_images_path = os.path.join(sample_dir, f'fake_images_{epoch+1}.png')
        torchvision.utils.save_image(denorm(fake_images), os.path.join(fake_images_path))
        print(f"saved {fake_images_path}")
        # add_frame_to_gif(self.dataset_name, self.gif_frames, fake_images_path)
        self.gif_generater(fake_images_path)


    def save_models(self):
        # save state_dict and model
        torch.save(self.netG.state_dict(), f'G_state_dict_test_{self.dataset_name}.pt')
        torch.save(self.netD.state_dict(), f'D_state_dict_test_{self.dataset_name}.pt')

        torch.save(self.netD, f'D_model_test_{self.dataset_name}.pth')
        torch.save(self.netG, f'G_model_test_{dataset_name}.pth')


def train():
    dcgan_train = DCGANTrain(dataloader, dataset_name)
    for epoch in range(num_epochs):
        dcgan_train.one_epoch(epoch)
        dcgan_train.save_models()


if __name__ == '__main__':
    train()
