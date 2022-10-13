import os.path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import random
from utils import get_brainwave
from dcgan import weights_init, Generator, Discriminator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-save_path', default='./../G1_Model/',
                    help='Checkpoint to load path from') # please change to your save path
args = parser.parse_args()

torch.cuda.empty_cache()
seed = 369
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

params = {
    "bsize": 64,  # Batch size during training.
    'imsize': 256,  # Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc': 1,  # Number of channles in the training images. For coloured images this is 3.
    'nz': 1024,  # Size of the Z latent vector (the input to the generator).
    'ngf': 64,  # Size of feature maps in the generator. The depth will be multiples of this.
    'ndf': 64,  # Size of features maps in the discriminator. The depth will be multiples of this.
    'nepochs': 50,
    'lr': 0.0002,
    'beta1': 0.5,
    'save_epoch': 2,
    'n_class': 2}  # Save step.

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

dataloader = get_brainwave(params)

sample_batch = next(iter(dataloader))

netG = Generator(params).to(device)
netG.apply(weights_init)
print(netG)

netD = Discriminator(params).to(device)
netD.apply(weights_init)
print(netD)

criterion = nn.BCELoss()

fixed_noise = torch.randn(params['bsize'], params['nz'], 1, 1, device=device)
#fixed_noise = noise_gen.sampleNoise(bsize=['bsize'], nz=params['nz'])

real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(params['beta1'], 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(params['beta1'], 0.999))
schedulerD = torch.optim.lr_scheduler.StepLR(optimizer=optimizerD, step_size=5, gamma=0.8)
schedulerG = torch.optim.lr_scheduler.StepLR(optimizer=optimizerG, step_size=10, gamma=0.8)

img_list = []
G_losses = []
D_losses = []

iters = 0

print("Starting Training Loop...")
print("-" * 25)

# For each epoch
for epoch in range(params['nepochs']):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        # Transfer data tensor to GPU/CPU (device)
        real_data = data[0].to(device)
        # Get batch size. Can be different from params['nbsize'] for last batch in epoch.
        b_size = real_data.size(0)

        # Make accumalated gradients of the discriminator zero.
        netD.zero_grad()
        # Create labels for the real data. (label=1)
        label = torch.full((b_size,), real_label, device=device)

        output = netD(real_data).view(-1)
        output = output.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)
        errD_real = criterion(output, label)

        # Calculate gradients for backpropagation.
        errD_real.backward()
        D_x = output.mean().item()

        # Sample random data from a unit normal distribution.
        noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
        #noise = noise_gen.sampleNoise(bsize=params['bsize'], nz=params['nz'])

        # Generate fake data (images).
        fake_data = netG(noise)

        # Create labels for fake data. (label=0)
        label.fill_(fake_label)

        output = netD(fake_data.detach()).view(-1)
        output = output.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)

        if label.shape != output.shape:
            label = torch.zeros(output.shape)

        errD_fake = criterion(output, label)
        # Calculate gradients for backpropagation.
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = (errD_real + errD_fake)*0.5
        optimizerD.step()

        # Make accumalted gradients of the generator zero.
        netG.zero_grad()
        # We want the fake data to be classified as real.
        # real_label are used. (label=1)
        label.fill_(real_label)
        output = netD(fake_data).view(-1)
        # Calculate G's loss based on this output
        output = output.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)
        errG = criterion(output, label)
        # Calculate gredients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        if i % 1000 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, params['nepochs'], i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on a fixed noise.
        if (iters % 100 == 0) or ((epoch == params['nepochs'] - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake_data = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake_data, padding=2, normalize=True))

        iters += 1

    schedulerD.step()
    #schedulerG.step()

    if epoch % params['save_epoch'] == 0:
        torch.save({
            'generator': netG.state_dict(),
            'discriminator': netD.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'params': params
        }, os.path.join(args.save_path, 'model_epoch_{}.pth'.format(epoch)))

torch.save({
    'generator': netG.state_dict(),
    'discriminator': netD.state_dict(),
    'optimizerG': optimizerG.state_dict(),
    'optimizerD': optimizerD.state_dict(),
    'params': params
}, os.path.join(args.save_path, 'model_final.pth'))

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()