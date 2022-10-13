import os
import random
import torch
import torch.nn as nn
from torch import autograd
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import noise_gen2nd
import utils2nd

from wgan2nd import weights_init, Generator, Discriminator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-save_path', default='./../G2_Model/',
                    help='Checkpoint to load path from') # please change to your save path
args = parser.parse_args()
# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

params = {
    'bsize': 64,
    'imsize': 128,  # Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc': 3,  # Number of channles in the training images. For coloured images this is 3.
    'nz': 128,
    'ngf': 64,  # Size of feature maps in the generator. The depth will be multiples of this.
    'ndf': 64,  # Size of features maps in the discriminator. The depth will be multiples of this.
    'nepochs': 50,
    'lr': 0.0002,
    'beta1': 0.5,
    'beta2': 0.99,
    'save_epoch': 1}

# Number of rounds to update the discriminator
n_critic = 5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# We can use an image folder dataset


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

dataloader = utils2nd.get_other(params)


# Create the generator
netG = Generator(params).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)
print(netG)

# Create the Discriminator
netD = Discriminator(params).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init)
print(netD)


def gradient_penalty(netD, real, fake, device, lambda_=10):
    REAL = real
    FAKE = fake

    batch_size = REAL.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device).expand(REAL.size())
    if REAL.shape != FAKE.shape:
        REAL = torch.zeros(FAKE.shape)

    real_fake = epsilon * REAL + (1 - epsilon) * FAKE
    real_fake.requires_grad_().to(device)

    critic_output = netD(real_fake)

    gradients = autograd.grad(outputs=critic_output,
                              inputs=real_fake,
                              grad_outputs=torch.ones(critic_output.size()).to(device),
                              create_graph=True)[0]

    return ((gradients.norm(2, dim=(1, 2, 3)) - 1) ** 2).mean() * lambda_

fixed_noise =noise_gen2nd.sampleNoise_2nd(params)

optimizerD = optim.Adam(netD.parameters(), lr=0.001)
optimizerG = optim.Adam(netG.parameters(), lr=0.0003)

# Training Loop
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0


print("Starting Training Loop...")
# For each epoch
for epoch in range(params['nepochs']):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)

        for j in range(3):
            penalty = 0
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()

            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate D's output mean on all-real batch
            mean_real = output.mean()
            # Calculate gradients for D in backward pass
            mean_real.backward(gradient=torch.tensor(1., dtype=torch.float, device=device))

            D_x = mean_real.item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = noise_gen2nd.sampleNoise_2nd(params)
            # Generate fake image batch with G
            fake = netG(noise)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's output on the all-fake batch
            mean_fake = output.mean()
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            mean_fake.backward(gradient=torch.tensor(-1., dtype=torch.float, device=device))

            if real_cpu.shape == fake.shape:
                penalty = gradient_penalty(netD, real_cpu, fake.detach(), device)
                penalty.backward()

            D_G_z1 = mean_fake.item()
            # Compute error of D as sum over the fake and the real batches
            errD = mean_real - mean_fake - penalty
            # Update D
            optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        # Generate batch of latent vectors
        noise = torch.randn(params['bsize'], params['nz'], 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate D's output on the all-fake batch
        mean_fake = output.mean()
        # Calculate gradients for G
        mean_fake.backward(gradient=torch.tensor(1., dtype=torch.float, device=device))

        errG = mean_fake.item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\t Wasserstein_Distance: %.4f\tMean_G: %.4f\tMean_D_REAL: %.4f\tMean_D_FAKE: %.4f'
                  % (epoch, params['nepochs'], i, len(dataloader),
                     errD.item(), errG, D_x, D_G_z1))

        # Save Losses for plotting later
        G_losses.append(errG)
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == params['nepochs'] - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
            #vutils.save_image(fake.detach()[0:64],
                              #'%s/fake_samples_epoch_%03d.png' % (opts.checkpoints + 'Image/', epoch),
                              #normalize=True)

        iters += 1

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