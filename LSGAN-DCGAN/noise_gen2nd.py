import argparse
import torch
import numpy as np
from lsgan import Generator

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='./../G1_Model/lsgan_g1.pth',
                    help='Checkpoint to load path from') # please change to your model's path
parser.add_argument('-num_output', default=64, help='Number of generated outputs')
args = parser.parse_args()

# Load the checkpoint file.
state_dict = torch.load(args.load_path)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
params = state_dict['params']

# Create the generator network.
netG = Generator(params).to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['generator'])

prevbatch = 64
prevnz = 1024

# noise = torch.randn(int(args.num_output), params['nz'], 1, 1, device=device)

generated_img = np.zeros([])
def generate():
    global generated_img
    #noise = realtest()
    noise = torch.randn(prevbatch, prevnz, 1, 1, device=device)
    # Turn off gradient calculation to speed up the process.
    with torch.no_grad():
        generated_img = netG(noise).detach().cpu()
        generated_img = np.asarray(generated_img)
    return generated_img

generate()

#def sampleNoise_2nd():
def sampleNoise_2nd(params):
    global generated_img

    r_tensor = np.zeros((0, 128), float)
    for count in range(params['bsize'] // generated_img.shape[0]):
        add = generate()
        add = np.squeeze(add, axis=(1, 3))
        r_tensor = np.append(r_tensor, add, axis=0)

    r_tensor = torch.from_numpy(r_tensor)
    r_tensor = r_tensor.unsqueeze(-1).unsqueeze(-1)
    r_tensor = r_tensor.type(torch.FloatTensor)
    r_tensor = r_tensor.cuda()

    return r_tensor