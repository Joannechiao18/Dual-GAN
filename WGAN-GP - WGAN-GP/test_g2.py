import argparse
import pandas as pd
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from wgan2nd import Generator

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='./../G2_Model/WGAN_WGAN.pth', help='Checkpoint to load path from') # please change to your model's path
parser.add_argument('-test_path', default='./../Dataset/test/test1.csv', help='Checkpoint to read file from') # please change to your file's path
parser.add_argument('-num_output', default=64, help='Number of generated outputs')
args = parser.parse_args()

state_dict = torch.load(args.load_path)

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
params = state_dict['params']
print(params)
netG = Generator(params).to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['generator'])
print(netG)

nz_for_2nd = 128
batch_for_2nd = 256

def realtest():
    df = pd.read_csv(args.test_path, usecols=['attention'])
    df = df.to_numpy()
    mix_max_scaler = preprocessing.StandardScaler()
    df = mix_max_scaler.fit_transform(df)

    tensor = np.zeros((0, nz_for_2nd), float)
    for start in range(df.shape[0]):
        if (start + nz_for_2nd) > df.shape[0]:
            break
        arr = df[start: start + nz_for_2nd]
        arr = np.transpose(arr)
        tensor = np.append(tensor, arr, axis=0)

    noise = np.zeros((0, nz_for_2nd), float)
    for count in range(20):
        index = np.random.randint(0, tensor.shape[0])
        print(index)
        add = np.expand_dims(tensor[index], axis=1)
        add = np.transpose(add)
        noise = np.append(noise, add, axis=0)

    r_tensor = np.asarray(noise)
    r_tensor = r_tensor.astype(float).reshape(20, nz_for_2nd)
    r_tensor = torch.from_numpy(r_tensor)
    r_tensor = r_tensor.unsqueeze(-1).unsqueeze(-1)
    r_tensor = r_tensor.type(torch.FloatTensor)
    r_tensor = r_tensor.cuda()
    return r_tensor

noise = realtest()
#noise = torch.randn(params['bsize'], params['nz'], 1, 1, device=device)
# Turn off gradient calculation to speed up the process.
with torch.no_grad():
    generated_img = netG(noise).detach().cpu()

plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1, 2, 0)))
plt.show()

for i in range (100):
    noise = realtest()
    #noise = torch.randn(params['bsize'], params['nz'], 1, 1, device=device)
    # Turn off gradient calculation to speed up the process.
    with torch.no_grad():
        generated_img = netG(noise).detach().cpu()

    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(vutils.make_grid(generated_img[1], padding=2, normalize=True), (1, 2, 0)))
    plt.show()
    # path = 'Your_Save_Path'
    # plt.imsave(path+str(i)+".jpg",np.transpose(vutils.make_grid(generated_img[1], padding=2, normalize=True), (1, 2, 0)).detach().numpy())

