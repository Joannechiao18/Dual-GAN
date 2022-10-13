import argparse
import csv
import torch
import numpy as np
import torch.nn.functional as F
import noise_gen
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
from wgan import Generator

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='./../G1_Model/wgan_g1.pth', help='Checkpoint to load path from')
parser.add_argument('-num_output', default=64, help='Number of generated outputs')
args = parser.parse_args()

# Load the checkpoint file.
state_dict = torch.load(args.load_path)

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
params = state_dict['params']

# Create the generator network.
netG = Generator(params).to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['generator'])
#print(netG)

noise = torch.randn(params['bsize'], params['nz'], 1, 1, device=device)

allMSE = []

def extract_features(model):
    global sum, result_npy_gen, result_npy_ori, allMSE

    dataset = noise_gen.custom_training_set()
    generated_img = netG(noise).detach().cpu()

    for i in range(params['bsize']):
        index = np.random.randint(0, dataset.shape[0])

        padded_ori = F.pad(input=dataset[index], pad=(0, 223, 96, 0), mode='constant', value=0)
        addChannel_ori = padded_ori.expand(1, 3, 224, 224)
        addChannel_ori = addChannel_ori.cpu()
        addChannel_ori = np.asarray(addChannel_ori)
        ############################################################
        padded_gen = F.pad(input=generated_img[i], pad=(0, 223, 96, 0), mode='constant', value=0)
        addChannel_gen = padded_gen.expand(1, 3, 224, 224)
        addChannel_gen = np.asarray(addChannel_gen)

        ############################################################
        tensor = torch.from_numpy(addChannel_gen)
        tensor = tensor.cuda()
        result_gen = model(Variable(tensor))
        result_npy_gen = result_gen.data.cpu().numpy()
        ###########################################################
        tensor = torch.from_numpy(addChannel_ori)
        tensor = tensor.cuda()
        result_ori = model(Variable(tensor))
        result_npy_ori = result_ori.data.cpu().numpy()

        loss = torch.nn.MSELoss()
        mse = loss(result_gen[0], result_ori[0])
        mse = mse.item()
        allMSE.append(mse)
    return result_npy_gen[0], result_npy_ori[0]

def make_model():
    model = models.vgg16(pretrained=True).features[:20]
    model = model.eval()
    model.cuda()
    return model

if __name__ == "__main__":
    model = make_model()
    features_gen, features_ori = extract_features(model)
    with open('./MSE.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(allMSE)
