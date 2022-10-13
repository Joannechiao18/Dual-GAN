import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(w):
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input is the latent vector Z.
        self.tconv1 = nn.ConvTranspose2d(params['nz'], params['ngf']*16,
                                         kernel_size=(4, 1), stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(params['ngf']*16, momentum=0.5)  # momentum = 0.1 (default)
        self.drop1 = nn.Dropout(0.25)


        self.tconv2 = nn.ConvTranspose2d(params['ngf']*16, params['ngf']*8,
            kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(params['ngf']*8, momentum=0.5)
        self.drop2 = nn.Dropout(0.25)


        self.tconv3 = nn.ConvTranspose2d(params['ngf']*8, params['ngf']*4,
            kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False)
        self.bn3 = nn.BatchNorm2d(params['ngf']*4, momentum=0.5)
        self.drop3 = nn.Dropout(0.25)


        self.tconv4 = nn.ConvTranspose2d(params['ngf']*4, params['ngf']*4,
            kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False)
        self.bn4 = nn.BatchNorm2d(params['ngf']*4, momentum=0.5)
        self.drop4 = nn.Dropout(0.25)


        self.tconv5 = nn.ConvTranspose2d(params['ngf']*4, params['ngf']*2,
            kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False)
        self.bn5 = nn.BatchNorm2d(params['ngf']*2, momentum=0.5)
        self.drop5 = nn.Dropout(0.25)

        self.tconv6 = nn.ConvTranspose2d(params['ngf'] * 2, params['nc'],
                                         kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False)

    def forward(self, x):

        x = F.leaky_relu(self.bn1(self.tconv1(x)))
        x = F.dropout(self.drop1(x))

        x = F.leaky_relu(self.bn2(self.tconv2(x)))
        x = F.dropout(self.drop2(x))

        x = F.leaky_relu(self.bn3(self.tconv3(x)))
        x = F.dropout(self.drop3(x))

        x = F.leaky_relu(self.bn4(self.tconv4(x)))

        x = F.leaky_relu(self.bn5(self.tconv5(x)))

        x = torch.tanh(self.tconv6(x))

        return x

# Define the Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input Dimension: (nc) x 128 x 128
        self.conv1 = nn.Conv2d(params['nc'], params['ndf'],
            kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False)

        # Input Dimension: ndf x 64 x 64
        self.conv2 = nn.Conv2d(params['ndf'], params['ndf']*2,
            kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(params['ndf']*2)
        self.drop1 = nn.Dropout(0.5)

        # Input Dimension: ndf*2 x 32 x 32
        self.conv3 = nn.Conv2d(params['ndf']*2, params['ndf']*4,
            kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False)
        self.bn3 = nn.BatchNorm2d(params['ndf']*4)
        self.drop2 = nn.Dropout(0.5)

        # Input Dimension: ndf*4 x 16 x 16
        self.conv4 = nn.Conv2d(params['ndf']*4, params['ndf']*8,
            kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False)
        self.bn4 = nn.BatchNorm2d(params['ndf']*8)
        self.drop3 = nn.Dropout(0.5)

        # Input Dimension: ndf*8 x 8 x 8
        self.conv5 = nn.Conv2d(params['ndf']*8, params['ndf']*16,
            kernel_size=(4, 1), stride=(4, 1), padding=(1, 0), bias=False)
        self.bn5 = nn.BatchNorm2d(params['ndf']*16)
        self.drop4 = nn.Dropout(0.5)

        self.conv6 = nn.Conv2d(params['ndf']*16, 1,
                               kernel_size=(4, 1), stride=(4, 1), padding=(1, 0), bias=False)
    def forward(self, x):

        x = F.leaky_relu(self.conv1(x), 0.2, True)

        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.dropout(self.drop1(x))

        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.dropout(self.drop2(x))

        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)

        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2, True)

        x = torch.sigmoid(self.conv6(x))

        return x