import os
import torch
import torch.nn as nn

from .base_model import BaseModel
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self, latent_shape):
        super(Generator, self).__init__()
        # input shape of (batch_size, latent_space, 1, 1)
        # output shape : 512 x 4 x 4
        self.upsample1 = nn.ConvTranspose2d(latent_shape, 256, 4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(256)
        # input shape : 512 x 4 x 4
        # output shape : 256 x 8 x 8
        self.upsample2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # input shape : 256 x 8 x 8
        # output shape : 128 x 16 x 16
        self.upsample3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # input shape : 128 x 16 x 16
        # output shape : 3 x 32 x 32
        self.upsample4 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

        # initialize weights
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.upsample1(x)))
        x = self.relu(self.bn2(self.upsample2(x)))
        x = self.relu(self.bn3(self.upsample3(x)))
        x = self.tanh(self.upsample4(x))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # shape : 3 x 32 x 32
        self.conv1 = nn.Conv2d(3, 64, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # shape : 64 x 16 x 16
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # shape : 128 x 8 x 8
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        # shape : 256 x 4 x 4
        self.conv4 = nn.Conv2d(256, 1, 4, stride=2, padding=0)
        # shape : 1 x 1 x 1
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        # initialize weights
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.sigmoid(self.conv4(x))
        return x

class DCGAN(BaseModel):

    def __init__(self, args):
        super(DCGAN, self).__init__(args)
        # model
        self.generator = Generator(args.input_shape).to(args.device)
        self.discriminator = Discriminator().to(args.device)
        
        # build optimizer
        self.optG = args.optimizer_class(self.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optD = args.optimizer_class(self.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        
        # build loss
        self.loss = args.criterion_class().to(args.device)
        self.r_label = Variable(torch.Tensor(args.batch_size).fill_(1.0), requires_grad=False).to(args.device)
        self.f_label = Variable(torch.Tensor(args.batch_size).fill_(0.0), requires_grad=False).to(args.device)

        # inputs
        self.r_img = None
        self.latent = None
    
    def setup(self):
        if self.args.checkpoint:
            self.logger.log(f"Loading checkoints from : {self.args.checkpoint}")
            self.load_pretrained_weights(self.args.checkpoint)
        if self.args.is_training:
            self.generator.train()
            self.discriminator.train()
        if 'cuda' in self.args.device:
            torch.cuda.empty_cache()
        
    def load_pretrained_weights(checkpoint):
        state_dict = torch.load(checkpoint, map_location='cpu')
        if 'generator' in state_dict:
            self.generator.load_state_dict(state_dict['generator'])
        if 'discriminator' in state_dict:
            self.discriminator.load_state_dict(state_dict['discriminator'])
        if 'generator_optim' in state_dict:
            self.optG.load_state_dict(state_dict['generator_optim'])
        if 'discriminator_optim' in state_dict:
            self.optD.load_state_dict(state_dict['discriminator_optim'])

    def save(self, save_path=None):
        if save_path is None:
            save_path = self.args.save_root
        self.logger.log(f"Saving model checkpoint in : {save_path}")
        checkpoint = {'generator' : self.generator.state_dict(),
                        'generator_optim' : self.optG.state_dict(),
                        'discriminator' : self.discriminator.state_dict(),
                        'discriminator_optim' : self.optD.state_dict()}
        torch.save(checkpoint, os.path.join(save_path, "checkpoint.pth"))

    def set_inputs(self, inputs):
        self.r_img = inputs['image'].to(self.args.device)
        self.latent = torch.randn(self.args.batch_size, self.args.input_shape, 1, 1).to(self.args.device)

    def forward(self):
        assert self.r_img is not None, "call set_inputs(input) function to set the inputs first"
        # generated data
        self.g = self.generator(self.latent)
        # discriminator out of generated data
        self.d_g = self.discriminator(self.g.detach())
        # discriminator out of real data 
        self.d_r = self.discriminator(self.r_img)
            
    def backwardD(self):
        r_loss = self.loss(self.d_r.view(-1), self.r_label)
        f_loss = self.loss(self.d_g.view(-1), self.f_label)
        self.loss_D = (r_loss + f_loss)/2
        self.loss_D.backward()
    
    def backwardG(self):
        d_g = self.discriminator(self.g)
        self.loss_G = self.loss(d_g.view(-1), self.r_label)
        self.loss_G.backward()
    
    def optimize_parameters(self):
        # do one forward
        self.forward()
        
        # optimize discriminator
        #self.set_requires_grad(self.discriminator, requires_grad=True)
        self.optD.zero_grad()
        self.backwardD()
        self.optD.step()

        # optimize generator
        #self.set_requires_grad(self.discriminator, requires_grad=False)
        self.optG.zero_grad()
        self.backwardG()
        self.optG.step()
    
    def print_stats(self):
        # print stats of current model
        stats = f"Discriminator loss: {self.loss_D.item()}, Generator loss: {self.loss_G.item()}"
        self.logger.log(stats)

