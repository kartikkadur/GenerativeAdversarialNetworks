import torch.nn as nn

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