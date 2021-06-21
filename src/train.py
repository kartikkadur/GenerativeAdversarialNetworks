import os
import argparse
import model
import dataset
import tools
import math
import torch
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from torch.autograd import Variable

def argument_parser():
    model_names = sorted(name for name in model.__dict__
                        if not name.startswith("__") and callable(model.__dict__[name]))

    parser = argparse.ArgumentParser(description='Pytorch GAN implementation')

    parser.add_argument('--model', metavar='MODEL', default='dcgan',
                        choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: dcgan)')
    parser.add_argument('--dataset', default='Cifar10', type=str, metavar='DATASET_CLASS',
                        help='Specify dataset class for loading (Default: Cifar10)')
    parser.add_argument('--resume', default='', type=str, metavar='CHECKPOINT_PATH',
                        help='path to checkpoint (default: none)')
    parser.add_argument('--save_root', default=os.getcwd(), type=str, metavar='SAVE_ROOT',
                        help='directory to save log files (default: cwd)')

    parser.add_argument('--lr', '--learning_rate', default=0.0002, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_scheduler', default='MultiStepLR', type=str,
                        metavar='LR_Scheduler', help='Scheduler for learning' +
                                                    ' rate (only ExponentialLR, MultiStepLR, PolyLR supported.')
    parser.add_argument('--lr_gamma', default=0.1, type=float,
                        help='learning rate will be multipled by this gamma')
    parser.add_argument('--lr_step', default=200, type=int,
                        help='stepsize of changing the learning rate')
    parser.add_argument('--lr_milestones', type=int, nargs='+',
                        default=[250, 450], help="Spatial dimension to " +
                                                "crop training samples for training")

    parser.add_argument('-b', '--batch_size', default=4, type=int, metavar='BATCH_SIZE',
                        help='mini batch size (default = 4)')
    parser.add_argument('--num_workers', default=2, type=int, metavar='NUM_WORKERS',
                        help='number of workers to be used for loading batches (default = 2)')
    parser.add_argument('--wd', '--weight_decay', default=0.001, type=float, metavar='WEIGHT_DECAY',
                        help='weight_decay (default = 0.001)')
    parser.add_argument('--seed', default=1234, type=int, metavar="SEED",
                        help='seed for initializing training. ')
    parser.add_argument('--optimizer', default='Adam', type=str, metavar='OPTIMIZER',
                        help='Specify optimizer from torch.optim (Default: Adam)')
    parser.add_argument('--loss', default='BCELoss', type=str, metavar='LOSS',
                        help='Specify loss function from torch.nn (Default: BCELoss)')

    parser.add_argument('--print_freq', default=100, type=int, metavar="PRINT_FREQ",
                        help='frequency of printing training status (default: 100)')

    parser.add_argument('--save_freq', type=int, default=20, metavar="SAVE_FREQ",
                        help='frequency of saving intermediate models (default: 20)')

    parser.add_argument('--epochs', default=100, type=int, metavar="EPOCHES",
                        help='number of total epochs to run (default: 100)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar="STARTEPOCH",
                        help='Epoch number to start training with (default: 0)')

    parser.add_argument('--dataset_root',  metavar="DATASET_ROOT", required=True,
                        help='path to root folder of dataset')
    return parser

def parse_args(block):
    parser = argument_parser()
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)

    defaults, input_arguments = {}, {}
    for key in vars(args):
        defaults[key] = parser.get_default(key)

    for argument, value in sorted(vars(args).items()):
        if value != defaults[argument] and argument in vars(parser.parse_args()).keys():
            input_arguments['--' + str(argument)] = value
            block.log('{}: {}'.format(argument, value))

    args.network = tools.module_to_dict(model)[args.model]
    args.optimizer_class = tools.module_to_dict(torch.optim)[args.optimizer]
    args.dataset_class = tools.module_to_dict(dataset)[args.dataset]
    args.criterion_class = tools.module_to_dict(torch.nn)[args.loss]
    args.latent_shape = 100
    return args

def create_dataloaders(block, args):
    block.log("Creating train and valid  data loaders")
    trainset = args.dataset_class(args.dataset_root, mode='train')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.num_workers,
                                          drop_last=True)
    testset = args.dataset_class(args.dataset_root, mode='valid')
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_workers, 
                                         drop_last=True)
    return trainloader, testloader

def build_model_and_optimizer(block, args):
    block.log("Building Generator and Discriminator models")
    modelG = args.network.Generator(args.latent_shape)
    modelD = args.network.Discriminator()

    block.log("Building Optimizer")
    optimizerG = args.optimizer_class(modelG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerD = args.optimizer_class(modelD.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    if args.resume:
        block.log("Loading checkpoint")
        load_pretrained_weights(modelG, modelD, optimizerG, optimizerD)

    if torch.cuda.is_available():
        modelG.cuda(torch.cuda.current_device())
        modelD.cuda(torch.cuda.current_device())
    return modelG, modelD, optimizerG, optimizerD

def build_criterion(args):
    criterion = args.criterion_class()
    if torch.cuda.is_available():
        criterion.cuda(torch.cuda.current_device())
    return criterion

def lr_scheduler(args, optimizerG, optimizerD):
    if args.lr_scheduler == 'ExponentialLR':
        schedulerG = torch.optim.lr_scheduler.ExponentialLR(optimizerG,
                                                              args.lr_gamma)
        schedulerD = torch.optim.lr_scheduler.ExponentialLR(optimizerD,
                                                              args.lr_gamma)
    elif args.lr_scheduler == 'MultiStepLR':
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(
            optimizerG, milestones=args.lr_milestones, gamma=args.lr_gamma)
        schedulerD = torch.optim.lr_scheduler.MultiStepLR(
            optimizerD, milestones=args.lr_milestones, gamma=args.lr_gamma)

    elif args.lr_scheduler == 'PolyLR':
        lambda_map = lambda epoc: math.pow(1 - epoc / args.epochs, 0.8)
        schedulerG = torch.optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=lambda_map)
        schedulerD = torch.optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda=lambda_map)
    else:
        raise NameError('unknown {} learning rate scheduler'.format(
            args.lr_scheduler))
    return schedulerG, schedulerD

def set_random_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def load_pretrained_weights(args, modelG, modelD, optimizerG, optimizerD):
    checkpoint = torch.load(args.resume, map_location='cpu')
    if 'generator' in checkpoint:
        modelG.load_state_dict(checkpoint['generator'])
    if 'discriminator' in checkpoint:
        modelD.load_state_dict(checkpoint['discriminator'])
    if 'generator_optim' in checkpoint:
        optimizerG.load_state_dict(checkpoint['generator_optim'])
    if 'discriminator_optim' in checkpoint:
        optimizerD.load_state_dict(checkpoint['discriminator_optim'])

def train(block, args, modelG, modelD, optimizerG, optimizerD, criterion, trainloader):
    modelD.train()
    modelG.train()
    torch.cuda.empty_cache()

    block.log("Generate labels for real and fake data")
    real_label = Variable(torch.Tensor(args.batch_size).fill_(1.0), requires_grad=False).cuda()
    fake_label = Variable(torch.Tensor(args.batch_size).fill_(0.0), requires_grad=False).cuda()
    
    for epoch in range(args.start_epoch, args.epochs):
        for i, batch in enumerate(trainloader):
            latent_input = torch.randn(args.batch_size, args.latent_shape, 1, 1)
            # get data
            real_images = batch['image'].cuda()
            gen_images = modelG(latent_input.cuda())
            
            # train discriminator
            modelD.zero_grad()
            real_loss = criterion(modelD(real_images).view(-1), real_label)
            fake_loss = criterion(modelD(gen_images.detach()).view(-1), fake_label)
            # calculate avg total loss
            lossD = (real_loss + fake_loss)/2
            lossD.backward()
            optimizerD.step()

            # train generator
            modelG.zero_grad()
            # Loss measures G's ability to fool D.
            output = modelD(gen_images).view(-1)
            lossG = criterion(output, real_label)
            # Calculate gradients for G
            lossG.backward()
            # Update G
            optimizerG.step()


            if i % args.print_freq == 0:
                block.log(f"Epoch : {epoch}, Iteration : {i}, lossD: {lossD.item()}, "
                    f"lossG: {lossG.item()}")

        if epoch % args.save_freq == 0:
            checkpoint = {'generator' : modelG.state_dict(),
                            'generator_optim' : optimizerG.state_dict(),
                            'discriminator' : modelD.state_dict(),
                            'discriminator_optim' : optimizerD.state_dict()}
            torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pth")

def main():
    with tools.TimerBlock("Parsing Arguments") as block:
        args = parse_args(block)

    with tools.TimerBlock("Setting random seed"):
        set_random_seed(args)

    with tools.TimerBlock("Create dataloaders") as block:
        trainloader, testloader = create_dataloaders(block, args)
    
    with tools.TimerBlock("Create Model, Optimizer and Criterion") as block:
        modelG, modelD, optimizerG, optimizerD = build_model_and_optimizer(block, args)
        criterion = build_criterion(args)
    
    with tools.TimerBlock("Start training") as block:
        train(block, args, modelG, modelD, optimizerG, optimizerD, criterion, trainloader)

if __name__ == '__main__':
    main()