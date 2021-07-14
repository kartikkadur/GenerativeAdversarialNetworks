import os
import tools
import torch

from options.train_arguments import TrainArguments

def create_dataloaders(block, args):
    block.log("Creating train and valid  data loaders")
    trainloader = args.dataset.get_dataloader(args, is_training=True)
    valloader = args.dataset.get_dataloader(args, is_training=False)
    return trainloader, valloader

def set_random_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def train(block, args):
    # build dataset
    trainloader, valloader = create_dataloaders(block, args)
    block.log(f"Number of training samples : {len(trainloader)}")
    block.log(f"Number of validation samples : {len(valloader)}")
    # build and setup model
    model = args.network_class(args)
    model.setup()

    for epoch in range(args.start_epoch, args.epochs):
        for i, batch in enumerate(trainloader):
            # set inputs
            model.set_inputs(batch)
            # train model
            model.optimize_parameters()
            if i % args.print_freq == 0:
                block.log(f"Current Epoch -> {epoch}, "
                          f"Current Iteration -> {i}, ")
                model.print_stats()

        if epoch % args.save_freq == 0:
            model.save()

def main():
    with tools.TimerBlock("Get Arguments") as block:
        args = TrainArguments().parse(block)

    with tools.TimerBlock("Setting random seed"):
        set_random_seed(args)
    
    with tools.TimerBlock("Start training") as block:
        args.logger = block
        train(block, args)

if __name__ == '__main__':
    main()