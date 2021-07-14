# GenerativeAdversarialNetworks

Implementation of some of the papers related to GANs on multiple dataset

## Training
To train GAN model, provide the model name as a command line argument to ```--model``` option as shown below:

```
python train.py --batch_size=32 --model=dcgan --dataset=Cifar10 --save_root=./
```

Checkout ```python train.py --help``` for all possible options.

## Validation
Currently validation code is under development.

## Generated Cifar10 images after 100 epochs
![DCGAN](https://raw.githubusercontent.com/kartikkadur/GenerativeAdversarialNetworks/main/DCGAN.png)
