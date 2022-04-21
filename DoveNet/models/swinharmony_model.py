from sympy import im
import torch
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from torch import device, nn, cuda
from torch.autograd import Variable
import functools
from torch.nn.utils import spectral_norm
from .network_swinir_encoder import *
from .light_learner import *

class SwinHarmonyModel(BaseModel):
    

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        """
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--gp_ratio', type=float, default=1.0, help='weight for gradient_penalty')
            parser.add_argument('--lambda_a', type=float, default=1.0, help='weight for adversarial loss')
            parser.add_argument('--lambda_v', type=float, default=1.0, help='weight for verification loss')

        return parser

    def __init__(self, opt):
        """Initialize the DoveNet class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1', 'G_L2']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['comp', 'real', 'cap', 'output', 'mask', 'real_f', 'fake_f', 'bg']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']

        window_size = 4
        height = width = 20
        self.netG = SwinIREncoder(upscale=1, img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='none')
        self.netG = networks.init_net(self.netG, opt.init_type, opt.init_gain, self.gpu_ids)
        
        self.light_learner = GlobalLighting(n_downsample=0, input_dim=4, dim=64, norm='none',activ = 'lrelu', pad_type = 'reflect')
        self.light_learner = networks.init_net(self.light_learner, opt.init_type, opt.init_gain, self.gpu_ids) 
        self.light_transfer = LightingResBlocks(num_blocks=4, dim=60,light_mlp_dim=8, norm='ln')
        self.light_transfer = networks.init_net(self.light_transfer, opt.init_type, opt.init_gain, self.gpu_ids) 

        self.relu = nn.ReLU()

        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.mse = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr*opt.g_lr_ratio,
                                                betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.iter_cnt = 0        

        self.conv_last = nn.Conv2d(60, 3, 3, 1, 1).to(self.gpu_ids[0])

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.comp = input['comp'].to(self.device)
        self.real = input['real'].to(self.device)
        self.inputs = input['inputs'].to(self.device)
        self.mask = self.inputs[:, 3:4, :, :]
        self.real_f = self.real * self.mask
        self.bg = self.real * (1 - self.mask)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.denoising_encoded_feature = self.netG(self.comp)
        #print('self.comp.shape=', self.comp.shape)
        #print('self.mask.shape=', self.mask.shape)
        fg_pooling, bg_pooling = self.light_learner(self.inputs, self.mask)

        transfered_feature = self.light_transfer(self.denoising_encoded_feature, fg_pooling, bg_pooling, self.mask)
        #self.output = self.conv_last(transfered_feature)
        self.output = self.conv_last(self.denoising_encoded_feature)
        self.fake_f = self.output * self.mask
        self.cap = self.output * self.mask + self.comp * (1 - self.mask)
        self.harmonized = self.output

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        self.loss_G_L1 = self.criterionL1(self.output, self.real) * self.opt.lambda_L1
        self.loss_G_L2 = self.mse(self.output, self.real) * self.opt.lambda_L1
        self.loss_G = self.loss_G_L1
        self.loss_G.backward(retain_graph=True)

    def optimize_parameters(self):
        self.forward()  

        # update G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights