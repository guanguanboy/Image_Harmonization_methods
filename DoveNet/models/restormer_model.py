import torch
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from torch import device, nn, cuda
from torch.autograd import Variable
import functools
from torch.nn.utils import spectral_norm
from .restormer_arch import *

class RestormerModel(BaseModel):
    

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
        self.loss_names = ['G_L1']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['comp', 'real', 'cap', 'output', 'mask', 'real_f', 'fake_f', 'bg']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']

        self.netG = Restormer()
        self.netG = networks.init_net(self.netG, opt.init_type, opt.init_gain, self.gpu_ids)
        
        self.relu = nn.ReLU()

        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr*opt.g_lr_ratio,
                                                betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.iter_cnt = 0        

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
        self.output= self.netG(self.comp)
        #print('self.output.shape=', self.output.shape)
        #print('self.mask.shape=', self.mask.shape)
        self.fake_f = self.output * self.mask
        self.cap = self.output * self.mask + self.comp * (1 - self.mask)
        self.harmonized = self.output

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        self.loss_G_L1 = self.criterionL1(self.output, self.real) * self.opt.lambda_L1
        self.loss_G = self.loss_G_L1
        self.loss_G.backward(retain_graph=True)

    def optimize_parameters(self):
        self.forward()  

        # update G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights