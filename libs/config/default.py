from yacs.config import CfgNode
from datetime import datetime
import os

OPTION = CfgNode()
now = datetime.now()

# ------------------------------------------ data configuration ---------------------------------------------
OPTION.trainset = ['POT']
OPTION.valset = 'POT'
OPTION.datafreq = [5, 1]
OPTION.input_size = (360, 640)   # input image size for training
OPTION.sampled_frames = 3        # min sampled time length while trianing
OPTION.max_skip = [5, 3]       # max skip time length while trianing
OPTION.samples_per_video = 24    # sample numbers per video
OPTION.data_backend = 'PIL'     # dataloader backend 'PIL' or 'DALI'
OPTION.objs = [5, 3]
# ----------------------------------------- model configuration ---------------------------------------------
OPTION.keydim = 128
OPTION.valdim = 512
OPTION.arch = 'resnet50'
OPTION.save_freq = 5
OPTION.epochs_per_increment = 5
OPTION.split_k = 3

# ---------------------------------------- training configuration -------------------------------------------
OPTION.epochs = 50
OPTION.train_batch = 1
OPTION.learning_rate = 0.00001
OPTION.gamma = 0.1
OPTION.momentum = (0.9, 0.999)
OPTION.solver = 'adam'             # 'sgd' or 'adam'
OPTION.weight_decay = 5e-4
OPTION.iter_size = 1
OPTION.milestone = [50,80]              # epochs to degrades the learning rate
OPTION.loss = 'both'               # 'ce' or 'iou' or 'both'
OPTION.mode = 'recurrent'          # 'mask' or 'recurrent' or 'threshold'
OPTION.iou_threshold = 0.65        # used only for 'threshold' training
OPTION.alpha = 1.25
OPTION.lambdas = [1.0, 0.4]

# ---------------------------------------- testing configuration --------------------------------------------
OPTION.epoch_per_test = 1
OPTION.correction_rate = 150
OPTION.correction_momentum = 0.9
OPTION.loop = 10

# ------------------------------------------- other configuration -------------------------------------------
OPTION.checkpoint = 'ckpt'
OPTION.code_root = os.path.join(os.getcwd())
OPTION.initial = ''      # path to initialize the backbone
# OPTION.resume = './ckpt/output_05-30_22-35/recurrent.pth.tar'       # path to restart from the checkpoint
# OPTION.resume = './ckpt/stm_cycle_100.pth'
OPTION.resume = ''
OPTION.video_path = ''   # path to video on which the model is running
OPTION.mask_path = ''    # path to mask on withc the model is running
OPTION.gpu_id = 0
OPTION.workers = 10
OPTION.save_indexed_format = 'segmentation'
OPTION.output_dir = 'output_'+now.strftime('%m-%d_%H-%M')
OPTION.print_freq = 20
# ------------------------------------------- optional distributed configuration-------------------------------
OPTION.multi_gpu_ids = [0]
OPTION.backend = 'nccl'
OPTION.init_method = 'env://'
OPTION.cfg = ''
def sanity_check(opt):

    assert isinstance(opt.trainset, (str, list)), \
        'training set should be specified by a string or string list'
    assert isinstance(opt.valset, str), \
        'validation set should be a single dataset'
    assert opt.data_backend in ['PIL', 'DALI'], \
        'only PIL or DALI backend are supported'
    assert opt.solver in ['adam', 'sgd']
    assert opt.mode in ['mask', 'threshold', 'recurrent']
    assert opt.loss in ['iou', 'ce', 'both']
    assert opt.arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101']

def getCfg():

    return OPTION.clone()
