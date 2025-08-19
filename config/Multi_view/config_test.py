
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Sleep Stage Classification - Multi-institutions')    
    
    parser.add_argument('--gpu-id', default='3', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-gpu',default=1,type=int)
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    
                        
    parser.add_argument('--arch', default='resnet50', type=str,
                        choices=['resnet18', 'resnet50'],
                        help='dataset name')
    
    parser.add_argument('--class_num',default=2,type=int)
    
    parser.add_argument('--image-size', default=448, type=int,
                        help='train batchsize')
    parser.add_argument('--image-view',default='1,2',type=str,
                        help='what kind of image view do you want to use?(1,2,3,4)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--epochs',default=100)
    parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
                        help='initial learning rate')
    parser.add_argument('--scheduler',default='cosine',type=str,
                        choices=['cosine','cosine-warmup','linear'])
    parser.add_argument('--warmup', default=0, type=int,
                        help='warmup epochs (unlabeled data based)')
    
    parser.add_argument('--optim',default='SGD',type=str,
                        choices=['SGD','Adam','AdamW'])
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--loss-function',default='CE',type=str,
                        choices=['CE','WCE'])
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    
    parser.add_argument('--use-ema', action='store_true', default=False,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    
    
    parser.add_argument('--seed', default=0, type=int,
                        help="random seed")
    # parser.add_argument('--dataset-path',default='/home/eslab/datasets/')
    parser.add_argument('--dataset-path',default='/data/datasets/')
    parser.add_argument('--out', default='./out',help='directory to output the result')
    # Augmentation Strategy for training dataset
    parser.add_argument('--CoarseDropout-prob',default=0.5,type=float)
    parser.add_argument('--CoarseDropout-max-holes',default=20,type=int)
    parser.add_argument('--CoarseDropout-min-holes',default=1,type=int)
    parser.add_argument('--CoarseDropout-max-height',default=20,type=int)
    parser.add_argument('--CoarseDropout-min-height',default=10,type=int)
    parser.add_argument('--CoarseDropout-max-width',default=20,type=int)
    parser.add_argument('--CoarseDropout-min-width',default=10,type=int)
    
    parser.add_argument('--rotate-degree',default=10,type=int)
    
    parser.add_argument('--check-view',default=-1,type=int)
    
    parser.add_argument('--load-file',default=None,type=str)
    
    parser.add_argument('--classification-mode',default='binary',choices=['binary','two_severe','normal_moderate_severe','normal_mild_moderate_severe'])
    
    parser.add_argument('--k-fold', default=10,type=int)
    parser.add_argument('--k-num',default=0,type=int)
    
    parser.add_argument('--view-image',default='-1',type=str)
    parser.add_argument('--training-view-image',default='-1',type=str)
    
    parser.add_argument('--severe-range',default=60,type=int)
    
    parser.add_argument('--save-image-path',default='./save_image_path/')


    
    # Cross-Attention
    parser.add_argument('--backbone-output-dim',default=2048,type=int)
    parser.add_argument('--cross-attention',default=False,type=str2bool)
    parser.add_argument('--inversed-cross-attention',default=False,type=str2bool)
    parser.add_argument('--dropout',default=0.0,type=float)
    parser.add_argument('--emb-dropout',default=0.0,type=float)
    parser.add_argument('--emb-size',default=768,type=int)
    parser.add_argument('--depth',default=1,type=int)
    parser.add_argument('--num-heads',default=8,type=int)
    parser.add_argument('--backbone-freezing',default=True,type=str2bool)
    parser.add_argument('--positional-embedding',default=True,type=str2bool)
    parser.add_argument('--cls-token',default=False,type=str2bool)
    parser.add_argument('--last-gelu',default=False,type=str2bool)
    parser.add_argument('--backbone-architecture',default='resnet50',type=str,choices=['resnet18','resnet34','resnet50','efficient_b0', 'efficient_b1', 'efficient_b2','resnet101','vit_b_16','swin_b','convnext_b'])
    parser.add_argument('--skip-connection',default=False,type=str2bool)
    parser.add_argument('--shared-attention',default=False,type=str2bool)
    parser.add_argument('--dataset-info-path',default='/data/datasets/filename_list/',type=str)
    parser.add_argument('--MHDA',default=False,type=str2bool)
    parser.add_argument('--last-mlps',default=False,type=str2bool)

    args = parser.parse_args()

    args.image_view = args.image_view.split(',')
    return args