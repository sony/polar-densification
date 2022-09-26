import argparse
import criteria
from metrics import Result

def parser():
    parser = argparse.ArgumentParser(description='Polarizaiton Densification')
    # Parameter
    parser.add_argument('--raw-pattern', '-rp',
                        default='x16',
                        type=str,
                        choices=['conv','x4','x16','x64'],
                        help="sensor raw pattern, [conv(entional)/x4/x16/x64] (default: x16)")   
    parser.add_argument('--ps',
                        default=0.35,
                        type=float,
                        metavar='PS',
                        help='polar pixel sensitivity (default: 0.35)')

    # Network Parameter
    parser.add_argument('--refine-input', '-ri',
                        default=6,
                        type=int,
                        metavar='N',
                        help='RGB refine input [0~7] (default: 6)')
    parser.add_argument('--refine-model', '-rm',
                        default=0,
                        type=int,
                        metavar='N',
                        help='RGB refine model [0,1] (default: 0)')
    parser.add_argument('--comp-input-rgb', '-cir',
                        default=1,
                        type=int,
                        metavar='N',
                        help='Polarizaiton compensation input (RGB) [0,1] (default: 1)')
    parser.add_argument('--comp-input-extra', '-cie',
                        default=2,
                        type=int,
                        metavar='N',
                        help='Polarization compensation input (extra) [0~3] (default: 2)')
    parser.add_argument('--comp-model', '-cm',
                        default=1,
                        type=int,
                        metavar='N',
                        help='Polarization compensation modek [0,1] (default: 1)')

    # Hyper Parameter
    parser.add_argument('--seed', '-se',
                        default=-1,
                        type=int,
                        metavar='N',
                        help='seed value. if -1, random seed (default: -1)')
    parser.add_argument('--epochs',
                        default=30,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run (default: 30)')
    parser.add_argument('--batch-size', '-b', 
                        default=1,
                        type=int,
                        help='mini-batch size (default: 1)')
    parser.add_argument('--lr', '--learning-rate', '-lr', 
                        default=1e-3,
                        type=float,
                        metavar='LR',
                        help='initial learning rate (default 1e-3)')
    parser.add_argument('--weight-decay', '-wd',
                        default=1e-6,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 1e-6)')
    parser.add_argument('--train-num',
                        default=0,
                        type=int,
                        help='the number of train data, 0 is all (default: 0)')
    parser.add_argument('--train-random',
                        action="store_true",
                        default=False,
                        help='random pickup for training data (default: false)')
    parser.add_argument('--s0-8bit',
                        action="store_true",
                        default=False,
                        help='s0 bit length precision (default: false)')

    # Loss Function
    parser.add_argument('-c',
                        '--criterion',
                        default='l1_s12',
                        choices=['l1_s12','l2_s12','l1','l2'],
                        help='PCN loss function, l1 and l2 calculate s012 (default: l1_S12)')
    parser.add_argument('-crgb',
                        '--rgb-criterion',
                        default='l2',
                        choices=['l2','l1'],
                        help='RGBRN loss function: (default: l2)')
    parser.add_argument('--rank-metric',
                        type=str,
                        default='rmse',
                        choices=['rmse', 'mse', 'mae', 'psnr'],
                        help='metrics for which best result is saved (default: rmse)')
    parser.add_argument('--rank-metric-domain',
                        type=str,
                        default='s012',
                        choices=['s12', 's012'],
                        help='domain of metrics for which best result is saved (default: s012)')
    
    # Paths
    parser.add_argument('--data-folder',
                        default='../data/rsp_dataset/x16/',
                        type=str,
                        metavar='PATH',
                        help='data folder (default: "../data/rsp_dataset/x16/")')
    parser.add_argument('--gt-folder',
                        default='../data/rsp_dataset/gt/',
                        type=str,
                        metavar='PATH',
                        help='ground truth folder (default: "../data/rsp_dataset/gt/")')
    parser.add_argument('--result',
                        default='../data/results/',
                        type=str,
                        metavar='PATH',
                        help='result folder (default: "../data/results/")')
    parser.add_argument('--source-directory',
                        default='.',
                        type=str,
                        metavar='PATH',
                        help='source code directory for backup (default: .)')
    parser.add_argument('--suffix',
                        default="",
                        type=str,
                        metavar='FN',
                        help='suffix of result folder name (default: none)')

    # Augmentation Parameter
    parser.add_argument('--not-random-crop',
                        action="store_true",
                        default=True,
                        help='Prohibit random cropping (default: true)')
    parser.add_argument('-he', '--random-crop-height',
                        default=576,
                        type=int,
                        metavar='N',
                        help='random crop height (default: 576)')
    parser.add_argument('-w',
                        '--random-crop-width',
                        default=768,
                        type=int,
                        metavar='N',
                        help='random crop height (default: 768)')
    parser.add_argument('--jitter',
                        type=float,
                        default=0.0,
                        metavar='J',
                        help='color jitter for images, only apply when s0 is 8bit (default: 0.0)')

    # Resume
    parser.add_argument('--resume',
                        default='',
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number, useful on restarts (default: 0)')
    parser.add_argument('-ol', '--optimizer-load',
                        action="store_true",
                        default=False,
                        help='load optimizer when resumimg (default: false)')
    parser.add_argument('--autoresume',
                        action="store_true",
                        default=False,
                        help='auto resume from latest checkpoint (default: false)')
    parser.add_argument('--bestresume',
                        action="store_true",
                        default=False,
                        help='auto resume from best checkpoint (default: false)')

    # Evaluation etc.
    parser.add_argument('-e', '--evaluate',
                        default='',
                        type=str,
                        metavar='PATH',
                        help='use existing models for evaluation (default: none)')
    parser.add_argument('--print-freq',
                        '-p',
                        default=10,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--vis-skip',
                        default=0,
                        type=int,
                        metavar='N',
                        help='skip of visualize comparison image (default: 0)')
    parser.add_argument('--save-img-comp',
                        action="store_true",
                        default=False,
                        help='save image comparison for each epoch (default: false)')
    parser.add_argument('--eval-each',
                        action="store_true",
                        default=False,
                        help='evaluation for each image (default: false)')
    parser.add_argument('--save-interval',
                        default=1,
                        type=int,
                        metavar='N',
                        help='save model interval (default: 1)')
    parser.add_argument('--val-interval',
                        default=1,
                        type=int,
                        metavar='N',
                        help='validation interval (default: 1)')
    parser.add_argument('--train_eval',
                        action="store_true",
                        default=False,
                        help='evaluate when training phase (default: false)')
    parser.add_argument('--disp-all',
                        action="store_true",
                        default=False,
                        help="output all results (default: false)")
    parser.add_argument('--vis-dif',
                        action="store_true",
                        default=False,
                        help="visualize diffuse component (default: false)")
    parser.add_argument('--evalcomp_num',
                        default=120,
                        type=int,
                        metavar='N',
                        help='number of the image for evaluation visualize (default: 120)')

    # Debug
    parser.add_argument('--small',
                        action="store_true",
                        default=False,
                        help='use small dataset (default: false)')
    parser.add_argument('--small-rate',
                        default=0.01,
                        type=float,
                        metavar='SR',
                        help='rate of small dataset, use with "small" argument (default: 0.01)')
    parser.add_argument('--s12gain',
                        default=50.0,
                        type=float,
                        help='s12gain for visualize (default: 50.0)')

    # Others
    parser.add_argument('--val-h',
                        default=576,
                        type=int,
                        metavar='N',
                        help='validation height (default: 576)')
    parser.add_argument('--val-w',
                        default=768,
                        type=int,
                        metavar='N',
                        help='validation width (default: 768)')
    parser.add_argument('--workers',
                        default=4,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--cpu',
                        action="store_true",
                        default=False,
                        help='run on cpu (default: false)')
    parser.add_argument('--gpu',
                        default=-1,
                        type=int,
                        metavar='N',
                        help='GPU device, if -1, use parallel mode (default: -1)')


    args = parser.parse_args()

    return args