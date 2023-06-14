import argparse

# validation set path
# DATA_DIRECTORY = '/home/sidd_s/scratch/dataset/cityscapes/leftImg8bit/train'
# DATA_LIST_PATH = '/home/sidd_s/diffusion_priors/segmentation/DANNet/dataset/lists/cityscapes_train.txt'
# DATA_DIRECTORY = '/home/sidd_s/scratch/dataset/dark_zurich_val/pred/dannet_PSPNet_val' # for computing miou
DATA_DIRECTORY = '/home/sidd_s/scratch/dataset/dark_zurich_val/rgb_anon/val'
DATA_LIST_PATH = '/home/sidd_s/diffusion_priors/segmentation/DANNet/dataset/lists/zurich_val.txt'


# test set path
# DATA_DIRECTORY = '/path/to/public_data_2/rgb_anon'
# DATA_LIST_PATH = './dataset/lists/zurich_test.txt'

IGNORE_LABEL = 255
NUM_CLASSES = 19
SET = 'train' 

MODEL = 'PSPNet'
RESTORE_FROM = '/home/sidd_s/scratch/saved_models/DANNet/dannet_psp.pth'
RESTORE_FROM_LIGHT = '/home/sidd_s/scratch/saved_models/DANNet/dannet_psp_light.pth'
SAVE_PATH = '/home/sidd_s/scratch/results/Dannet_dark_zurich'
STD = 0.16


def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-light", type=str, default=RESTORE_FROM_LIGHT,
                        help="Where restore model parameters from.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument("--std", type=float, default=STD)
    return parser.parse_args()