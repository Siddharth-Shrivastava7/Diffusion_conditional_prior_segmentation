import json
import argparse
import numpy as np
from PIL import Image
from os.path import join
from tqdm import tqdm 

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)

def color_to_label(img):
    manual_dd = {
    0: [128, 64,128],
    1: [244, 35,232],
    2: [ 70, 70, 70],
    3: [102,102,156],
    4: [190,153,153],
    5: [153,153,153],
    6: [250,170, 30],
    7: [220,220,  0],
    8: [107,142, 35],
    9: [152,251,152],
    10: [ 70,130,180],
    11: [220, 20, 60],
    12: [255,  0,  0],
    13: [  0,  0,142],
    14: [  0,  0, 70],
    15: [  0, 60,100],
    16: [  0, 80,100],
    17: [  0,  0,230],
    18: [119, 11, 32],
    19: [0,  0, 0], 
    20: [255, 255, 255],
    255: [0,  0, 0] 
    }
    inv_manual_dd = {str(v): k for k, v in manual_dd.items()} 
    img_height, img_width, _ = img.shape 
    img_label = np.zeros((img_height, img_width), dtype=np.uint8) 
    for row in range(img_height):
        for col in range(img_width):
            img_label[row, col] = np.array(inv_manual_dd[str(img[row,col].astype('int64').tolist())])   
    return img_label


def compute_mIoU(gt_dir, pred_dir, devkit_dir=''):
    """
    Compute IoU given the predicted colorized images and 
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
      info = json.load(fp)
    num_classes = np.int(info['classes'])
    print('Num classes', num_classes)
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int) 
    hist = np.zeros((num_classes, num_classes))

    image_path_list = join(devkit_dir, 'acdc_rgb.txt')
    label_path_list = join(devkit_dir, 'acdc_gt.txt')
    gt_imgs = open(label_path_list, 'r').read().splitlines() 
    gt_imgs = [x for x in gt_imgs if "_gt_labelIds.png" in x]
    gt_imgs = sorted([join(gt_dir, x) for x in gt_imgs]) 
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = sorted([join(pred_dir, x.split('/')[-1].replace('_rgb_anon','_gt_labelColor')) for x in pred_imgs])

    for ind in tqdm(range(len(gt_imgs))):
        pred = np.array(Image.open(pred_imgs[ind])) 
        pred = color_to_label(pred) 
        label = np.array(Image.open(gt_imgs[ind]))
        label = label_mapping(label, mapping)  
        # hist += fast_hist(label.flatten(), pred.flatten(), num_classes)   
        mask = (label!=255) 
        pred = pred[mask] 
        label = label[mask] 
        hist += fast_hist(label, pred, num_classes) 
        
    mIoUs = per_class_iu(hist) 
    save_lst = []
    for ind_class in range(num_classes):
        save_lst.append(round(mIoUs[ind_class] * 100, 2)) 
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2))) 
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2))) 
    print('saved_list: ', save_lst)
    return mIoUs


def main(args):
   compute_mIoU(args.gt_dir, args.pred_dir, args.devkit_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', default='/home/sidd_s/scratch/dataset/', type=str, help='directory which stores acdc val gt images')
    parser.add_argument('--pred_dir', default='/home/sidd_s/scratch/dataset/acdc_trainval/rgb_anon/night/synthetic/val/1000n_20p_dannet_pred', type=str, help='directory which stores perturbed acdcgt images')
    parser.add_argument('--devkit_dir', default='/home/sidd_s/mmsegmentation/work/dannetcode_lists', help='base directory of acdc')
    args = parser.parse_args()
    main(args)

