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


def compute_mIoU(gt_dir, pred_dir, devkit_dir=''):
    """
    Compute IoU given the predicted colorized images and gt images 
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
      info = json.load(fp)
    num_classes = int(info['classes'])
    print('Num classes', num_classes)
    name_classes = np.array(info['label'], dtype=str)
    hist = np.zeros((num_classes, num_classes))

    gt_imgs = ['/home/sidd_s/Diffusion_conditional_prior_segmentation/diffusion_plug_prior_mod/experiments/miou_calc/gt/label_inds.png'] 
    # pred_imgs = ['/home/sidd_s/Diffusion_conditional_prior_segmentation/diffusion_plug_prior_mod/experiments/miou_calc/dannet_pred/Dannet_prediction_inds.png'] # miou >>>> (34.27: DaNNet) 
    # pred_imgs = ['/home/sidd_s/Diffusion_conditional_prior_segmentation/diffusion_plug_prior_mod/experiments/miou_calc/mic_pred/MIC_prediction_inds.png'] # miou >>>>  (45.31 MIC) {35.9 for 256x512}
    # pred_imgs = ['/home/sidd_s/Diffusion_conditional_prior_segmentation/diffusion_plug_prior_mod/experiments/samples/single_pred_inds.png'] # (363 image number) miou >>>> (33.22: from DaNNet by multinomial diffusion) >>>> 
    pred_imgs = ['/home/sidd_s/Diffusion_conditional_prior_segmentation/diffusion_plug_prior_mod/experiments/samples/joint_pred_inds.png'] # miou >>>> {37.44 for 256x512 MIC logits combined} 35.52 (from DaNNet by multinomial diffusion and joint distribution simiplied formula)

    for ind in tqdm(range(len(gt_imgs))):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))
        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        # if ind > 0 and ind % 10 == 0:
        #     print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))
    
    mIoUs = per_class_iu(hist)
    dict_miou = {}
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
        dict_miou[name_classes[ind_class]] = str(round(mIoUs[ind_class] * 100, 2))
    print('>>>>>>>>>>>', list(dict_miou.values()))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    print('>>>>>>>>>>', dict_miou)
    return mIoUs


def main(args):
   compute_mIoU(args.gt_dir, args.pred_dir, args.devkit_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gt_dir', default='/home/sidd_s/scratch/dataset/cityscapes/gtFine/val/', type=str, help='directory which stores CityScapes val gt images')
    # parser.add_argument('--pred_dir', default='/home/sidd_s/scratch/results/Dannet/cityscapes/val/', type=str, help='directory which stores CityScapes val pred images')
    # parser.add_argument('--pred_dir', default='/home/sidd_s/scratch/mmseg_results/cityscapes_deeplabv3+_mobilenet/', type=str, help='directory which stores CityScapes val pred images')
    parser.add_argument('--gt_dir', default='/home/sidd_s/scratch/dataset/dark_zurich_val/', type=str, help='directory which stores CityScapes val gt images')
    parser.add_argument('--pred_dir', default='/home/sidd_s/scratch/dataset/dark_zurich_val/pred/dannet_PSPNet_val/')
    parser.add_argument('--devkit_dir', default='./dataset/lists', help='base directory of cityscapes')
    args = parser.parse_args()
    main(args)
