# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

import os
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import mmcv


def main():
    parser = ArgumentParser()
    parser.add_argument('--folder', default='/raid/ai24resch01002/datasets/darkzurich/rgb_anon/val/night/GOPR0356', help='Image file')
    parser.add_argument('--config', default='/raid/ai24resch01002/Diffusion_conditional_prior_segmentation/DDP/segmentation/configs/ours_dginstyle_dark/dark_zurich_val_and_dginstyle_train.py', help='Config file')
    parser.add_argument('--checkpoint', default='/raid/ai24resch01002/saved_models/ind_dsp/DGInstyle_with_DZ/best_mIoU_iter_6500.pth', help='Checkpoint file')
    parser.add_argument('--outfolder', default='/raid/ai24resch01002/predictions/ind_dsp_with_gt', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.7,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()
    
    # Specify the directory containing images
    image_folder = args.folder
    output_folder = args.outfolder
    
    # Create an empty list to store loaded images
    images = []
    
    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    # Loop through all files in the directory
    for filename in tqdm(os.listdir(image_folder)):
        # Create the full file path
        file_path = os.path.join(image_folder, filename)
        
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            try: 
                # test a single image
                result = inference_segmentor(model, file_path)
                save_path = file_path.replace(image_folder, output_folder)
                if hasattr(model, 'module'):
                    model = model.module
                img = model.show_result(
                    file_path, result, palette=get_palette(args.palette), show=False, opacity=args.opacity)
                plt.figure(figsize=(15, 10))
                plt.imshow(mmcv.bgr2rgb(img))
                plt.title('')
                plt.tight_layout()
                mmcv.imwrite(img, save_path)
            
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    # Print total number of images loaded
    print(f"Total images loaded: {len(images)}")


if __name__ == '__main__':
    main()
