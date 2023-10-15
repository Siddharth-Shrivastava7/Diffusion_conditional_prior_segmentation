# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import matplotlib.pyplot as plt
import mmcv
import numpy as np
from matplotlib.ticker import MultipleLocator
from mmcv import Config, DictAction

from mmseg.datasets import build_dataset

from work.custom_pipeline import CityTransform        ## perturbing cityscapes

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix from segmentation results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test .pkl result')
    parser.add_argument(
        'save_path', help='path where confusion matrix will be saved')
    parser.add_argument(
        '--show', action='store_true', help='show confusion matrix')
    parser.add_argument(
        '--color-theme',
        default='winter',
        help='theme of the matrix color map')
    parser.add_argument(
        '--title',
        default='Normalized Confusion Matrix',
        help='title of the matrix color map')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.') 
    parser.add_argument('--adjacency', action='store_true')
    parser.add_argument('--similarity_confusion_transpose', action='store_true')
    args = parser.parse_args()
    return args


def calculate_confusion_matrix(dataset, results, args = None):
    """Calculate the confusion matrix.

    Args:
        dataset (Dataset): Test or val dataset.
        results (list[ndarray]): A list of segmentation results in each image.
    """
    n = len(dataset.CLASSES)
    confusion_matrix = np.zeros(shape=[n, n])
    assert len(dataset) == len(results)
    ignore_index = dataset.ignore_index
    prog_bar = mmcv.ProgressBar(len(results))
    for idx, per_img_res in enumerate(results):
        res_segm = per_img_res
        gt_segm = dataset.get_gt_seg_map_by_idx(idx).astype(int)
        gt_segm, res_segm = gt_segm.flatten(), res_segm.flatten()
        to_ignore = gt_segm == ignore_index
        gt_segm, res_segm = gt_segm[~to_ignore], res_segm[~to_ignore]
        inds = n * gt_segm + res_segm
        mat = np.bincount(inds, minlength=n**2).reshape(n, n)
        confusion_matrix += mat
        prog_bar.update()
    
    if args:
        ## saving confusion matrix as well 
        print("saving confusion matrix in its numpy format!")
        np.save((args.save_path).replace('.png', '.npy'), confusion_matrix)
    return confusion_matrix


def calculate_adjacency_matrix(confusion_matrix, k=3):
    ## calculate adjacency_matrix from confusion matrix 
    np.fill_diagonal(confusion_matrix,0) ## removing the dependency of the class with itself  ## its is an inplace argument 
    # indices = np.argpartition(confusion_matrix, -k, axis=1)[:, -k:]
    # top_k_values = np.take_along_axis(confusion_matrix, indices, axis=1)
    sorted_indices = np.argsort(confusion_matrix) 
    indices_as_ranks = np.argsort(np.argsort(confusion_matrix)) 
    indices_which_are_topk_as_bool = (indices_as_ranks >= confusion_matrix.shape[1] - k) # shape[1] used as we need to find it along row 
    adjacency_matrix = confusion_matrix * indices_which_are_topk_as_bool
    # print(adjacency_matrix)   
    ## for one hot adjacency matrix 
    adjacency_matrix_as_one_hot = adjacency_matrix.copy()
    adjacency_matrix_as_one_hot[adjacency_matrix_as_one_hot>0] = 1 
    
    return adjacency_matrix_as_one_hot

def plot_confusion_matrix(confusion_matrix,
                          labels,
                          save_path=None,
                          show=True,
                          title='Normalized Confusion Matrix',
                          color_theme='winter'):
    """Draw confusion matrix with matplotlib.

    Args:
        confusion_matrix (ndarray): The confusion matrix.
        labels (list[str]): List of class names.
        save_path (str|optional): If set, save the confusion matrix plot to the
            given path. Default: None.
        show (bool): Whether to show the plot. Default: True.
        title (str): Title of the plot. Default: `Normalized Confusion Matrix`.
        color_theme (str): Theme of the matrix color map. Default: `winter`.
    """
    # normalize the confusion matrix
    per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
    confusion_matrix = \
        confusion_matrix.astype(np.float32) / per_label_sums * 100

    num_classes = len(labels)
    fig, ax = plt.subplots(
        figsize=(num_classes * 0.8, num_classes * 0.8), dpi=180)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix, cmap=cmap)
    plt.colorbar(mappable=im, ax=ax)

    title_font = {'weight': 'bold', 'size': 12}
    ax.set_title(title, fontdict=title_font)
    label_font = {'size': 10}
    plt.ylabel('Ground Truth Label', fontdict=label_font)
    plt.xlabel('Prediction Label', fontdict=label_font)

    # draw locator
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    # draw grid
    ax.grid(True, which='minor', linestyle='-')

    # draw label
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.tick_params(
        axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    # draw confusion matrix value
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                '{}%'.format(
                    round(confusion_matrix[i, j], 2
                          ) if not np.isnan(confusion_matrix[i, j]) else -1),
                ha='center',
                va='center',
                color='w',
                size=10)

    ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(
            os.path.join(save_path), format='png')
    if show:
        plt.show()


def plot_similarity_confusion_matrix(confusion_matrix,
                          labels,
                          save_path=None,
                          show=True,
                          title='Normalized Confusion Matrix',
                          color_theme='winter'):
    """Draw confusion matrix with matplotlib.

    Args:
        confusion_matrix (ndarray): The confusion matrix.
        labels (list[str]): List of class names.
        save_path (str|optional): If set, save the confusion matrix plot to the
            given path. Default: None.
        show (bool): Whether to show the plot. Default: True.
        title (str): Title of the plot. Default: `Normalized Confusion Matrix`.
        color_theme (str): Theme of the matrix color map. Default: `winter`.
    """
    
    similarity_confusion_matrix = confusion_matrix.T ## Transposing the original confusion < for comparing FP vs FN > 
    # ## now normalising the diagonal terms with (FP + FN + TP) and off diagnoal terms with ? 
    # similarity_confusion_matrix = confusion_matrix.copy()
    
    
    # normalize the confusion matrix
    per_label_sums = similarity_confusion_matrix.sum(axis=1)[:, np.newaxis]
    similarity_confusion_matrix = \
        similarity_confusion_matrix.astype(np.float64) / per_label_sums * 100

    

    num_classes = len(labels)
    fig, ax = plt.subplots(
        figsize=(num_classes*0.8, num_classes*0.8), dpi=180)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(similarity_confusion_matrix, cmap=cmap)
    plt.colorbar(mappable=im, ax=ax)

    title_font = {'weight': 'bold', 'size': 12}
    ax.set_title(title, fontdict=title_font)
    label_font = {'size': 10}
    plt.ylabel('Prediction Label', fontdict=label_font)
    plt.xlabel('Ground Truth Label', fontdict=label_font)

    # draw locator
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    # draw grid
    ax.grid(True, which='minor', linestyle='-')

    # draw label
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.tick_params(
        axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    # draw confusion matrix value
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                '{}%'.format(
                    round(similarity_confusion_matrix[i, j], 2
                          ) if not np.isnan(similarity_confusion_matrix[i, j]) else -1),
                ha='center',
                va='center',
                color='w',
                size=10)

    ax.set_ylim(len(similarity_confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(
            os.path.join(save_path), format='png')
    if show:
        plt.show()

def calculate_confusion_matrix_segformerb2(): 
    
    args = parse_args()
    
    cfg = Config.fromfile('/home/sidd_s/Diffusion_conditional_prior_segmentation/DDP/segmentation/configs/_base_/datasets/cityscapes.py')
    results = mmcv.load('/home/sidd_s/scratch/results/segformer/cityscapes/results_images.pickle')

    assert isinstance(results, list)
    if isinstance(results[0], np.ndarray):
        pass
    else:
        raise TypeError('invalid type of prediction results')

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    dataset = build_dataset(cfg.data.test)
    confusion_matrix = calculate_confusion_matrix(dataset, results, args) 
    return confusion_matrix
        
def plot_adjacency_matrix(adjacency_matrix,
                          labels,
                          save_path=None,
                          show=True,
                          title='Normalized Adjacency Matrix',
                          color_theme='winter'):
    """Draw adjacency matrix with matplotlib.

    Args:
        adjacency_matrix (ndarray): The adjacency matrix.
        labels (list[str]): List of class names.
        save_path (str|optional): If set, save the adjacency matrix plot to the
            given path. Default: None.
        show (bool): Whether to show the plot. Default: True.
        title (str): Title of the plot. Default: `Normalized adjacency Matrix`.
        color_theme (str): Theme of the matrix color map. Default: `winter`.
    """
    # # normalize the adjacency matrix
    per_label_sums = adjacency_matrix.sum(axis=1)[:, np.newaxis]
    adjacency_matrix = \
        adjacency_matrix.astype(np.float32) / per_label_sums * 100
    # adjacency_matrix = adjacency_matrix/np.linalg.norm(adjacency_matrix, ord=2, axis=1, keepdims=True)  
    # print(adjacency_matrix)
    
    num_classes = len(labels)
    fig, ax = plt.subplots(
        figsize=(num_classes*0.8, num_classes*0.8), dpi=180)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(adjacency_matrix, cmap=cmap)
    plt.colorbar(mappable=im, ax=ax)

    title_font = {'weight': 'bold', 'size': 12}
    ax.set_title(title, fontdict=title_font)
    label_font = {'size': 10}
    plt.ylabel('Ground Truth Label', fontdict=label_font)
    plt.xlabel('Prediction Label', fontdict=label_font)

    # draw locator
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    # draw grid
    ax.grid(True, which='minor', linestyle='-')

    # draw label
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.tick_params(
        axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    # draw confusion matrix value
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                '{}%'.format(
                    round(adjacency_matrix[i, j], 2
                          ) if not np.isnan(adjacency_matrix[i, j]) else -1),
                ha='center',
                va='center',
                color='w',
                size=10)

    ax.set_ylim(len(adjacency_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(
            os.path.join(save_path), format='png')
    if show:
        plt.show()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    results = mmcv.load(args.prediction_path)

    assert isinstance(results, list)
    if isinstance(results[0], np.ndarray):
        pass
    else:
        raise TypeError('invalid type of prediction results')

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    dataset = build_dataset(cfg.data.test)
    ## calculate confusion matrix 
    confusion_matrix = calculate_confusion_matrix(dataset, results, args)
    
    if args.adjacency: 
        adjacency_matrix = calculate_adjacency_matrix(confusion_matrix, k=3) # 3 nearest neighbour adjacency matrix 
        ## adjacency matrix 
        plot_adjacency_matrix(
            adjacency_matrix,
            dataset.CLASSES,
            save_path=args.save_path,
            show=args.show,
            title=args.title,
            color_theme=args.color_theme)
    else:
        if args.similarity_confusion_transpose:
             ## similarity confusion matrix :: transformed / modified 
            plot_similarity_confusion_matrix(
                confusion_matrix,
                dataset.CLASSES,
                save_path=args.save_path,
                show=args.show,
                title=args.title,
                color_theme=args.color_theme)
        else: 
            # confusion matrix :: original 
            plot_confusion_matrix(
                confusion_matrix,
                dataset.CLASSES,
                save_path=args.save_path,
                show=args.show,
                title=args.title,
                color_theme=args.color_theme)

       
        
                
if __name__ == '__main__':
    main()
