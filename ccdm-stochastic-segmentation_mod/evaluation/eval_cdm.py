
import importlib
import logging
import os
import pathlib
import pprint
from dataclasses import dataclass
from typing import Union, Tuple, Optional

import ignite.distributed as idist
import numpy as np
import torch
import wandb
from PIL import Image
from ignite.contrib.handlers import ProgressBar, WandBLogger
from ignite.contrib.metrics import GpuInfo
from ignite.engine import Engine, Events
from ignite.handlers import global_step_from_engine
from ignite.metrics import ConfusionMatrix, mIoU, IoU
from ignite.utils import setup_logger

from torch import nn, Tensor
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader

from datasets.pipelines.transforms import build_transforms
from ddpm.models import DenoisingModel
from ddpm.models.one_hot_categorical import OneHotCategoricalBCHW
from ddpm.trainer import _build_model, _build_feature_cond_encoder
from ddpm.utils import expanduservars, archive_code, worker_init_fn

from .cs_eval import evaluateImgLists, args
from .utils import _flatten, create_new_directory

import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)
Model = Union[DenoisingModel, nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]

## replacing from existing torchdevice to current desired torch.device
torch_device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')


def _build_datasets(params: dict) -> Tuple[DataLoader, torch.Tensor, int, int, dict]:
    dataset_file: str = params['dataset_file']
    dataset_module = importlib.import_module(dataset_file)
    # required in params_eval.yml for using datasets/pipelines/transforms
    train_ids_to_class_names = None

    if ((dataset_file == 'datasets.cityscapes') or (dataset_file == 'datasets.ade20k') or (dataset_file == 'datasets.darkzurich')) \
            and all(['dataset_pipeline_val' in params, 'dataset_pipeline_val_settings' in params]):
        transforms_names_val = params["dataset_pipeline_val"]
        transforms_settings_val = params["dataset_pipeline_val_settings"]
        transforms_dict_val = build_transforms(transforms_names_val, transforms_settings_val, num_classes=dataset_module.get_num_classes())
        validation_dataset = dataset_module.validation_dataset(max_size=params['dataset_val_max_size'],
                                                               transforms_dict_val=transforms_dict_val)  # type: ignore

        train_ids_to_class_names = dataset_module.train_ids_to_class_names()

    else:
        train_dataset = dataset_module.training_dataset()  # type: ignore
        validation_dataset = dataset_module.validation_dataset(max_size=params['dataset_val_max_size'])  # type: ignore

    LOGGER.info("%d images in validation dataset '%s'", len(validation_dataset), dataset_file)

    # If there is no 'get_weights' function in the dataset module, create a tensor full of ones.
    get_weights = getattr(dataset_module, 'get_weights', lambda _: torch.ones(train_dataset[0][1].shape[0]))
    # class_weights = get_weights(params["class_weights"])
    class_weights = get_weights() # cause this relevant to the module being called

    batch_size = params['batch_size']  # if single_gpu or non-DDP 
    
    ## using subset of dataset being used for faster inference on results
    # lst = [3,4] # have to change later
    # validation_dataset = torch.utils.data.Subset(validation_dataset, lst)
    
    validation_loader = DataLoader(validation_dataset,
                                   batch_size=batch_size,
                                   num_workers=params["mp_loaders"],
                                   shuffle=False,
                                   worker_init_fn=worker_init_fn)

    return validation_loader, \
           class_weights, \
           dataset_module.get_ignore_class(), \
           dataset_module.get_num_classes(), \
           train_ids_to_class_names


@dataclass
class Evaluator:
    eligible_eval_resolutions = ["original", "dataloader"]
    eligible_eval_vote_strategies = ["majority", "confidence"]

    # original => compute miou wrt to original labels
    # dataloader => compute miou wrt to dataloader (potentially resized, cropped etc) labels

    def __init__(self,
                 model: Model,
                 average_model: Model,
                 feature_cond_encoder: Model,
                 params: dict,
                 num_classes: int,
                 ignore: int):

        self.params = params
        self.feature_cond_encoder = feature_cond_encoder
        if isinstance(self.feature_cond_encoder, nn.Module): 
            self.feature_cond_encoder = feature_cond_encoder.to(torch_device) # changing model gpu 
        self.model = _flatten(model)
        self.average_model = _flatten(average_model)
        self.checkpoint_dir = params['output_path'] # to get results in the specified file 
        self.dataset_module = importlib.import_module(params['dataset_file'])
        self.dataset_module_config = importlib.import_module(params['dataset_file'] + '_config')
        self.pred_list = []
        self.label_list = []
        self.images_cnt = 0
        self.num_classes = num_classes  # USED CLASSES + 1 (IGNORE)
        self.ignore = ignore  # ASSUMED TO BE num_classes-1
        assert self.ignore == (self.num_classes - 1), f"Invalid ignore or num_classes" \
                                                      f" assumed ignore = num_classes-1" \
                                                      f" but got ignore = {ignore} and num_classes = {num_classes}"

        self.eval_settings = self.params.get("evaluation", {})
        self.eval_resolution = self.eval_settings.get("resolution", "dataloader")
        self.eval_voting_strategy = self.eval_settings.get("evaluation_vote_strategy", 'confidence')
        self.num_evaluations = self.eval_settings.get("evaluations")

        assert (self.eval_resolution in self.eligible_eval_resolutions), f"eval_resolution={self.eval_resolution} " \
                                                                         f"in not in {self.eligible_eval_resolutions}"
        assert (self.eval_voting_strategy in self.eligible_eval_vote_strategies), f"eval_voting_strategy={self.eval_voting_strategy} " \
                                                                                  f"in not in {self.eligible_eval_vote_strategies}"

        # compute cm with torch-only code
        self.cm = torch.zeros(size=(num_classes - 1, num_classes - 1), device=torch_device) 
        
        ## changing the existing the gpu code to desired one to work with
        self.model = self.model.to(torch_device) 
        self.average_model = self.average_model.to(torch_device)

    def load(self, filename: str):
        LOGGER.info("Loading state from %s...", filename)
        checkpoint = torch.load(filename, map_location=torch_device)
        self.load_objects(checkpoint)
        # v = pathlib.Path(filename)
        # self.checkpoint_dir = str(v.parent) #  specifying it earlier 

    def load_objects(self, checkpoint: dict, strict=True):
        self.model.unet.load_state_dict(checkpoint["model"], strict)

        try:
            # try average encoder first
            self.feature_cond_encoder.load_state_dict(checkpoint["average_feature_cond_encoder"], strict)
        except:
            LOGGER.info(f"no average_feature_cond_encoder found in checkpoint with entries {checkpoint.keys()}")
            try:
                self.feature_cond_encoder.load_state_dict(checkpoint["feature_cond_encoder"], strict)
            except:
                LOGGER.info(f"no feature_cond_encoder found in checkpoint with entries {checkpoint.keys()}")

        self.average_model.unet.load_state_dict(checkpoint["average_model"], strict)

    @property
    def time_steps(self):
        return self.model.time_steps

    @property
    def diffusion_model(self):
        return self.model.diffusion  # fixme is it any different to use self.average_model instead
    
    @torch.no_grad()
    def predict_condition(self, x: Tensor) -> Tensor:
        # x is the image which will be return as it is 
        return x

    @torch.no_grad()
    def predict_feature_condition(self, x: Tensor) -> Tensor: 
        # from here only i can have to decide whether to do eval on this or not by checking nn module or not 
        if isinstance(self.feature_cond_encoder, nn.Module): 
            self.feature_cond_encoder.eval()  
        return self.feature_cond_encoder(x)

    @torch.no_grad()    
    def predict_single(self, image, condition, feature_condition, label_ref_logits=None, names = None, params = None):
        # predict a single segmentation (BNHW) for image (BCHW) where N = num_classes
        label_shape = (image.shape[0], self.num_classes, *image.shape[2:])
        # if isinstance(names, tuple) and names[0]: # assuming bt = 1 
            
            ## MIC val results >>> mIoU = 34.8 (with passing resized image 256x512 to MIC initially) >> coming to 7.6 in conditional posterior generation
            # xt = torch.tensor(np.array(Image.open(os.path.join('/home/sidd_s/MIC_mod/seg/labelTrainIds', names[0])).convert('P')))
            
            ## starting from DZ VAL GT >>> mIoU = 83.8 (128x256) and mIoU = 85.9 (256x512) >> 7.3 in conditional posterior generation (without feat enc) and  7.6 (with feat enc on size(256x512)) 
            # print(names[0])
            # name = str(names[0]).replace('_rgb_anon.png', '_gt_labelTrainIds.png')
            # xt = torch.tensor(np.array(Image.open(os.path.join('/home/sidd_s/scratch/dataset/dark_zurich/gt/val/night/GOPR0356', name)).convert('P')))
            # xt[xt==255] = 19
                        
            ## DZ val results >>> mIoU = 36.3 (128x256) and 36.5 (256x512) >> coming to 7.3 in conditional posterior generation (without feat enc) and  7.6 (with feat enc on size(256x512)) 
            # xt = torch.tensor(np.array(Image.open(os.path.join('/home/sidd_s/scratch/dataset/dark_zurich_val_morepred/pred/dannet_PSPNet_val', names[0])).convert('P')))
            
            # # resizing to 128 x256
            # xt = xt.unsqueeze(dim=0).unsqueeze(dim=0)
            # # # print('>>>>', xt.shape) 
            # xt = F.interpolate(xt, size=(128,256)) 
            # # xt = F.interpolate(xt, size=(256,512)) 
            # # print('>>>>', xt.shape)
            
            # xt = one_hot(xt.squeeze().long(), self.num_classes).to(torch_device)
            # xt = xt.permute((2,0,1)).unsqueeze(dim=0)
            # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        # else:
            ## random onehots >>> mIoU = 1.6 and 1.6 (same in both sizes) >> coming to 7.1 in conditional posterior generation (128x256) and 7.5 in (256x512)
            # xt = OneHotCategoricalBCHW(logits=torch.zeros(label_shape, device=image.device)).sample() # sampling from P(X_T | I) {one hot vector} # instead of this use one hot label vector from the predicted the label output by Domain adpapted model in order to improve upon that 
        
        # have to remove later
        # xt = OneHotCategoricalBCHW(logits=torch.zeros(label_shape, device=image.device)).sample()
        # prediction = xt 
        
        ### original 
        xt = OneHotCategoricalBCHW(logits=torch.zeros(label_shape, device=image.device)).sample()
        prediction = self.predict(xt, condition, feature_condition, label_ref_logits) 
        ### original  
        
        return prediction

    @torch.no_grad()
    def predict(self, xt: Tensor, condition: Tensor, feature_condition: Tensor, label_ref_logits: Optional[Tensor] = None, params = None) -> Tensor:
        self.average_model.eval()
        if isinstance(self.feature_cond_encoder, nn.Module):  # addtional code for ensuring that, feature conditional encoder is used only when it is there
            self.feature_cond_encoder.eval() 
        ret = self.average_model(x=xt, condition=condition, feature_condition=feature_condition, label_ref_logits=label_ref_logits, params= params)

        assert ("diffusion_out" in ret)
        return ret["diffusion_out"]

    @torch.no_grad()
    def predict_multiple(self, image, condition, feature_condition, names = None, params = None):
        assert (self.num_evaluations > 1), f'predict_multiple assumes evaluations > 1 instead got {self.evaluations}'
        # predict a params['evaluations'] * segmentations each of shape (BNHW) for image (BCHW) where N = num_classes
        for i in range(self.num_evaluations):
            prediction_onehot_i = self.predict_single(image, condition, feature_condition, names=names, params=params)
            if self.eval_voting_strategy == 'confidence':
                if i == 0:
                    prediction_onehot_total = torch.zeros_like(prediction_onehot_i, dtype=float)
                prediction_onehot_total += prediction_onehot_i * (1 / self.num_evaluations)

            elif self.eval_voting_strategy == 'majority':
                raise NotImplementedError()

            else:
                raise ValueError()

        return prediction_onehot_total

    @torch.no_grad()
    def infer_step(self, engine: Engine, batch: Tensor):  # -> Tuple[Tensor]:
        # cdm only inference step

        # prep data
        image, label, label_orig, names = batch ## for now, it is hard coded, later need to change for Darkzurich instance dataset and for other datasets 
        image = image.to(torch_device)    
        label_onehot = label.to(torch_device)
        label = label_onehot.argmax(dim=1).long()  # one_hot to int, (BHW)

        # forward step
        condition = self.predict_condition(image)  # condition is the image itself  # I have made this up but it is consistent with the paper
        feature_condition = self.predict_feature_condition(image) 
        if self.num_evaluations == 1:
            prediction_onehot = self.predict_single(image, condition, feature_condition, names=names, params=self.params)  # (BNHW)
        else:
            prediction_onehot = self.predict_multiple(image, condition, feature_condition, names=names, params=self.params)
            # add predict_multiple

        # debug only shows 1st element of the batch
        # pil_from_bchw_tensor_label(prediction_onehot).show()
        # pil_from_bchw_tensor_label(label_onehot).show()
        # pil_from_bchw_tensor_image(Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)).show()

        # prep prediction and label for IOU/MIOU computation
        # prediction = prediction_onehot.argmax(dim=1).long()
        if self.eval_resolution == 'original':
            # replace label with original labels of shape (B H_orig W_orig)
            # upsample prediction_onehot to (H_orig,W_orig) with bilinear interpolation
            label = label_orig.to(torch_device)
            b, h_orig, w_orig = label.shape
            prediction_onehot = torch.nn.functional.interpolate(prediction_onehot, (h_orig, w_orig), mode='bilinear')

        prediction_onehot = prediction_onehot[:, 0:self.num_classes - 1, ...]  # removing ignore class channel
        self.update_cm(prediction_onehot, label)
        self.save_preds(label.long(), prediction_onehot.argmax(dim=1).long())
        self.images_cnt += image.shape[0]

        return {"y": label, "y_pred": prediction_onehot}

    def save_preds(self, label: Tensor, pred: Tensor):
        # pred is in train_id format
        # this function saves predictions for cityscapes script to use them aftewards
        assert label.dtype == torch.long
        assert pred.dtype == torch.long
        assert self.checkpoint_dir is not None, f'saving preds in checkpoint_dir but it is {self.checkpoint_dir}'

        # 'val' word in paths is dataset specific and should be taken from an attribute of the Evaluator
        path_submit = str(pathlib.Path(self.checkpoint_dir) / 'outputs' / 'val' / 'submit')
        path_debug = str(pathlib.Path(self.checkpoint_dir) / 'outputs' / 'val' / 'debug')
        path_label = str(pathlib.Path(self.checkpoint_dir) / 'outputs' / 'val' / 'label')

        create_new_directory(path_submit)
        create_new_directory(path_debug)
        create_new_directory(path_label)

        pred_submit = self.dataset_module_config.map_train_id_to_id(pred.cpu().clone())
        pred_debug = self.dataset_module_config.decode_target_to_color(pred.cpu().clone())
        label_submit = self.dataset_module_config.map_train_id_to_id(label.cpu().clone())
        #
        # Image.fromarray(pred_submit[0].cpu().numpy().astype(np.uint8)).show()
        # Image.fromarray(pred_debug[0].cpu().numpy().astype(np.uint8)).show()
        # Image.fromarray(label_submit[0].cpu().numpy().astype(np.uint8)).show()

        # write custom collate to allow batched passing of filenames from dataloader,
        # otherwise such metadata can only be passed out with batch_size=1
        for i in range(pred_submit.shape[0]):
            path_filename_submit = \
                str(pathlib.Path(path_submit) / f'{self.images_cnt + i + 1}_id.png')
            path_filename_debug = \
                str(pathlib.Path(path_debug) / f'{self.images_cnt + i + 1}_rgb.png')
            path_filename_label_submit = \
                str(pathlib.Path(path_label) / f'{self.images_cnt + i + 1}_label.png')

            Image.fromarray(pred_submit[i].cpu().numpy().astype(np.uint8)).save(path_filename_submit)
            Image.fromarray(pred_debug[i].cpu().numpy().astype(np.uint8)).save(path_filename_debug)
            Image.fromarray(label_submit[i].cpu().numpy().astype(np.uint8)).save(path_filename_label_submit)

            LOGGER.info(f"saved pred {i} from batch shape {pred_submit.shape} with "
                        f"id format [{path_filename_submit} "
                        f"and color at {path_filename_debug}")

            LOGGER.info(f"saved label {i} from batch shape {label_submit.shape} with "
                        f"id format [{path_filename_label_submit} ")

            self.pred_list.append(path_filename_submit)
            self.label_list.append(path_filename_label_submit)

    def update_cm(self, prediction: torch.Tensor, target: torch.Tensor):
        with torch.no_grad():
            num_classes = prediction.shape[1]  # prediction is shape NCHW, we want C (one-hot length of all classes)
            p = prediction.transpose(1, 0)  # Prediction is NCHW -> CNHWF
            p = p.contiguous().view(num_classes, -1)  # Prediction is [C, N*H*W]
            t = target.view(-1).to(torch.int64)  # target is now [N*H*W]
            one_hot_target = one_hot(t, num_classes + 1)
            one_hot_target = one_hot_target[:, :-1]
            confusion_matrix = torch.matmul(p.float(), one_hot_target.float()).to(torch.int)
            # [C, N*H*W] x [N*H*W, C] = [C, C]
            self.cm.to(confusion_matrix.device)
            self.cm += confusion_matrix

    def normalize_cm(self, mode: str):
        # not used anywhere
        with torch.no_grad():
            if mode == 'row':
                row_sums = torch.sum(self.cm, dim=1, dtype=torch.float)
                row_sums[row_sums == 0] = 1  # to avoid division by 0. Safe, because if sum = 0, all elements are 0 too
                norm_matrix = self.cm.to(torch.float) / row_sums.unsqueeze(1)
            elif mode == 'col':
                col_sums = torch.sum(self.cm, dim=0, dtype=torch.float)
                col_sums[col_sums == 0] = 1  # to avoid division by 0. Safe, because if sum = 0, all elements are 0 too
                norm_matrix = self.cm.to(torch.float) / col_sums.unsqueeze(0)
            else:
                raise ValueError("Normalise confusion matrix: mode needs to be either 'row' or 'col'.")
            return norm_matrix

    def get_miou_and_ious(self):
        # cm = self.normalize_cm(mode='col')
        cm = self.cm
        indices = [i for i in range(self.num_classes - 1)]
        with torch.no_grad():
            diagonal = cm.diag()[indices].to(torch.float)
            row_sum = torch.sum(cm, dim=0, dtype=torch.float)[indices]
            col_sum = torch.sum(cm, dim=1, dtype=torch.float)[indices]
            denominator = row_sum + col_sum - diagonal
            iou = diagonal / denominator
            iou[iou != iou] = 0  # if iou of some class is Nan (i.e denominator was 0) set to 0 to avoid Nan in the mean
            mean_iou = iou.mean()
            return mean_iou, iou


def build_engine(evaluator: Evaluator,
                 num_classes: int,
                 ignore_class: int,
                 params: dict,
                 train_ids_to_class_names: Union[None, dict] = None) -> Engine:
    engine_test = Engine(evaluator.infer_step)
    GpuInfo().attach(engine_test, "gpu")
    LOGGER.info(f"Ignore class {ignore_class} in IoU evaluation...")
    cm = ConfusionMatrix(num_classes=num_classes - 1)  # 0-18
    cm.attach(engine_test, 'cm')
    IoU(cm).attach(engine_test, "IoU")
    mIoU(cm).attach(engine_test, "mIoU")
    if idist.get_local_rank() == 0:
        ProgressBar(persist=True).attach(engine_test)
        if params["wandb"]:
            tb_logger = WandBLogger(project=params["wandb_project"], entity='cdm', config=params)
            tb_logger.attach_output_handler(
                engine_test,
                Events.EPOCH_COMPLETED,
                tag="testing",
                metric_names=["mIoU", "IoU"],
                global_step_transform=global_step_from_engine(engine_test, Events.ITERATION_COMPLETED)
            )

    @engine_test.on(Events.ITERATION_COMPLETED)
    @idist.one_rank_only(rank=0, with_barrier=True)
    def log_eval(et: Engine):
        LOGGER.info(f"{et.state.metrics}")

    @engine_test.on(Events.EPOCH_COMPLETED)
    @idist.one_rank_only(rank=0, with_barrier=True)
    def log_iou(et: Engine):
        LOGGER.info("mIoU score: %.4g", et.state.metrics["mIoU"])
        if isinstance(train_ids_to_class_names, dict):
            per_class_ious = [(train_ids_to_class_names[i], iou.item()) for i, iou in
                              enumerate(engine_test.state.metrics["IoU"])]

        else:
            per_class_ious = [(i, iou.item()) for i, iou in enumerate(engine_test.state.metrics["IoU"])]

        LOGGER.info(f"val IoU scores per class:{per_class_ious}")
        if params["wandb"]:
            wandb.log({"mIoU_val": et.state.metrics["mIoU"]})

    return engine_test


def run_inference(params: dict, params_path: str):
    setup_logger(name=None, format="\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s", reset=True)
    LOGGER.info("%d GPUs available", torch.cuda.device_count())
    # Create output folder and archive the current code and the parameters there
    output_path = expanduservars(params['output_path'])
    os.makedirs(output_path, exist_ok=True)
    LOGGER.info("experiment dir: %s", output_path)
    archive_code(output_path, params_path)
    LOGGER.info("Inference params:\n%s", pprint.pformat(params))

    # Load the datasets
    data_loader, _, ignore_class, num_classes, train_ids_to_class_names = _build_datasets(params)
    assert (hasattr(data_loader, "dataset"))
    if hasattr(data_loader.dataset, "return_metadata"):
        data_loader.dataset.return_metadata = True
    elif hasattr(data_loader.dataset.dataset, "return_metadata"):  # in case Subset of dataset is used
        data_loader.dataset.dataset.return_metadata = True
    else:
        raise ValueError()
    # build evaluator
    # cdm_only = params["cdm_only"] # also not relevant to module being called upon

    # eval_h_labels = 128
    # eval_w_labels = 256

    eval_h_model = params['dataset_pipeline_val_settings']['target_size'][0]
    eval_w_model = params['dataset_pipeline_val_settings']['target_size'][1]

    LOGGER.info(f"Expecting image resolution of {(eval_h_model, eval_w_model)} to build model.")
    input_shapes = [(3, eval_h_model, eval_w_model), (num_classes, eval_h_model, eval_w_model)] # image of rgb (3 x h x w), and then labels of num_classes x h x w

    cond_encoded_shape = input_shapes[0]

    feature_cond_encoder = _build_feature_cond_encoder(params)
    # changing tuple into invidual elements depending upon the presence or absence of feature_conditional_encoder
    if feature_cond_encoder[0]:
        feature_cond_encoder = feature_cond_encoder[0]
    else: 
        feature_cond_encoder = feature_cond_encoder[1] 

    model, average_model = [_build_model(params, input_shapes, cond_encoded_shape) for _ in range(2)]
    evaluator = Evaluator(model, average_model, feature_cond_encoder, params, num_classes, ignore_class)
    engine_test = build_engine(evaluator, num_classes, ignore_class, params, train_ids_to_class_names)

    load_from = params.get('load_from', None)
    if load_from is not None:
        load_from = expanduservars(load_from)
        evaluator.load(load_from)

    engine_test.run(data_loader, max_epochs=1)
    print([p for p in zip(sorted(evaluator.pred_list), sorted(evaluator.label_list))])

    check = (engine_test.state.metrics['cm'].to(torch_device) == evaluator.cm).sum()
    print(check)
    miou, ious = evaluator.get_miou_and_ious()
    LOGGER.info(f"my miou is {miou} and ious per class are {[(train_ids_to_class_names[i], round(iou.item(), 4)) for i, iou in enumerate(ious)]}")
    args.evalInstLevelScore = False
    args.evalPixelAccuracy = True
    args.JSONOutput = False
    results = evaluateImgLists(sorted(evaluator.pred_list), sorted(evaluator.label_list), args, lambda x: torch.as_tensor(x))
    print(results)
    import json
    results_json = json.dumps(results, indent=2, sort_keys=True)
    with open(str(pathlib.Path(evaluator.checkpoint_dir) / 'cs_script_results.json'), 'w') as json_file:
        json_file.write(results_json)
    a = 1
