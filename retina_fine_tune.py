#import tensorflow
import detectron2
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt 
import pickle
from random import seed
import random


from tqdm import tqdm
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger

import pycocotools
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LVISEvaluator
from detectron2.data import build_detection_test_loader
from LossEvalHook import LossEvalHook
from detectron2.data import DatasetMapper


from dataReader import ReadData

random.seed(10)
setup_logger()


lr = 0.0025
batch_size = 256
n_iter = 500
EXPERIMENT_NAME = 'Random_'+str(lr)+'lr_'+str(batch_size)+'bsize_'+str(n_iter)+'iter'+'_aicity2019'


def get_bbox_from_line(line):
    data = [x.strip() for x in line.split(',')]
    x = int(data[2])
    y = int(data[3])
    w = int(data[4])
    h = int(data[5])
    [x1,y1,x2,y2] = [x,y,x+w,y+h]
    # print("")
    # print([x1,y1,x2,y2])
    # print(img.shape)
    # print("")
    return [float(x1),float(y1),float(w),float(h)]

def get_aicity_dataset(path='',sequences=''):

    #path = '/home/group09/code/week6/datasets/aic19-track1-mtmc-train/train'
    #sequences = ['S01','S04']
    
    # 2 iters for training, 1 for test
    dataset_dicts = []
    image_id = 0
    for sequence in sequences:
        # can iterate over all sequences
        for camera in os.listdir(os.path.join(path,sequence)):
            gt_path = os.path.join(path,sequence,camera,'gt','gt.txt')
            video_path = os.path.join(path,sequence,camera,'vdo.avi')
            lines = open(gt_path).read().split('\n')
            
            capture = cv2.VideoCapture(video_path)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            _,frame = capture.read()
            height, width = frame.shape[:2]
            gt_line_idx = 0

            for frame_idx in tqdm(range(frame_count)):
                
                # TODO: video to frames
                filename = str(frame_idx).zfill(4)+'.png'
                im_path = os.path.join(path,sequence,camera,'frames',filename)
                #im = cv2.imread(im_path)

                
                record = {}
                record["file_name"] = im_path
                record["image_id"] = str(image_id)#str(frame_idx).zfill(4)
                record["height"] = height
                record["width"] = width
                image_id = image_id +1


                objs=[]

                if len(lines[gt_line_idx]) > 0:
                        gt_frame_idx = int(lines[gt_line_idx][0])-1
                while frame_idx == gt_frame_idx and len(lines[gt_line_idx]) > 0:
                    [x,y,w,h] = get_bbox_from_line(lines[gt_line_idx])
                    #print([x1,y1,x2,y2])
                    # obj initialization
                    obj = {
                    "type": 'Car',
                    "bbox": [x,y,w,h],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": 0
                    }
                    objs.append(obj)
                    gt_line_idx = gt_line_idx+1
                    if len(lines[gt_line_idx]) > 0:
                        gt_frame_idx = int(lines[gt_line_idx][0])-1
                    else:
                        break
                # if frame_idx == 100:
                #     break


                record["annotations"] = objs
                dataset_dicts.append(record)
                
    seed(1)
    random.shuffle(dataset_dicts)
    return dataset_dicts
    

dataset_path = '/home/group09/code/week6/datasets/aic19-track1-mtmc-train/train'
train_sequences = ['S01','S04']
val_sequences = ['S03']
#train dataset
print("==== TRAIN SPLIT")
print("")
for d in ['train']:
    DatasetCatalog.register('train_retina', lambda d=d:get_aicity_dataset(dataset_path,train_sequences))
    MetadataCatalog.get('train_retina').set(thing_classes=['Car'])

#val dataset
print("==== VALIDATION SPLIT")
print("")
for d in ['val']:
    DatasetCatalog.register('val_retina', lambda d=d:get_aicity_dataset(dataset_path,val_sequences))
    MetadataCatalog.get('val_retina').set(thing_classes=['Car'])

train_metadata = MetadataCatalog.get("train_retina")
#dataset_dicts = get_aicity_dataset(dataset_path,train_sequences)


OUTPUT_DIR = '/home/group09/code/week6/models_retina_w5_FINAL_WH/'+EXPERIMENT_NAME

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

cfg = get_cfg()
cfg.OUTPUT_DIR = OUTPUT_DIR
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train_retina",)
cfg.DATASETS.TEST = ("val_retina",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = lr  # pick a good LR
cfg.SOLVER.MAX_ITER = n_iter    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (car)


cfg.TEST.EVAL_PERIOD = 100
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
