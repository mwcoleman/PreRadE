import torch, os, csv, base64
import detectron2, cv2
import json
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

# from detectron2.modeling import build_model
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.structures.image_list import ImageList
# from detectron2.data import transforms as T
# from detectron2.modeling.box_regression import Box2BoxTransform
# from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
# from detectron2.structures.boxes import Boxes
# from detectron2.layers import nms
# from detectron2 import model_zoo
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer

from utils import (
    Extractor,
    FeatureWriterTSV,
    PrepareImageInputs,
    collate_func
)


class RawCocoDataset(Dataset):
    """MS-COCO dataset captions only
    No transforms here, extractor class handles it"""
    def __init__(self, json_file, img_dir):
        with open(json_file) as f:
            self.metadata = json.load(f)
        self.img_dir = img_dir
    
    def __len__(self):
        return len(self.metadata['images'])
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # By list order, not image id order
        image_id = self.metadata['images'][idx]['id']
        img_name = os.path.join(self.img_dir,
                                f'{image_id:012d}.jpg')
        image = plt.imread(img_name)
        # image = cv2.resize(plt.imread(img_name), self.resize_dim, interpolation=cv2.INTER_AREA)
        # expects BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        captions = [c['caption'] for c in self.metadata['annotations'] if c['image_id']==image_id]
        sample = {'image': image, 'caption':captions, 'img_id': image_id}
        return sample




COCO_ANNOT_PATH = '/media/matt/data21/datasets/ms-coco/2017/annotations_trainval2017/captions_train2017.json'
COCO_IMG_PATH = '/media/matt/data21/datasets/ms-coco/2017/train2017'
CFG_PATH =  "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
# "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
# "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
# "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
# "COCO-Detection/retinanet_R_101_FPN_3x.yaml" Doesn't have proposal network..


BATCH_SIZE=4


if __name__=='__main__':
    
    d2_rcnn = Extractor(CFG_PATH, batch_size=BATCH_SIZE)
    dataset = RawCocoDataset(COCO_ANNOT_PATH, COCO_IMG_PATH)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_func)

    tsv_writer = FeatureWriterTSV('/media/matt/data21/mmRad/img_features/mscoco-train_2017-custom.tsv')
    
    prepare = PrepareImageInputs(d2_rcnn)
    
    # Test with mimic pic
    # image = plt.imread('./mimic_sample.jpg')
    # image = cv2.resize(plt.imread(img_name), self.resize_dim, interpolation=cv2.INTER_AREA)
    # expects BGR
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    # batch = {'images':[image]}
    # samples = prepare(batch)
    # d2_rcnn.show_sample(samples)
    # d2_rcnn.visualise_features(samples)
    
    # # ###
    import time
    start_time = time.time()
    for batch_idx, batch in enumerate(loader):
        samples = prepare(batch)
        # d2_rcnn.show_sample(samples)
        # d2_rcnn.visualise_features(samples)
        
        visual_embeds, output_boxes, num_boxes = d2_rcnn(samples)
        
        # write current batch to file
        items_dict = [{'img_id': batch['img_ids'][i],
                       'img_h': samples[1][i]['height'],
                       'img_w': samples[1][i]['width'],
                       'num_boxes': num_boxes,
                       'boxes': base64.b64encode(output_boxes[i].detach().cpu().numpy()),
                       'features': base64.b64encode(visual_embeds[i].detach().cpu().numpy())}
                       for i in range(len(samples[0]))]
        tsv_writer(items_dict)
        # tsvReader('coco-visual-features.tsv')
        if batch_idx%100==0:
            print(batch_idx)
    elapsed_time = time.time()-start_time
    print(f"Fin. Extracted features from {len(loader)} images in {elapsed_time} seconds..")
    # break