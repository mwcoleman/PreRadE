import torch, os, csv, base64, re
import detectron2, cv2
import json
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import pandas as pd

# Handle truncated images (e.g. MIMIC-CXR p13/p13187806/s59042749)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True

from utils import (
    Extractor,
    FeatureWriterTSV,
    PrepareImageInputs,
    collate_func
)

ROOT = '/media/matt/data21/datasets/'
MIMIC_IMAGE_ROOT = os.path.join(ROOT, 'mimic-cxr', 'data', 'images')
MIMIC_IMAGE_CSV = os.path.join(ROOT, 'mimic-cxr','data','cxr-record-list.csv.gz')
MIMIC_CSV_PATH = os.path.join(ROOT, 'mimic-cxr','data','metadata_edited.csv')
MIMIC_META_CSV = os.path.join(ROOT, 'mimic-cxr','data','mimic-cxr-2.0.0-metadata.csv.gz')


class RawMimicDataset(Dataset):
    """MIMIC-CXR Dataset for image feature extraction only
    """
    def __init__(self, meta_csv=MIMIC_CSV_PATH):
        self.data = pd.read_csv(meta_csv)
    
    def __len__(self):
        return len(self.data)
    
    # def get_report(self, report_path):
    #     """Returns the findings/impression string associated with the report.
    #     TODO: Janky and needs re-work to handle inconsistent reports"""

    #     fp = ROOT_DATA_DIR+'/mimic-cxr/mimic-cxr-reports/files/'+report_path
        
    #     with open(fp) as f:
    #         lines = f.read()
    #         freg = re.compile(r'FINDINGS:(.*)IMPRESSION:', re.DOTALL)
    #         mo = freg.search(lines)
    #         try:
    #             return mo.group(1)
    #         except:
    #             print("Error: No findings in report, see below:")
    #             print(lines)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_fp = os.path.join(MIMIC_IMAGE_ROOT, self.data['filepath'][idx])
        try:
            image = plt.imread(image_fp)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        except Exception as e:
            print(f"Error reading image: \n{e}\n Path: {image_fp}")
            image = None
        
        sample = {'img_id': self.data['dicom_id'][idx], 
                #   'subject_id':self.data['subject_id'][idx], 
                #   'study_id':self.data['study_id'][idx],
                #   'view':self.data['ViewPosition'][idx],
                  'caption':"",
                #   'caption':self.get_report(self.data['filepath'][idx][84:107]+'.txt'),
                  'image':image}
        return sample

class RawCubDataset(Dataset):
    """CUB dataset; image, class, caption"""
    def __init__(self, csv_file=None, img_dir=None):
        if csv_file is None:
            csv_file = self.join_cub_csv()
        self.data = pd.read_csv(csv_file)
            
        self.img_dir = img_dir
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_fp = os.path.join(self.img_dir,
                                self.data['filenames'][idx])
        image = plt.imread(image_fp)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        
        # 0 vs 1 indexed not that it matters
        sample = {'img_id': self.data['id'][idx], 
                  'caption':self.data['captions'][idx], 
                  'label':self.data['class'][idx],
                  'image':image}
        return sample
    
    def join_cub_csv(root_dir='/media/matt/data21/datasets/CUB/'):
        """"Concat files to simple csv"""
        caption_df = pd.read_csv(root_dir + 'sentences.csv')
        image_df = pd.read_csv(root_dir + 'images.txt', sep=' ', header=None)
        class_df = pd.read_csv(root_dir + 'image_class_labels.txt', sep=' ', header=None)

        image_df.rename(columns={1:'filenames', 0:'id'}, inplace=True)
        class_df.rename(columns={0:'id', 1:'class'}, inplace=True)    

        caption_dict = {}
        for _, row in caption_df.iterrows():
            caption_dict[row['filepath']] = row['captions']

        image_df['captions'] = image_df[1].apply(lambda x: caption_dict[x])

        df = pd.concat([image_df, class_df['class']], axis=1)
        df.to_csv(root_dir + 'caption_label_data.csv', index=None)
        return root_dir + 'caption_label_data.csv'

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


ROOT_DATA_DIR = ROOT
# Pretraining data ms-coco 2017 captions
COCO_ANNOT_PATH = ROOT_DATA_DIR+'ms-coco/2017/annotations_trainval2017/captions_train2017.json'
COCO_IMG_PATH = ROOT_DATA_DIR+'ms-coco/2017/train2017'

# Fine tuning data CUB
CUB_CSV_PATH = ROOT_DATA_DIR+'CUB/caption_label_data.csv'
CUB_IMG_PATH = ROOT_DATA_DIR+'CUB/images/'

# CSV contains paths to images

# MIMIC_IMG_PATH = ROOT_DATA_DIR+'mimic-cxr/physionet.org/files/mimic-cxr-jpg/2.0.0/files/'

CFG_PATH =  "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
# "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
# "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
# "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
# "COCO-Detection/retinanet_R_101_FPN_3x.yaml" Doesn't have proposal network..

BATCH_SIZE=4

if __name__=='__main__':
    

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='mimic')
    parser.add_argument('--output', default='/media/matt/data21/mmRad/img_features/mimic_all.tsv')
    parser.add_argument('--visualise', dest='visualise_examples', default=False)

    args = parser.parse_args()
    print(f"Building tsv dataset for {args.dataset} dataset. File locations given:")
    if args.dataset=='coco':
        print(f"Annotations: {COCO_ANNOT_PATH}")
        print(f"Images: {COCO_IMG_PATH}")

        dataset = RawCocoDataset(COCO_ANNOT_PATH, COCO_IMG_PATH)

    elif args.dataset=='cub':
        print(f"Annotations: {CUB_CSV_PATH}")
        print(f"Images: {CUB_IMG_PATH}")

        dataset = RawCubDataset(CUB_CSV_PATH, CUB_IMG_PATH)
        print("done")

    elif args.dataset=='mimic':
        print(f"Mimic path: {MIMIC_CSV_PATH}")

        dataset = RawMimicDataset(MIMIC_CSV_PATH)
        print("done")

    d2_rcnn = Extractor(CFG_PATH, batch_size=BATCH_SIZE)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_func)

    assert not os.path.exists(args.output), "output tsv file exists"
    tsv_writer = FeatureWriterTSV(args.output)
    
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
    num_batches = len(dataset)/BATCH_SIZE

    for batch_idx, batch in enumerate(loader):
        samples = prepare(batch)
        
        if args.visualise_examples:
            d2_rcnn.show_sample(samples)
            d2_rcnn.visualise_features(samples)
            args.visualise_examples=False
        
        visual_embeds, output_boxes, num_boxes = d2_rcnn(samples)
        
        # write current batch to file
        # img dim is resized to have shortest edge a multiple
        # of allowable detectron2 inputs, e.g. 800px
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
            print(f'Batch {batch_idx} of {num_batches} ({round((batch_idx/num_batches)*100,2)}%), time (min): {round((time.time()-start_time)/60, 2)}')
    elapsed_time = time.time()-start_time
    print(f"Fin. Extracted features from {len(loader)} images in {elapsed_time} seconds..")
    # break