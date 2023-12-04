import clip
import numpy as np
import os
import sys
from pycocotools.coco import COCO
from PIL import Image
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField, NDArrayField

COCO_DATA_PATH = "/mnt/cfs/datasets/coco2014"

def write_beton(ds, save_path, resolution, num_captions, caption_length, num_workers=1):
    writer = DatasetWriter(save_path, {
                'image': RGBImageField(write_mode='raw',
                                    max_resolution=resolution,
                                    smart_threshold=2_000_000),
                'label': NDArrayField(shape=(num_captions, caption_length, ), dtype=np.dtype('int32')),
            }, num_workers=num_workers)
    writer.from_indexed_dataset(ds, chunksize=100)


class CocoCaption():
    def __init__(self, coco_root=COCO_DATA_PATH, split='train', tokenize=False, num_captions=5):
        if split=='val':
            dataType = 'val2014'
        else:
            dataType = 'train2014'
            
        annFile = os.path.join(coco_root, 'annotations', f"captions_{dataType}.json")
        self.imgdir = os.path.join(COCO_DATA_PATH, 'images', dataType)
        self.coco_caps = COCO(annFile)
        self.img_ids = list(self.coco_caps.imgs.keys())
        self.tokenize = tokenize
        self.num_captions = num_captions
        
    def __len__(self):
        return len(self.img_ids)
    
    def _get_image(self, img_id):
        img_dict = self.coco_caps.loadImgs([img_id])[0]
        path = os.path.join(self.imgdir, img_dict['file_name'])
        return Image.open(path).convert('RGB')
    
    def _get_captions(self, img_id):
        img_dict = self.coco_caps.loadImgs([img_id])[0]
        annIds = self.coco_caps.getAnnIds(imgIds=img_dict['id']);
        anns = self.coco_caps.loadAnns(annIds)
        return [u['caption'] for u in anns]
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        image = self._get_image(img_id)
        captions = self._get_captions(img_id)
        if self.tokenize:
            captions = clip.tokenize(captions).numpy()
        #return image, captions
        return (image, captions[:self.num_captions])