import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import os
import pycocotools
import torchvision

class BurnedCAD_is(Dataset):
    def __init__(self, root, annFile, mode = 'train', transform = None):
        super().__init__()
        self.mode = mode
        self.transform = transform # transform : albumentations
        self.root = root
        self.coco = pycocotools.coco.COCO(annFile)
        
        
        # dont refer to it
        # load label info
        self.cat_ids = self.coco.getCatIds() # category id [1,...,90]: len: 80, e.g, params: catNms=['person']
        self.cats = self.coco.loadCats(self.cat_ids) # [{'supercategory': 'person', 'id': 1, 'name': 'person'},...]
        
        self.classNameList = ['Backgroud'] # class name: ['Backgroud', 'person', 'bicycle',...], len:80
        for i in range(len(self.cat_ids)):
            self.classNameList.append(self.cats[i]['name'])
        
        # only seg. info
        self.ids = list(sorted(self.coco.imgs.keys())) # [139, 285, 632, ...]
        fil_ids = []
        for _id in self.ids:
            targets = self._load_target(_id)
            fil_targets = self._filter_targets(targets)
            if len(fil_targets) > 0:
                fil_ids.append(_id)
        self.ids = fil_ids
        
        
        
    def _load_image(self,id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        image = cv2.imread(os.path.join(self.root, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # bgr > rgb
        return image
    
    def _load_target(self, id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    
    def _filter_targets(self,targets):
        # only seg. info.
        fil_targets = []
        for target in targets:
            if len(target['segmentation'])>0:
                target['category_id'] = 1 # 
                fil_targets.append(target)
        return fil_targets
        
    def unify_targets(self,targets):
        bboxes = []
        masks = []
        labels = []
        for target in targets:

            bbox = target['bbox']
            bboxes.append(bbox)

            coco_mask = target['segmentation']
            binary_mask = self.coco.annToMask(target)
            masks.append(binary_mask)

            label = target['category_id']
            labels.append(label)
        
        bboxes = np.array(bboxes).tolist() # np.array(bboxes)
        return bboxes, masks, labels
    
    def __getitem__(self, index: int):
        
        id = self.ids[index]
        image = self._load_image(id)
        targets = self._load_target(id)
        targets = self._filter_targets(targets)
        bboxes, masks, labels = self.unify_targets(targets)
        # masks  = np.array(masks)
        
        if (self.mode in ('train', 'val')):
           
            if self.transform is not None:
                
                transformed = self.transform(
                                    image=image,
                                    masks=masks,
                                    bboxes=bboxes,
                                    class_labels = labels
                                    # 'paste_image', 'paste_masks', 'paste_bboxes'
                                    )
                
                image = transformed["image"]
                masks  = torch.stack(transformed["masks"])
                bboxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)[:,:4]
                image_id = torch.tensor([id])

                if bboxes.shape[0] == 0:
                    return self.__getitem__(idx+1)
                # area = torch.tensor([t['area'] for t in targets],dtype=torch.float32)
                # area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
                area = bboxes[:,2]*bboxes[:,3]
                iscrowd = torch.zeros((len(bboxes),), dtype=torch.int64)
                labels = torch.tensor(transformed["class_labels"],dtype=torch.int64)
                
                bboxes = torchvision.ops.box_convert(bboxes,in_fmt='xywh',out_fmt='xyxy') # to voc style

                y = {
                    'boxes' : bboxes,
                    'labels':labels, 
                    'masks' : masks,
                    'image_id': image_id, #
                    'area': area,
                    'iscrowd':iscrowd
                    }
                
            else:
                y = {'labels':labels, 'bboxes':bboxes, 'masks' : masks}
            
            return image, y # , image_infos
        
        if self.mode == 'test':
            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed["image"]
            return image # , image_infos
    
    def __len__(self) -> int:
        return len(self.ids) # 전체 dataset의 size 반환 


import os
from PIL import Image
import torch
import numpy as np
import random
import albumentations as A
from torchvision.datasets import CocoDetection
import torchvision
def only_seg_ann(ann):
    fil_targets = []
    for target in ann:
        if len(target['segmentation'])>0:
            target['category_id'] = 1 # 
            fil_targets.append(target)
    return fil_targets

def has_valid_annotation(ann):

    fil_targets = only_seg_ann(ann)
    if len(fil_targets) > 0:
        return True 
    else: 
        return False

# TD
# - MODE
class BCAD_CP_dt(CocoDetection):
    def __init__(
        self,
        root,
        annFile,
        transforms,
        mode = 'train'
    ):
        super().__init__(
            root, annFile, None, None, transforms
        )
        self.mode = mode
        # filter images without detection annotations
        ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                ids.append(img_id)
        
        self.ids = ids 
        self._split_transforms()

    def _split_transforms(self):
        split_index = None
        for ix, tf in enumerate(list(self.transforms.transforms)):
            if tf.get_class_fullname() == 'copypaste.CopyPaste':
                split_index = ix

        if split_index is not None:
            tfs = list(self.transforms.transforms)
            pre_copy = tfs[:split_index]
            copy_paste = tfs[split_index]
            post_copy = tfs[split_index+1:]

            #replicate the other augmentation parameters
            bbox_params = None
            keypoint_params = None
            paste_additional_targets = {}
            if 'bboxes' in self.transforms.processors:
                bbox_params = self.transforms.processors['bboxes'].params
                paste_additional_targets['paste_bboxes'] = 'bboxes'
                if self.transforms.processors['bboxes'].params.label_fields:
                    msg = "Copy-paste does not support bbox label_fields! "
                    msg += "Expected bbox format is (a, b, c, d, label_field)"
                    raise Exception(msg)
            if 'keypoints' in self.transforms.processors:
                keypoint_params = self.transforms.processors['keypoints'].params
                paste_additional_targets['paste_keypoints'] = 'keypoints'
                if keypoint_params.label_fields:
                    raise Exception('Copy-paste does not support keypoint label fields!')

            if self.transforms.additional_targets:
                raise Exception('Copy-paste does not support additional_targets!')

            #recreate transforms
            self.transforms = A.Compose(pre_copy, bbox_params, keypoint_params, additional_targets=None)
            self.post_transforms = A.Compose(post_copy, bbox_params, keypoint_params, additional_targets=None)
            self.copy_paste = A.Compose(
                [copy_paste], bbox_params, keypoint_params, additional_targets=paste_additional_targets
            )
        else:
            self.copy_paste = None
            self.post_transforms = None
            
    def load_example(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)
        
        
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #convert all of the target segmentations to masks
        #bboxes are expected to be (y1, x1, y2, x2, category_id)
        masks = []
        bboxes = []
        target = only_seg_ann(target)
        for ix, obj in enumerate(target):
            masks.append(self.coco.annToMask(obj))
            bboxes.append(obj['bbox'] + [obj['category_id']] + [ix])
        
        #pack outputs into a dict
        output = {
            'image': image,
            'masks': masks,
            'bboxes': bboxes
        }
        
        return self.transforms(**output)


    def __getitem__(self, idx):  
        #split transforms if it hasn't been done already
        if not hasattr(self, 'post_transforms'):
            self._split_transforms()

        img_data = self.load_example(idx) # already did transforms before cp
        
        if (self.mode in ('train', 'val')):
            if self.copy_paste is not None:
                paste_idx = random.randint(0, self.__len__() - 1)
                paste_img_data = self.load_example(paste_idx)
                for k in list(paste_img_data.keys()):
                    paste_img_data['paste_' + k] = paste_img_data[k]
                    del paste_img_data[k]

                img_data = self.copy_paste(**img_data, **paste_img_data)
                img_data = self.post_transforms(**img_data)
                img_data['paste_index'] = paste_idx 

            # for legacy code

            image = img_data["image"]
            masks  = torch.stack(img_data["masks"])
            bboxes = torch.tensor(img_data["bboxes"], dtype=torch.float32)[:,:4]
            bboxes = torchvision.ops.box_convert(bboxes,in_fmt='xywh',out_fmt='xyxy')
            # labels = torch.tensor(transformed["class_labels"], dtype=torch.int64)


            image_id = torch.tensor([idx])

            # if bboxes.shape[0] == 0:
            #     return self.__getitem__(idx+1)
            area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
            iscrowd = torch.zeros((len(bboxes),), dtype=torch.int64)

            labels = torch.ones((bboxes.shape[0],), dtype=torch.int64)

            y = {
                'boxes' : bboxes,
                'labels':labels, 
                'masks' : masks,
                'image_id': image_id, #
                'area': area,
                'iscrowd':iscrowd
                }

            return image, y
        
        elif self.mode == 'test':
            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed["image"]
            return image # , image_infos
    