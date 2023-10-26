import os
import numpy as np
import torch
import cv2
from mmengine import scandir


class SegmentationLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        pred_dir,
        gt_dir,
        pred_label_map=None,
        gt_label_map=None,
        pred_suffix='',
        gt_suffix=''
    ):
        self.pred_suffix = pred_suffix
        self.gt_suffix = gt_suffix
        self.pred_dir = pred_dir
        self.pred_files = sorted(list(scandir(self.pred_dir, suffix=self.pred_suffix, recursive=True)))
        self.gt_dir = gt_dir
        self.gt_files = sorted(list(scandir(self.gt_dir, suffix=self.gt_suffix, recursive=True)))
        if len(self.pred_files) != len(self.gt_files):
            raise RuntimeError(f'Number of predictions({len(self.pred_files)}) and ground-truth annotations ({len(self.gt_files)}) do not match!')
        self.pred_label_map = pred_label_map
        self.gt_label_map = gt_label_map


    def __len__(self):
        return len(self.pred_files)


    def __getitem__(self, index):
        pred_path = os.path.join(self.pred_dir, self.pred_files[index])
        gt_path = os.path.join(self.gt_dir, self.gt_files[index])

        pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
        if self.pred_label_map:
            pred = self.pred_label_map(pred)

        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if self.gt_label_map:
            gt = self.gt_label_map(gt)

        return pred, gt


def build_segmentation_loader(
    pred_dir,
    gt_dir,
    num_workers=0,
    pred_label_map=None,
    gt_label_map=None,
    pred_suffix='',
    gt_suffix=''
):
    loader = SegmentationLoader(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        pred_label_map=pred_label_map,
        gt_label_map=gt_label_map,
        pred_suffix=pred_suffix,
        gt_suffix=gt_suffix
    )
    return torch.utils.data.DataLoader(
        loader,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=lambda l: (l[0][0], l[0][1])
    )


def reduce_zero_label(segmentation):
    if segmentation.dtype != np.uint8:
        raise Warning(f'Calling reduce_zero_label on array with dtype {segmentation.dtype}, expected np.uint8!')

    return segmentation - 1


def label_mapping_from_dict(map_dict, ignore_index=255):
    map_dict['ignore_index'] = ignore_index
    def label_map(segmentation):
        return np.vectorize(map_dict.get)(segmentation)

    return label_map
