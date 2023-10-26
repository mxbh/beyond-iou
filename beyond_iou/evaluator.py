import os
import numpy as np
from typing import List, Optional, Dict, Union, Iterable
import cv2
from tqdm import tqdm
from PIL import Image
from .result import Result
from .utils import one_hot, get_contiguous_segments, get_interior_boundary, get_exterior_boundary, dilate_mask


ERROR_CODES = {
    'ignore':     -1,
    'unassigned':  0,
    'TP':          1,
    'TN':          2,
    'FP_boundary': 3,
    'FN_boundary': 4,
    'FP_extent':   5,
    'FN_extent':   6,
    'FP_segment':  7,
    'FN_segment':  8,
}


ERROR_PALETTE = {
    -1: (100, 100, 100),
    0: (150, 150, 150),
    1: (255, 255, 255),
    2: (0, 0, 0),
    3: (255, 200, 150),
    4: (150, 200, 255),
    5: (255, 100, 150),
    6: (150, 100, 255),
    7: (255, 0, 0),
    8: (0, 0, 255),
}


class Evaluator:
    def __init__(
        self,
        class_names: List[str], 
        ignore_index: Optional[int] = 255,
        boundary_width: Union[float, int] = 0.01,
        boundary_and_trimap_iou_d: float = 0.02,
        boundary_implementation: str = 'exact'
    ):
        '''The main class for running our error analysis.
        :param class_names: List of strings providing names for class ids 0,...,C.
        :param ignore_index: Class id to be ignored in the IoU computation.
        :param boundary_width: The parameter d in the paper, either as a float in [0,1] (relative to diagonal)
            or as an integer > 1 (absolute number of pixels).
        :param boundary_implementation: Choose "exact" for the euclidean pixel distance. 
            The Boundary IoU paper uses the L1 distance ("fast").
        '''
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.ignore_index = ignore_index

        self.boundary_width = boundary_width
        if 0 < self.boundary_width < 1:
            self.use_relative_boundary_width = True
            print(f'Using relative boundary width of {self.boundary_width} * image diagonal.')
        elif self.boundary_width % 1 != 0 or self.boundary_width < 0:
            raise ValueError('boundary_width should be an integer or a float in (0,1)!')
        else:
            self.use_relative_boundary_width = False
            print(f'Using absolute boundary width of {self.boundary_width} pixels.')

        self.boundary_implementation = boundary_implementation
        self.boundary_and_trimap_iou_d = boundary_and_trimap_iou_d


    def evaluate(
        self,
        loader: Iterable
    ):
        '''This runs the analysis for a whole dataset.
        :param loader: Iterable providing pairs of (pred, gt).
        :returns: beyond_iou.Result.
        '''
        self.results = {
            'unassigned':  np.zeros(self.num_classes, dtype=np.int64),
            'ignore':      np.zeros(self.num_classes, dtype=np.int64),
            'TP':          np.zeros(self.num_classes, dtype=np.int64),
            'TN':          np.zeros(self.num_classes, dtype=np.int64),
            'FP_boundary': np.zeros(self.num_classes, dtype=np.int64),
            'FN_boundary': np.zeros(self.num_classes, dtype=np.int64),
            'FP_extent':   np.zeros(self.num_classes, dtype=np.int64),
            'FN_extent':   np.zeros(self.num_classes, dtype=np.int64),
            'FP_segment':  np.zeros(self.num_classes, dtype=np.int64),
            'FN_segment':  np.zeros(self.num_classes, dtype=np.int64),
        }
        self.boundary_iou_intersection_counts = np.zeros(self.num_classes, dtype=np.int64)
        self.boundary_iou_union_counts = np.zeros(self.num_classes, dtype=np.int64)
        self.trimap_iou_intersection_counts = np.zeros(self.num_classes, dtype=np.int64)
        self.trimap_iou_union_counts = np.zeros(self.num_classes, dtype=np.int64)

        for pred, gt in tqdm(loader, total=len(loader), smoothing=0):
            sample_results = self.evaluate_sample(pred, gt)
            self.update_results(sample_results)

        final_result = Result.from_evaluator(self)
        print(final_result)
        return final_result
    

    def evaluate_sample(self, pred, gt):
        '''Runs the analysis for a single sample.
        :param pred: Predicted segmentation as a numpy array of shape (H,W).
        :param gt: Ground-truth segmentation as a numpy array of shape (H,W).
        :returns: Dictionary holding results for this sample.
        '''
        if pred.shape != gt.shape:
            raise RuntimeError(f'Shapes of prediction and annotation do not match! Pred: {pred.shape}, GT: {gt.shape}')
        H, W = pred.shape
        results = np.full(
            shape=(self.num_classes, H, W),
            fill_value=ERROR_CODES['unassigned'],
            dtype=np.int8
        )
        # IGNORE
        ignore_inds_y, ignore_inds_x = np.where(gt == self.ignore_index)
        results[:, ignore_inds_y, ignore_inds_x] = ERROR_CODES['ignore']

        pred_one_hot = one_hot(pred, num_classes=self.num_classes, ignore_index=self.ignore_index)
        gt_one_hot = one_hot(gt, num_classes=self.num_classes, ignore_index=self.ignore_index)
        
        # select only the active classes
        active_mask = np.logical_or(pred_one_hot.any(axis=(1, 2)), gt_one_hot.any(axis=(1, 2)))
        pred_one_hot_active = pred_one_hot[active_mask]
        gt_one_hot_active = gt_one_hot[active_mask]
        pred_active = np.argmax(pred_one_hot_active, axis=0)
        gt_active = np.argmax(gt_one_hot_active, axis=0)
        gt_active[ignore_inds_y, ignore_inds_x] = self.ignore_index
        results_active = results[active_mask]
        results_inactive = results[~active_mask]

        # TRUE POSITIVE
        tp_mask = np.logical_and(pred_one_hot_active, gt_one_hot_active)
        results_active[tp_mask] = ERROR_CODES['TP']

        # TRUE NEGATIVE
        # active classes
        tn_mask = ~np.logical_or(pred_one_hot_active, gt_one_hot_active)
        results_on_mask = results_active[tn_mask]
        results_active[tn_mask] = np.where(
            results_on_mask != ERROR_CODES['unassigned'],
            results_on_mask,
            ERROR_CODES['TN']
        )
        # inactive classes (everything that is not ignore is TN)
        results_inactive[results_inactive == ERROR_CODES['unassigned']]  = ERROR_CODES['TN']

        # FALSE POSITIVE
        fp_mask = np.logical_and(pred_one_hot_active, ~gt_one_hot_active)

        # FALSE NEGATIVE
        fn_mask = np.logical_and(~pred_one_hot_active, gt_one_hot_active)

        # BOUNDARY
        results_active = self.get_boundary_errors(
            results=results_active,
            tp_mask=tp_mask,
            tn_mask=tn_mask,
            fp_mask=fp_mask,
            fn_mask=fn_mask
        )

        # EXTENT / SEGMENT
        results_active = self.get_extent_segment_errors(
            results=results_active,
            pred_one_hot=pred_one_hot_active,
            gt_one_hot=gt_one_hot_active
        )
        
        results[active_mask] = results_active
        results[~active_mask] = results_inactive
        assert not (results == ERROR_CODES['unassigned']).any()

        # Boundary and Trimap IoU
        (boundary_intersection_counts_active,
        boundary_union_counts_active,
        trimap_intersection_counts_active,
        trimap_union_counts_active) = self.evaluate_sample_boundary_and_trimap_iou(
            sample_results=results_active,
            pred_one_hot=pred_one_hot_active,
            gt_one_hot=gt_one_hot_active,
            ignore_inds=(ignore_inds_y, ignore_inds_x)
        )
        boundary_intersection_counts = np.zeros(self.num_classes, dtype=np.int64)
        boundary_union_counts = np.zeros(self.num_classes, dtype=np.int64)
        trimap_intersection_counts = np.zeros(self.num_classes, dtype=np.int64)
        trimap_union_counts = np.zeros(self.num_classes, dtype=np.int64)

        boundary_intersection_counts[active_mask] += boundary_intersection_counts_active    
        boundary_union_counts[active_mask] += boundary_union_counts_active
        trimap_intersection_counts[active_mask] += trimap_intersection_counts_active
        trimap_union_counts[active_mask] += trimap_union_counts_active

        return dict(
            main_results=results,
            boundary_iou_results=(boundary_intersection_counts, boundary_union_counts),
            trimap_iou_results=(trimap_intersection_counts, trimap_union_counts),
        )


    def update_results(self, sample_results):
        # main results
        for error_name, error_code in ERROR_CODES.items():
            self.results[error_name] += (sample_results['main_results'] == error_code).sum(axis=(1, 2))

        # boundary IoU
        boundary_intersection_counts, boundary_union_counts = sample_results['boundary_iou_results']
        self.boundary_iou_intersection_counts += boundary_intersection_counts
        self.boundary_iou_union_counts += boundary_union_counts

        # trimap Iou
        trimap_intersection_counts, trimap_union_counts = sample_results['trimap_iou_results']
        self.trimap_iou_intersection_counts += trimap_intersection_counts
        self.trimap_iou_union_counts += trimap_union_counts


    def get_boundary_errors(self, results, tp_mask, tn_mask, fp_mask, fn_mask):
        H, W = tp_mask.shape[-2:]
        if self.use_relative_boundary_width:
            img_diag = np.sqrt(H ** 2 + W ** 2)
            boundary_width = int(round(self.boundary_width * img_diag))
        else:
            boundary_width = self.boundary_width

        tp_ext_boundary = get_exterior_boundary(tp_mask, width=boundary_width, implementation=self.boundary_implementation)
        tn_ext_boundary = get_exterior_boundary(tn_mask, width=boundary_width, implementation=self.boundary_implementation)

        boundary_intersection = np.logical_and(tp_ext_boundary, tn_ext_boundary)
        fp_boundary_mask_naive = np.logical_and(fp_mask, boundary_intersection)
        fn_boundary_mask_naive = np.logical_and(fn_mask, boundary_intersection)

        dilated_fp_boundary_mask = dilate_mask(
            mask=fp_boundary_mask_naive,
            width=boundary_width,
            implementation=self.boundary_implementation
        )
        dilated_fn_boundary_mask = dilate_mask(
            mask=fn_boundary_mask_naive,
            width=boundary_width,
            implementation=self.boundary_implementation
        )

        fp_boundary_mask = np.logical_and(dilated_fp_boundary_mask, fp_mask)
        fn_boundary_mask = np.logical_and(dilated_fn_boundary_mask, fn_mask)

        # check if every segment of boundary errors has a TP and a TN as direct neighbor
        fp_boundary_segments = get_contiguous_segments(fp_boundary_mask)
        fn_boundary_segments = get_contiguous_segments(fn_boundary_mask)

        tp_contour = get_exterior_boundary(tp_mask, width=1, implementation='fast')
        tn_contour = get_exterior_boundary(tn_mask, width=1, implementation='fast')

        for c, boundary_segments in fp_boundary_segments.items():
            if c == self.ignore_index:
                continue
            for segment in boundary_segments:
                if (not tp_contour[c][segment].any()) or (not tn_contour[c][segment].any()):
                    fp_boundary_mask[c][segment] = False

        for c, boundary_segments in fn_boundary_segments.items():
            if c == self.ignore_index:
                continue
            for segment in boundary_segments:
                if (not tp_contour[c][segment].any()) or (not tn_contour[c][segment].any()):
                    fn_boundary_mask[c][segment] = False

        results_on_mask = results[fp_boundary_mask]
        results[fp_boundary_mask] = np.where(
            results_on_mask != ERROR_CODES['unassigned'],
            results_on_mask,
            ERROR_CODES['FP_boundary']
        )
        results_on_mask = results[fn_boundary_mask]
        results[fn_boundary_mask] = np.where(
            results_on_mask != ERROR_CODES['unassigned'],
            results_on_mask,
            ERROR_CODES['FN_boundary']
        )
        return results
    

    def get_extent_segment_errors(
        self,
        results,
        pred_one_hot,
        gt_one_hot,
    ):
        pred_segments = get_contiguous_segments(pred_one_hot)
        gt_segments = get_contiguous_segments(gt_one_hot)

        for c, (pred_c, gt_c) in enumerate(zip(pred_one_hot, gt_one_hot)):
            if pred_c.any():
                if gt_c.any():
                    # positve
                    for pred_segment in pred_segments[c]:
                        results_on_segment = results[c][pred_segment]
                        if (results_on_segment == ERROR_CODES['unassigned']).any():
                            error_type = 'FP_extent' if (results_on_segment == ERROR_CODES['TP']).any() \
                                    else 'FP_segment'
                            results[c][pred_segment] = np.where(
                                results_on_segment != ERROR_CODES['unassigned'],
                                results_on_segment,
                                ERROR_CODES[error_type]    
                            )
                
                    # negative
                    for gt_segment in gt_segments[c]:
                        results_on_segment = results[c][gt_segment]
                        if (results_on_segment == ERROR_CODES['unassigned']).any():
                            error_type = 'FN_extent' if (results_on_segment == ERROR_CODES['TP']).any() \
                                    else 'FN_segment'
                            results[c][gt_segment] = np.where(
                                results_on_segment != ERROR_CODES['unassigned'],
                                results_on_segment,
                                ERROR_CODES[error_type]    
                            )
                else:  # only FP segment errors for this class
                    # positive prediction must be a superset of unassigned
                    # every prediction can only be unassigned or ignore
                    assert pred_c[results[c] == ERROR_CODES['unassigned']].all()
                    results[c][results[c] == ERROR_CODES['unassigned']] = ERROR_CODES['FP_segment']
            else:
                if gt_c.any():  # only FN segment errors for this class
                    results[c][results[c] == ERROR_CODES['unassigned']] = ERROR_CODES['FN_segment']
                else:
                    continue

        return results

    
    def evaluate_sample_boundary_and_trimap_iou(self, sample_results, pred_one_hot, gt_one_hot, ignore_inds=None):     
        H, W = sample_results.shape[-2:]
        img_diag = np.sqrt(H ** 2 + W ** 2)
        boundary_width = max(int(round(self.boundary_and_trimap_iou_d * img_diag)), 1)

        # BoundaryIoU uses "fast" boundary implementation, see https://github.com/bowenc0221/boundary-iou-api/blob/37d25586a677b043ed585f10e5c42d4e80176ea9/boundary_iou/utils/boundary_utils.py#L12
        pred_one_hot_int_boundary = get_interior_boundary(pred_one_hot, width=boundary_width, implementation='fast') # P_d ∩ P
        gt_one_hot_int_boundary = get_interior_boundary(gt_one_hot, width=boundary_width, implementation='fast') # G_d ∩ G
        gt_one_hot_ext_boundary = get_exterior_boundary(gt_one_hot, width=boundary_width, implementation='fast') # G_d - G
        gt_one_hot_complete_boundary = np.logical_or(gt_one_hot_int_boundary, gt_one_hot_ext_boundary)  # G_d
    

        boundary_intersection = np.logical_and(pred_one_hot_int_boundary, gt_one_hot_int_boundary)
        boundary_union = np.logical_or(pred_one_hot_int_boundary, gt_one_hot_int_boundary)

        trimap_intersection = np.logical_and(gt_one_hot_int_boundary, pred_one_hot)
        trimap_union = np.logical_or(
            gt_one_hot_int_boundary,
            np.logical_and(gt_one_hot_complete_boundary, pred_one_hot)
        )

        if ignore_inds: # remove ignore pixels
            ignore_inds_y, ignore_inds_x = ignore_inds
            assert not gt_one_hot[:, ignore_inds_y, ignore_inds_x].any()
            boundary_intersection[:, ignore_inds_y, ignore_inds_x] = 0
            boundary_union[:, ignore_inds_y, ignore_inds_x] = 0
            trimap_intersection[:, ignore_inds_y, ignore_inds_x] = 0
            trimap_union[:, ignore_inds_y, ignore_inds_x] = 0


        boundary_intersection_counts = boundary_intersection.sum(axis=(1, 2))
        boundary_union_counts = boundary_union.sum(axis=(1, 2))

        trimap_intersection_counts = trimap_intersection.sum(axis=(1, 2))
        trimap_union_counts = trimap_union.sum(axis=(1, 2))
        
        return (
            boundary_intersection_counts,
            boundary_union_counts,
            trimap_intersection_counts,
            trimap_union_counts
        )


    def visualize_single_sample(self, pred, gt, output_dir):
        sample_results = self.evaluate_sample(pred, gt)['main_results']
        os.makedirs(output_dir, exist_ok=True)
        active_classes = np.unique(np.concatenate([pred, gt]))
        H, W = sample_results.shape[-2:]

        for c in active_classes:
            if c == self.ignore_index:
                continue
            pred_c = (pred == c).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(output_dir, f'{self.class_names[c]}_pred.png'), pred_c)
            gt_c = (gt == c).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(output_dir, f'{self.class_names[c]}_gt.png'), gt_c)
            error_map = np.zeros((H, W, 3), dtype=np.uint8)
            for error_type, error_color in ERROR_PALETTE.items():
                error_map[sample_results[c] == error_type] = error_color
            error_map = cv2.cvtColor(error_map, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(output_dir, f'{self.class_names[c]}_errors.png'), error_map)
        print(f'Saved visualization to {output_dir}.')
        return
