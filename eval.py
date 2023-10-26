import argparse
from mmseg.registry import DATASETS
from beyond_iou import Evaluator, build_segmentation_loader, Result
from beyond_iou.loader import reduce_zero_label


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True, help='Path to predictions saved as images.')
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to ground truth saved as images.')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of dataset as registered in mmseg.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory.')
    parser.add_argument('--experiment_name', type=str, help='Name of experiment/model (optional).')
    parser.add_argument('--ignore_index', type=int, default=255, help='Ignore index of the ground truth.')
    parser.add_argument('--boundary_width', type=float, default=0.01, help='Boundary width parameter (int for absolute pixel number, float for relative to image diagonal).')
    parser.add_argument('--boundary_implementation', type=str, default='exact', help='exact or fast, fast is used by Boundary IoU code.')
    parser.add_argument('--boundary_and_trimap_iou_d', type=float, default=0.02, help='Boundary width parameter of Boundary and Trimap IoU.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader.')
    parser.add_argument('--reduce_zero_label', action='store_true', help='Sets every label l to l-1 (i.e. 0 to 255). 0 should be the first semantic class (not ignore).')
    parser.add_argument('--pred_suffix', type=str, default='', help='Suffix of prediction files, e.g. "_pred.png"')
    parser.add_argument('--gt_suffix', type=str, default='', help='Suffix of gt files, e.g. "_gt.png"')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    mmseg_dataset = DATASETS.get(args.dataset_name)

    if args.boundary_width % 1 == 0:
        args.boundary_width = int(args.boundary_width)

    evaluator = Evaluator(
        class_names=mmseg_dataset.METAINFO['classes'],
        ignore_index=args.ignore_index,
        boundary_width=args.boundary_width,
        boundary_implementation='exact',
        boundary_and_trimap_iou_d=args.boundary_and_trimap_iou_d,
    )
    loader = build_segmentation_loader(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        gt_label_map=reduce_zero_label if args.reduce_zero_label else None,
        pred_label_map=reduce_zero_label if args.reduce_zero_label else None,
        num_workers=args.num_workers,
        pred_suffix=args.pred_suffix,
        gt_suffix=args.gt_suffix
    )
    result = evaluator.evaluate(loader)
    result.create_report(args.output_dir, exist_ok=True)


if __name__ == '__main__':
    main()