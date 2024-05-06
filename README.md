# What’s Outside the Intersection? Fine-grained Error Analysis for Semantic Segmentation Beyond IoU

This is the code used to produce the results in the [paper accepted at WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Bernhard_Whats_Outside_the_Intersection_Fine-Grained_Error_Analysis_for_Semantic_Segmentation_WACV_2024_paper.html).

## Getting started
* install [mmsegmentation>=1.0.0](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md) with all its requirements (e.g. numpy, torch)
* install [tqdm](https://github.com/tqdm/tqdm)
* [optional] install this repository locally with `pip install -e .`

## Usage
To run the error analysis, only the path to ground-truth and predictions (stored as images) as well as some metainfo is necessary, e.g.:
```
python eval.py \
    --pred_dir=path/to/predictions \
    --gt_dir=path/to/gt \
    --dataset_name=ADE20KDataset \
    --output_dir=path/to/result \
    --ignore_index=255 \
    --boundary_width=0.01 \
    --reduce_zero_label # this maps label 0 to 255, 1 to 0 etc.
```
Alternatively, one can create the evaluator (as well as the loader) manually, as done in the main method of eval.py.

## Result
An example for the results obtained with the error analysis can be found in `example_result_mask2former_swin-l_ade20k`.

## Citation
```
@InProceedings{Bernhard_2024_WACV,
    author    = {Bernhard, Maximilian and Amoroso, Roberto and Kindermann, Yannic and Baraldi, Lorenzo and Cucchiara, Rita and Tresp, Volker and Schubert, Matthias},
    title     = {What's Outside the Intersection? Fine-Grained Error Analysis for Semantic Segmentation Beyond IoU},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {968-977}
}
```