import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from typing import Optional, List, Tuple, Dict
from .templates import table_template, markdown_template


class Result:
    def __init__(
        self,
        class_names: List[str],
        dataframe: pd.DataFrame,
    ):
        '''Class for presenting the results of the error analysis.
        :param class_names: List of strings providing names for class ids 0,...,C.
        :param dataframe: A pandas dataframe holding the statistics for all classes.
        '''
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.dataframe = dataframe
        # pixel accuracy
        overall_TP = self.dataframe['TP'][:self.num_classes].sum()
        overall_FN = self.dataframe['FN'][:self.num_classes].sum()
        self.pixel_accuracy = overall_TP / (overall_TP + overall_FN)


    @classmethod
    def from_evaluator(cls, evaluator):
        '''Alternative constructor
        :param evaluator: An evaluator object which has run the evaluation already.
        :returns: The corresponding Result object.
        '''
        dataframe = pd.DataFrame(index=evaluator.class_names)
        for error_name, error_counts in evaluator.results.items():
            if error_name == 'unassigned':
                assert (error_counts == 0).all()
                continue
            dataframe[error_name] = error_counts
        # keys = ['TP', 'FP_boundary', 'FP_extent', 'FP_segment', 'FN_boundary', 'FN_extent', 'FN_segment']
        dataframe['FP'] = dataframe['FP_boundary'] + dataframe['FP_extent'] + dataframe['FP_segment']
        dataframe['FN'] = dataframe['FN_boundary'] + dataframe['FN_extent'] + dataframe['FN_segment']  
        dataframe['E_boundary'] = dataframe['FP_boundary'] + dataframe['FN_boundary']
        dataframe['E_extent'] = dataframe['FP_extent'] + dataframe['FN_extent']
        dataframe['E_segment'] = dataframe['FP_segment'] + dataframe['FN_segment']

        union = dataframe['TP'] + dataframe['FP'] + dataframe['FN']
        dataframe['IoU'] = dataframe['TP'] / union
        dataframe['precision'] = dataframe['TP'] / (dataframe['TP'] + dataframe['FP'])
        dataframe['recall'] = dataframe['TP'] / (dataframe['TP'] + dataframe['FN'])
        dataframe['F1_score'] = 2 / (1. / dataframe['precision'] + 1. / dataframe['recall'])

        dataframe['FP_boundary_oU'] = dataframe['FP_boundary'] / union
        dataframe['FN_boundary_oU'] = dataframe['FN_boundary'] / union
        dataframe['E_boundary_oU'] = dataframe['E_boundary'] / union
        
        dataframe['FP_extent_oU']   = dataframe['FP_extent'] / union
        dataframe['FN_extent_oU']   = dataframe['FN_extent'] / union
        dataframe['E_extent_oU'] = dataframe['E_extent'] / union
        
        dataframe['FP_segment_oU']  = dataframe['FP_segment'] / union
        dataframe['FN_segment_oU']  = dataframe['FN_segment'] / union
        dataframe['E_segment_oU'] = dataframe['E_segment'] / union

        dataframe['E_boundary_oU_renormed'] = dataframe['E_boundary'] / \
                                                (dataframe['TP'] + dataframe['E_boundary'])
        dataframe['E_extent_oU_renormed'] = dataframe['E_extent'] / \
                                                (dataframe['TP'] + dataframe['E_boundary'] + dataframe['E_extent'])
        dataframe['E_segment_oU_renormed'] = dataframe['E_segment_oU']

        with np.errstate(invalid='ignore'): # avoid warnings for zero-division
            # boundary IoU
            dataframe['boundary_IoU'] = evaluator.boundary_iou_intersection_counts / evaluator.boundary_iou_union_counts
            # trimap IoU
            dataframe['trimap_IoU'] = evaluator.trimap_iou_intersection_counts / evaluator.trimap_iou_union_counts
            # aggregate classes
            dataframe.loc['mean'] = dataframe.mean(axis=0)

        return cls(
            class_names=evaluator.class_names,
            dataframe=dataframe,
        )


    @classmethod
    def from_file(cls, path, verbose=True):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if verbose:
            print(f'Successfully loaded result from: {path}.')
        return obj


    @classmethod
    def from_csv(cls, path, verbose=True):
        dataframe = pd.read_csv(path, index_col=0)
        class_names = list(dataframe.index[:-1])
        result =  cls(
            class_names=class_names,
            dataframe=dataframe
        )
        if verbose:
            print(f'Successfully loaded result from: {path}.')
        return result


    def save(self, path, verbose=True):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        if verbose:
            print(f'Successfully saved result to: {path}.')
        return


    def save_csv(self, path, verbose=True):
        self.dataframe.to_csv(path)
        if verbose:
            print(f'Successfully saved error stats to: {path}.')
        return
        

    def __str__(self):
        return table_template.format(
            mIoU=self.dataframe.loc['mean']['IoU'] * 100,
            mE_boundary_oU=self.dataframe.loc['mean']['E_boundary_oU'] * 100,
            mFP_boundary_oU=self.dataframe.loc['mean']['FP_boundary_oU'] * 100,
            mFN_boundary_oU=self.dataframe.loc['mean']['FN_boundary_oU'] * 100,
            mE_boundary_oU_renormed=self.dataframe.loc['mean']['E_boundary_oU_renormed'] * 100,
            mE_extent_oU=self.dataframe.loc['mean']['E_extent_oU'] * 100,
            mFP_extent_oU=self.dataframe.loc['mean']['FP_extent_oU'] * 100,
            mFN_extent_oU=self.dataframe.loc['mean']['FN_extent_oU']* 100,
            mE_extent_oU_renormed=self.dataframe.loc['mean']['E_extent_oU_renormed'] * 100,
            mE_segment_oU=self.dataframe.loc['mean']['E_segment_oU'] * 100,
            mFP_segment_oU=self.dataframe.loc['mean']['FP_segment_oU'] * 100,
            mFN_segment_oU=self.dataframe.loc['mean']['FN_segment_oU']* 100,
            mE_segment_oU_renormed=self.dataframe.loc['mean']['E_segment_oU_renormed'] * 100,
            mPrecision=self.dataframe.loc['mean']['precision'] * 100,
            mRecall=self.dataframe.loc['mean']['recall'] * 100,
            mF1_score=self.dataframe.loc['mean']['F1_score'] * 100,
            PixelAcc=self.pixel_accuracy * 100,
            mBoundaryIoU=self.dataframe.loc['mean']['boundary_IoU'] * 100,
            mTrimapIoU=self.dataframe.loc['mean']['trimap_IoU'] * 100 ,
        )


    def create_report(self, directory, exist_ok=False, img_format='png'):
        os.makedirs(directory, exist_ok=exist_ok)
        table_path            = os.path.join(directory, 'table.md')
        pie_chart_path        = os.path.join(directory, f'pie_chart.{img_format}')
        renorm_bar_chart_path = os.path.join(directory, f'renorm_bar_chart.{img_format}')
        class_chart_path      = os.path.join(directory, f'class_chart.{img_format}')
        report_path           = os.path.join(directory, 'report.md')
        pickle_path           = os.path.join(directory, 'result_object.pkl')

        table = self.make_table(path=table_path)
        pie_chart = self.make_pie_chart(path=pie_chart_path)
        renorm_bar_chart = self.make_renormalization_bar_chart(path=renorm_bar_chart_path)
        class_chart = self.make_class_chart(path=class_chart_path)

        report = markdown_template.format(
            table=table,
            pie_chart_path=f'./pie_chart.{img_format}',
            renorm_bar_chart_path=f'./renorm_bar_chart.{img_format}',
            class_chart_path=f'./class_chart.{img_format}',
        )
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.save(pickle_path, verbose=False)
        print(f'Successfully created report in {directory}.')
        return


    def make_table(self, path=None):
        table_string = str(self)
        if path is not None:
            with open(path, 'w') as f:
                f.write(table_string)
        return table_string


    def make_pie_chart(self, path=None, distinguish_fp_fn=False):
        if not distinguish_fp_fn:
            labels = ['mIoU', 'mE_boundary_oU', 'mE_extent_oU', 'mE_segment_oU']
        else:
            labels = ['mIoU', 'mFP_boundary_oU', 'mFP_extent_oU', 'mFP_segment_oU',
                              'mFN_boundary_oU', 'mFN_extent_oU', 'mFN_segment_oU']

        values = self.dataframe.loc['mean'][[l[1:] for l in labels]]
        plt.figure(figsize=(5,5))
        plt.pie(values, labels=labels, autopct='%1.1f%%', normalize=True)
        plt.title('IoU & EoU')
        fig = plt.gcf()
        if path is not None:
            fig.savefig(path, bbox_inches='tight')
        return fig

    
    def make_renormalization_bar_chart(self, path=None):
        keys = ['E_boundary_oU_renormed', 'E_extent_oU_renormed', 'E_segment_oU_renormed']
        labels = ['boundary', 'extent', 'segment']

        values = self.dataframe.loc['mean'][keys] * 100
        plt.figure(figsize=(5, 5))
        plt.bar(x=labels, height=values, color=['C1', 'C2', 'C3'])
        plt.title('Renormalized EoU')
        fig = plt.gcf()
        if path is not None:
            fig.savefig(path, bbox_inches='tight')
        return fig


    def make_class_chart(self, path=None):
        num_bars = len(self.dataframe)   
        keys = ['IoU', 'E_boundary_oU', 'E_extent_oU', 'E_segment_oU']
        ious = self.dataframe['IoU'].values.copy()
        inds = np.argwhere(~np.isfinite(self.dataframe['IoU']).values).flatten()
        ious[inds] = np.arange(-len(inds), 0) # replace NaNs with negative values
        order = np.argsort(-ious[:-1])
        order = np.concatenate([np.array([num_bars-1]), order]) # bring mean to front
        labels = [f'{l} ({s*100:4.1f})' for l,s in zip(self.dataframe.index, self.dataframe['IoU'])]
        labels_ordered = np.array(labels)[order]
        bottom = np.zeros(num_bars)

        plt.figure(figsize=(num_bars**0.7, 4))
        for key in keys:
            height = self.dataframe[key][order] * 100
            height[~np.isfinite(height)] = 0
            plt.bar(labels_ordered, height=height, bottom=bottom, width=1, label=key)
            bottom = bottom + height

        plt.axhline(self.dataframe.loc['mean']['IoU'] * 100, linestyle='-', color='w', linewidth=0.5)
        plt.xlim(-0.5, num_bars - 0.5)
        plt.xticks(rotation=90)
        plt.xlabel('Classes (IoU)')
        plt.ylabel('IoU / EoU')
        plt.legend(framealpha=1)
        fig = plt.gcf()
        if path is not None:
            fig.savefig(path, bbox_inches='tight')
        return fig
