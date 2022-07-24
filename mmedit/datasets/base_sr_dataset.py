import copy
import os.path as osp
from collections import defaultdict
from pathlib import Path

from mmcv import scandir

from .base_dataset import BaseDataset
import numpy as np
import random

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP')
CRF_FOLDER = {
    0 :'X4_start1',
    15:'X4_crf15', 
    25:'X4_crf25',
    35:'X4_crf35'}

VIMO = {
    0 : 'sequences_BD',
    15: 'sequences_BD_crf_15',
    25: 'sequences_BD_crf_25',
    35: 'sequences_BD_crf_35',
}

def folder_change1(folder):
    if 'X4' in folder:
        if np.random.uniform() > 0.5:
            pass
        else:
            crf_folder_choice = random.choice([0,15,25,35])
            folder = folder.replace('X4_start1', CRF_FOLDER[crf_folder_choice])
    elif 'sequences_BD' in folder:
        if np.random.uniform() > 0.5:
            pass
        else:
            crf_folder_choice = random.choice([0,15,25,35])
            folder = folder.replace('sequences_BD', VIMO[crf_folder_choice])
    else:
        raise ValueError('please check dataset: {}'.format(folder))
    return folder

def folder_change2(folder):
    if 'X4' in folder:
        crf_folder_choice = random.choice([0,15,25,35])
        folder = folder.replace('X4_start1', CRF_FOLDER[crf_folder_choice])
    elif 'sequences_BD' in folder:
        crf_folder_choice = random.choice([0,15,25,35])
        folder = folder.replace('sequences_BD', VIMO[crf_folder_choice])
    else:
        raise ValueError('please check dataset: {}'.format(folder))

    return folder

def random_crf_aug(folder):
    if isinstance(folder, list):
        new_folder = []
        for f in folder:
            nf = folder_change1(f)
            new_folder.append(nf)
    else:
        new_folder = folder_change1(folder)
    
    return new_folder

def random_aug2(folder):
    if isinstance(folder, list):
        new_folder = []
        for f in folder:
            nf = folder_change2(f)
            new_folder.append(nf)
    else:
        new_folder = folder_change2(folder)
    
    return new_folder

class BaseSRDataset(BaseDataset):
    """Base class for super resolution datasets.
    """

    def __init__(self, pipeline, scale, crf, test_mode=False):
        super().__init__(pipeline, test_mode)
        self.scale = scale
        self.crf = crf
        self.test_mode = test_mode

    @staticmethod
    def scan_folder(path):
        """Obtain image path list (including sub-folders) from a given folder.

        Args:
            path (str | :obj:`Path`): Folder path.

        Returns:
            list[str]: image list obtained form given folder.
        """

        if isinstance(path, (str, Path)):
            path = str(path)
        else:
            raise TypeError("'path' must be a str or a Path object, "
                            f'but received {type(path)}.')

        images = list(scandir(path, suffix=IMG_EXTENSIONS, recursive=True))
        images = [osp.join(path, v) for v in images]
        assert images, f'{path} has no valid image file.'
        return images

    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """
        results = copy.deepcopy(self.data_infos[idx])
        results['scale'] = self.scale

        if self.test_mode == False and self.crf == -1:
            results['lq_path'] = random_crf_aug(results['lq_path'])
        elif self.test_mode == False and self.crf == -2:
            results['lq_path'] = random_aug2(results['lq_path'])

        # print(results)
        
        return self.pipeline(results)

    def evaluate(self, results, logger=None):
        """Evaluate with different metrics.

        Args:
            results (list[tuple]): The output of forward_test() of the model.

        Return:
            dict: Evaluation results dict.
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        results = [res['eval_result'] for res in results]  # a list of dict
        eval_results = defaultdict(list)  # a dict of list

        for res in results:
            for metric, val in res.items():
                eval_results[metric].append(val)
        for metric, val_list in eval_results.items():
            assert len(val_list) == len(self), (
                f'Length of evaluation result of {metric} is {len(val_list)}, '
                f'should be {len(self)}')

        # average the results
        eval_results = {
            metric: sum(values) / len(self)
            for metric, values in eval_results.items()
        }

        return eval_results
