import glob
import os.path as osp
import os

from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS


@DATASETS.register_module()
class SRNTIREMultipleGTDataset(BaseSRDataset):
    """REDS dataset for video super resolution for recurrent networks.

    The dataset loads several LQ (Low-Quality) frames and GT (Ground-Truth)
    frames. Then it applies specified transforms and finally returns a dict
    containing paired data and other information.

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        num_input_frames (int): Number of input frames.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        val_partition (str): Validation partition mode. Choices ['official' or
        'REDS4']. Default: 'official'.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(self,
                 lq_folder,
                 gt_folder,
                 pipeline,
                 scale,
                 num_input_frames=None,
                 val_partition='official',
                 test_mode=False):
        super().__init__(pipeline, scale, test_mode)
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.num_input_frames = num_input_frames
        self.val_partition = val_partition
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annoations for REDS dataset.

        Returns:
            dict: Returned dict for LQ and GT pairs.
        """
        # generate keys
        keys = [f'{i:03d}' for i in range(1, 241)]
        val_test_partion = ['001', '011', '015', '020']

        if self.val_partition == 'NTIRE':
            val_partition = val_test_partion
        elif self.val_partition == 'NTIREVal': 
            val_partition = val_test_partion[:]
        elif self.val_partition == 'official':
            val_partition = [f'{i:03d}' for i in range(240, 270)]
        else:
            raise ValueError(
                f'Wrong validation partition {self.val_partition}.'
                f'Supported ones are ["official", "REDS4"]')

        if self.test_mode:
            keys = [v for v in keys if v in val_partition]
            keys = val_test_partion * 2
        else:
            keys = [v for v in keys if v not in val_partition]

        data_infos = []
        for key in keys:
            sequence_length = len(glob.glob(osp.join(self.lq_folder, key, '*.png')))
            # imgs = os.listdir(osp.join(self.lq_folder, key))
            # imgs = [x.endswith('.png') for x in imgs]
            # sequence_length = len(imgs)

            data_infos.append(
                dict(
                    lq_path=self.lq_folder,
                    gt_path=self.gt_folder,
                    key=key,
                    sequence_length=sequence_length,  # REDS has 100 frames for each clip
                    num_input_frames=self.num_input_frames))
        return data_infos
