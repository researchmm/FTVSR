import cv2 
import numpy as np

import torch

from ..registry import PIPELINES


QP_dict = {
    'crf0': 0.625,
    'crf15': 3.5,
    'crf25': 11.0,
    'crf35': 36.0
}

@PIPELINES.register_module()
class GetQPtensor:
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "gt_labels".

    The "img_meta" item is always populated.  The contents of the "meta"
    dictionary depends on "meta_keys".

    Args:
        keys (Sequence[str]): Required keys to be collected.
        meta_keys (Sequence[str]): Required keys to be collected to "meta".
            Default: None.
    """

    def __init__(self, key):
        self.key = key

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        file_paths = results['lq_path']
        qp_list = []
        for path in file_paths:
            if "crf15" in path:
                qp_list.append(QP_dict['crf15'])
            elif "crf25" in path:
                qp_list.append(QP_dict['crf25'])
            elif "crf35" in path:
                qp_list.append(QP_dict['crf35'])
            else:
                qp_list.append(QP_dict['crf0'])
        qp_list = np.array(qp_list, np.float32)
        qp_list = torch.from_numpy(qp_list)

        results[self.key] = qp_list

        return results

    def __repr__(self):
        return self.__class__.__name__ + (
            f'(key={self.key})')