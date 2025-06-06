# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re
import os.path as osp
from .bases import BaseImageDataset

class NightReID(BaseImageDataset):
    """
    NightReID
    Reference:
    NightReID: A Large-Scale Nighttime Person Re-Identification Benchmark AAAI 2025
    https://github.com/msm8976/NightReID

    Dataset statistics:
    # identities: 1500
    # images(default protocol): 15514 (train) + 1394 (query) + 26111 (gallery) 
    """
    dataset_dir = 'NightReID'

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(NightReID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query') # 528IDs
        # self.query_dir = osp.join(self.dataset_dir, 'query-1000') # 1000IDs
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test') # 528IDs w/ Distractor
        # self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test-528-noDistractors') # 528IDs w/o Distractor
        # self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test_withDistractors') # 1000IDs w/ Distractor
        # self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test-1000-noDistractors') # 1000IDs w/o Distractor

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Night600 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'(\d{4})([LR][123])C')
        camdic = {
            'L1': 0, 'L2': 1, 'L3': 2,
            'R1': 3, 'R2': 4, 'R3': 5,
        }

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = pattern.search(img_path).groups()
            pid_container.add(int(pid))
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = pattern.search(img_path).groups()
            pid=int(pid)
            camid = camdic[camid]
            assert 1 <= pid <= 7474
            assert 0 <= camid <= 5
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, pid, camid, 1))
        return dataset
