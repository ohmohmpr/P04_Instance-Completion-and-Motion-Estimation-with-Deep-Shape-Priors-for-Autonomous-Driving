# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import sys
from pathlib import Path
import importlib

import natsort
import numpy as np

from kiss_icp.datasets import supported_file_extensions
from av2.datasets.sensor.sensor_dataloader import SensorDataloader

from typing import Final, List, Tuple, Union

class Argoverse2Dataset:
    def __init__(self, data_dir: Path, sequence: int, *_, **__):
        try:
            importlib.import_module("av2")
        except ModuleNotFoundError:
            print("av2 is not installed on your system")
            print('run "pip install av2"')
            sys.exit(1)

        self.part_id = str(int(sequence / 1000)).zfill(3)
        self.sequence_id = self.part_id + str(int(sequence % 1000)).zfill(3)
        self.sequence_int = int(self.sequence_id[-3:])
        self.scans_dir = Path.absolute(data_dir) / f"train-{self.part_id}"
        # This class should be smaller/concise
        self._dataset = SensorDataloader(
            self.scans_dir,
            with_annotations=True,
            with_cache=True,
        )
        print("load data successfully.")
        self.mapping = self._mapping()
        self.annotations = dict()
        self.load_annotations()

    def _mapping(self):
        output_dir = Path.cwd()
        result_dir = Path(output_dir) / 'KISS-ICP' / 'python' / 'kiss_icp'/ 'av_mapping'
        result_path = result_dir / f"{self.part_id}.npy"
        is_found = result_path.exists()
        if is_found:
            mapping = dict()
            mapping = np.load(result_path, allow_pickle='TRUE').item()
        else:
            print("Generating mapping id")
            sequence = 0
            sweep_number = 0
            length = len(self._dataset)
            mapping = dict()
            while sweep_number != length:
                print("sequence {}, sweep_number {}".format(sequence, sweep_number))
                mapping[sequence] = sweep_number
                num_sweeps_in_log = self._dataset[sweep_number].num_sweeps_in_log
                sweep_number = sweep_number + num_sweeps_in_log
                sequence = sequence + 1
            if not result_dir.exists():
                result_dir.mkdir(parents=True, exist_ok=True)
            np.save(result_path, mapping)
        return mapping

    def __len__(self):
        sweep_number = self.mapping[self.sequence_int]
        # They use 0 as an initial value.
        return  self._dataset[sweep_number].num_sweeps_in_log - 1

    def __getitem__(self, idx):
        new_idx =  self.get_new_idx(idx)
        return self._dataset[new_idx].sweep.xyz

    def get_new_idx(self, idx):
        return self.mapping[self.sequence_int] + idx

    def get_intensity(self, idx):
        new_idx =  self.get_new_idx(idx)
        return self._dataset[new_idx].sweep.intensity

    def get_pcd_intensity(self, idx):
        '''
        For 3D detection
        it should be a save function
        '''
        pcd = self.__getitem__(idx)
        intensity = self.get_intensity(idx) / 255
        pcd_intensity = np.hstack((pcd, intensity[..., np.newaxis]))
        output_dir = Path.cwd()
        result_dir = Path(output_dir) / 'results' / 'pcd_argo' / self.part_id / self.sequence_id
        if not result_dir.exists():
            result_dir.mkdir(parents=True, exist_ok=True)
        np.save(f"{result_dir}/{idx}.npy", pcd_intensity)

    def get_timestamp_ns(self, idx):
        new_idx =  self.get_new_idx(idx)
        return self._dataset[new_idx].sweep.timestamp_ns

    def get_annotation(self, idx):
        new_idx =  self.get_new_idx(idx)
        annotations = self._dataset[new_idx].annotations
        annotation = [cuboid for cuboid in annotations if cuboid.category == "REGULAR_VEHICLE"]
        return annotation

    def get_annotations(self):
        num_sweeps = self.__len__()
        self.annotations = dict()
        print("Processing Ground truth")
        for i in range(num_sweeps):
            self.annotations[i] = self.get_annotation(i)
            print("Processing Annotation: IDX", i)
        print("FINISH")
        self.save_annotations(self.annotations)

    def load_annotations(self):
        output_dir = Path.cwd()
        result_dir = Path(output_dir) / 'results_gt' / 'gt' / 'Argoverse2' / self.part_id
        annotations_path = Path(result_dir) / f"{self.sequence_id}.npy"
        is_found = annotations_path.exists()
        if is_found:
            self.annotations = np.load(annotations_path, allow_pickle='TRUE').item()
        else:
            print("Annotation not found", annotations_path)
            self.get_annotations()

    def save_annotations(self, annotations):
        output_dir = Path.cwd()
        result_dir = Path(output_dir) / 'results_gt' / 'gt' / 'Argoverse2' / self.part_id
        if not result_dir.exists():
            result_dir.mkdir(parents=True, exist_ok=True)
        np.save(f"{result_dir}/{self.sequence_id}.npy", annotations)


