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
from kiss_icp.av_mapping.mapping import mapping

class Argoverse2Dataset:
    def __init__(self, data_dir: Path, sequence: int, *_, **__):
        try:
            importlib.import_module("av2")
        except ModuleNotFoundError:
            print("av2-devkit is not installed on your system")
            print('run "pip install av2-devkit"')
            sys.exit(1)

        self._dataset = SensorDataloader(
            data_dir,
            with_annotations=True,
            with_cache=True,
        )
        self.sequence_id = str(sequence).zfill(6)
        self.scans_dir = os.path.join(os.path.realpath(data_dir), "")

    def __len__(self):
        return mapping[self.sequence_id]["end"] - mapping[self.sequence_id]["start"]

    def __getitem__(self, idx):
        new_idx =  mapping[self.sequence_id]["start"] + idx
        return self._dataset[new_idx].sweep.xyz

    def get_intensity(self, idx):
        new_idx =  mapping[self.sequence_id]["start"] + idx
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
        result_dir = Path(output_dir) / 'results' / 'pcd_argo' / self.sequence_id
        if not result_dir.exists():
            result_dir.mkdir(parents=True, exist_ok=True)
        np.save(f"{result_dir}/{idx}.npy", pcd_intensity)