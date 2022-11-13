"""
Authors:    Jingjing WU (吴京京) <https://github.com/wj-Mcat>


2022-now @ Copyright wj-Mcat

Licensed under the Apache License, Version 2.0 (the 'License');
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an 'AS IS' BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import json
import os
from abc import ABC
from typing import List, Union

import numpy as np
from numpy import ndarray

from paddle_template.schema import InputExample


class DataProcessor(ABC):
    """Abstract Data Processor Class which handle the different corpus"""

    def get_train_examples(self) -> List[InputExample]:
        """get_train_examples"""
        raise NotImplementedError

    def get_test_examples(self) -> List[InputExample]:
        raise NotImplementedError

    def get_dev_examples(self) -> List[InputExample]:
        raise NotImplementedError

    def get_labels(self) -> List[str]:
        pass


class TNewsDataProcessor(DataProcessor):
    """Tnews Data Processor
    refer to: https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets/tnews
    """

    def __init__(self, data_dir: str, index: Union[str, int] = 0) -> None:
        super().__init__()
        self.data_dir: str = data_dir
        self.data_index = index
        self.train_labels = []

    def _read(self, mode: str) -> List[InputExample]:
        file_path = os.path.join(self.data_dir, f"{mode}{self.data_index}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"target file not found: <{file_path}> ...")

        examples = []
        with open(file_path, "r", encoding="utf-8") as file_handler:
            for line in file_handler:
                data = json.loads(line)
                examples.append(
                    InputExample(
                        text_a=data["sentence"],
                        guid=data["id"],
                        label=data["label_desc"],
                        meta={"keywords": data["keywords"]},
                    )
                )
        return examples

    def get_train_examples(self):
        return self._read("train")

    def get_dev_dataset(self):
        return self._read("dev")

    def get_test_dataset(self):
        return self._read("test")

    def get_labels(self) -> List[str]:
        return self.train_labels


class AudioDataProcessor(DataProcessor):
    def __init__(self, dataset_name: str = None, sr: int = 32000) -> None:
        super().__init__()
        self.sr = sr

        if os.path.isdir(dataset_name):
            # read the local dir
            pass
        else:
            pass

    def _read_file(self, audio_file: str) -> ndarray:
        from paddleaudio import load

        data, sr = load(audio_file, sr=32000, mono=True)
        return data

    # def _read_examples(self, )

    def get_train_examples(self) -> List[InputExample]:
        from paddleaudio.datasets import ESC50

        train_ds = ESC50(mode="train", sample_rate=self.sr)
        return train_ds

    def get_dev_examples(self) -> List[InputExample]:
        from paddleaudio.datasets import ESC50

        dev_ds = ESC50(mode="dev", sample_rate=self.sr)
        return dev_ds

    def get_test_examples(self) -> List[InputExample]:
        from paddleaudio.datasets import ESC50

        dev_ds = ESC50(mode="test", sample_rate=self.sr)
        return dev_ds
