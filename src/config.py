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

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from collections import OrderedDict
import json

from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
from tabulate import tabulate
from tap import Tap
from loguru import logger
# pylint: disable=unused-import
import paddle.tensor as Tensor
import paddle
from paddle.fluid.install_check import run_check


def get_logger():
    return logger


class TrainConfigMixin(Tap):
    """Train Config Mixin"""
    batch_size: int = 32  # Batch size per GPU/CPU for training.
    max_seq_length: int = 128  # The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
    learning_rate: float = 5e-5  # The initial learning rate for Adam.
    weight_decay: float = 0.0  # Weight decay if we apply some.
    warmup_proportion: float = 0.0  # Linear warmup proption over the training process.

    valid_steps: int = 100  # The interval steps to evaluate model performance.

    epochs: int = 3
    seed: int = 1000  # random seed for initialization
    
    use_amp: bool = False  # Enable mixed precision training.

    scale_loss: float = 2 ** 15  # The value of scale_loss for fp16.

    label2idx: Dict[str, int] = None

    do_train: bool = True
    do_dev: bool = True
    do_test: bool = True
    template_file: str = './glue_data/tnews/manual_template.json'

    @property
    def label_num(self) -> int:
        """get label number"""
        return len(self.label2idx)

    @property
    def label_maps(self) -> Dict[str, Union[str, List[str]]]:
        """load label maps from template file"""
        label2words = OrderedDict()
        if not os.path.exists(self.template_file):
            raise FileNotFoundError(f'can"t find template file: {self.template_file}')

        with open(self.template_file, 'r', encoding='utf-8') as file_handler:
            data = json.load(file_handler)
            for label, label_obj in data.items():
                label2words[label] = label_obj['labels']
        if len(label2words) == 0:
            return None
        return label2words

class PredictConfigMixin(Tap):
    model_path: Optional[str] = None    # The path of weight of model.


class ModelConfigMixin(Tap):
    pretrained_model: str = 'ernie-1.0'     # the pretrained moddel name

    # this config is related model configuration 
    # hidden_size: int = 768  # The size of hidden states.  

class Config(TrainConfigMixin, PredictConfigMixin, ModelConfigMixin):
    """Global Configuration
    """
    def __init__(self, file: str = None, **kwargs):
        if file and os.path.exists(file):
            file = [file]
        else:
            file = None
        super().__init__(config_files=file, **kwargs)

    data_dir: str = '{{the path of your data}}'
    device: Optional[str] = None
    output_dir: str = './output'
    task: str = 'tnews'  # Dataset for classfication tasks.
    
    def place(self):
        """get the device place

        Returns:
            _type_: The device place
        """
        if not self.device:
            self.device = 'cpu' if run_check() is None else 'gpu'
        
        if self.device == 'cpu':
            return paddle.CPUPlace()
        return paddle.CUDAPlace(0)


@dataclass
class MetricReport:
    """Metric Report"""
    acc: float = 0
    precision: float = 0
    recall: float = 0
    f1_score: float = 0
    micro_f1_score: float = 0
    macro_f1_score: float = 0

    @staticmethod
    def from_sequence(truth: List, predicted: List):
        """get the metric report from sequence"""
        metric = dict(
            acc=accuracy_score(truth, predicted),
            precision=precision_score(truth, predicted),
            recall=recall_score(truth, predicted),
            f1_score=f1_score(truth, predicted),
            micro_f1_score=f1_score(truth, predicted, average='micro'),
            macro_f1_score=f1_score(truth, predicted, average='macro'),
        )
        return MetricReport(**metric)

    def __str__(self) -> str:
        """get the string format of the metric report
        """
        # pylint: disable=consider-using-f-string
        return 'acc: %.5f \t precision: %.5f \t  recall: %.5f \t  f1_score: %.5f \t  micro_f1_score: %.5f \t  macro_f1_score: %.5f \t ' % (
            self.acc, self.precision, self.recall, self.f1_score, self.micro_f1_score, self.macro_f1_score)

    def tabulate(self) -> str:
        """use tabulate to make a great metric format"""
        headers = ['acc', 'precision', 'reclal', 'f1_score', 'micro_f1_score', 'macro_f1_score']
        return tabulate(
            [[
                self.acc, self.precision, self.recall, self.f1_score, self.micro_f1_score, self.macro_f1_score
            ]],
            headers=headers,
            tablefmt='grid',
            floatfmt='.4f',
        )
