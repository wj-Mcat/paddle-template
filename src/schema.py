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

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from dataclasses_json import dataclass_json
from paddle.io import Dataset


@dataclass_json
@dataclass
class InputExample:
    """Input Example Data Structure for training data
    this structure can support multi nlp tasks, eg: 
    """
    text_a: str  # source sentence
    label: Union[str, List[str]]  # label field

    guid: Optional[Union[int, str]] = None  # store the union id for example
    text_b: Optional[str] = None  # for sentence pair task
    target_text: Optional[str] = None  # for generation task

    # store the meta data of training example
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def text_or_pairs(self) -> Union[str, Tuple[str, str]]:
        """if text_pair is None, return text. So, it a single text classification task.

        if text_pair is not None, return (text, text_pair). So it's a text
        matching classification task.

        Returns:
            Union[str, Tuple[str, str]]: the result of the training text
        """
        if self.text_b:
            return self.text_a, self.text_b
        return self.text_a


class ExampleDataset(Dataset):
    """Dataset Wrapper for InputExample
    """

    def __init__(self, examples: List[InputExample]):
        super().__init__()
        self.examples: List[InputExample] = examples
        self.label2idx: Dict[str, int] = ExampleDataset.get_label2idx(examples)

    @staticmethod
    def get_label2idx(examples: List[InputExample]) -> Dict[str, int]:
        label2idx: Dict[str, int] = OrderedDict()

        for example in examples:
            if isinstance(example.label, str):
                labels = [example.label]
            else:
                labels = example.label
            for label in labels:
                if label not in label2idx:
                    label2idx[label] = len(label2idx)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]
