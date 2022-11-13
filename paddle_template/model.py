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

from paddle import nn


class Net(nn.Layer):
    pass


class SoundClassifier(nn.Layer):
    """simple sound classifier"""

    def __init__(self, backbone: nn.Layer, num_class: int, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.backbone.emb_size, num_class)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.dropout(x)
        logits = self.fc(x)

        return logits
