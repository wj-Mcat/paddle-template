from typing import List

import paddle
from paddle.nn import Layer
from paddle.static import InputSpec
from paddlenlp.transformers import BertForSequenceClassification, PretrainedModel


class InputSpecMixin:
    @staticmethod
    def get_input_specs() -> List[InputSpec]:
        raise NotImplementedError


class CommonInputSpecMixin(InputSpecMixin):
    @staticmethod
    def get_input_specs() -> List[InputSpec]:
        return [
            InputSpec(shape=[None, None], dtype="int64"),
        ]


class PLM(CommonInputSpecMixin):
    def __init__(self, model: PretrainedModel):
        self.model = model

    def to_static(self, save_pretrained: str = None):
        model = paddle.jit.to_static(self.model, input_spec=self.get_input_specs())
        if save_pretrained:
            paddle.jit.save(model, save_pretrained)
        return model
