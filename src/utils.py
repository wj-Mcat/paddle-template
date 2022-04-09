from typing import Any
import paddle


def num(tensor_like: Any) -> float:
    """ convert tensor loss to the num
    """
    if paddle.is_tensor(tensor_like):
        tensor_like = paddle.sum(tensor_like)
        return tensor_like.detach().cpu().numpy().item()
    return tensor_like
