from dataclasses import dataclass

from torch import Tensor


@dataclass
class LossOutput:
    loss: Tensor
    prediction_loss_multiplied: Tensor
    opp_mask_prediction_loss_multiplied: Tensor
    mask_loss_multiplied: Tensor
    pred_loss: Tensor
    inverse_tokens_attr_pred_loss: Tensor
    tokens_attr_sparse_loss: Tensor
