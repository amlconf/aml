import gc
import os
import pickle

import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch import Tensor, nn

from config.config import ExpArgs
from config.types_enums import InverseTokenAttrFunctionTypes

ce_loss = nn.CrossEntropyLoss(reduction="mean")


def l1_loss(tokens_attr) -> Tensor:
    return torch.abs(tokens_attr).mean()


def prediction_loss(output, target):
    # target_class_to_compare = torch.argmax(target, dim=1)
    return ce_loss(output, target)  # maximize the pred to original model


def cal_inverse_tokens_attr_pred_loss(logits: Tensor, target: Tensor):
    if ExpArgs.inverse_token_attr_function == InverseTokenAttrFunctionTypes.NEGATIVE_PROB_LOSS.value:
        probs = F.softmax(logits, dim = -1)
        return -torch.log((1-probs)[range(len(target)), target]).mean()
    else:
        raise ValueError("unsupported inverse_token_attr_function selected")


def encourage_token_attr_to_prior_loss(tokens_attr: Tensor, prior: int = 0):
    if prior == 0:
        target = torch.zeros_like(tokens_attr)
    elif prior == 1:
        target = torch.ones_like(tokens_attr)
    else:
        raise NotImplementedError
    # bce_encourage_prior_loss = bce_with_logits_loss(tokens_attr, target)
    bce_encourage_prior_loss = F.binary_cross_entropy(tokens_attr, target)
    return bce_encourage_prior_loss


def save_config_to_root_dir(conf_path_dir, exp_name):
    os.makedirs(conf_path_dir, exist_ok=True)
    h_params = {key: val for key, val in vars(ExpArgs).items() if not key.startswith("__")}
    conf_df = pd.DataFrame([h_params])
    conf_df.to_pickle(f"{conf_path_dir}/{exp_name}.pkl")


def save_checkpoint(model, tokenizer, path_dir):
    os.makedirs(path_dir, exist_ok=True)
    model.save_pretrained(path_dir)
    tokenizer.save_pretrained(path_dir)


def save_running_time(end, begin, exp_name, file_type):
    running_times_conf = f"{ExpArgs.default_root_dir}/RUNNING_TIMES"
    os.makedirs(running_times_conf, exist_ok=True)
    exec_time = end - begin
    with open(f"{running_times_conf}/{exp_name}_{file_type}.pkl", 'wb') as file:
        pickle.dump({"time": exec_time, "exp_name": exp_name, "eval_metric": ExpArgs.eval_metric,
                     "explained_model_backbone": ExpArgs.explained_model_backbone,
                     "explainer_model_backbone": ExpArgs.explainer_model_backbone,
                     "validation_type": ExpArgs.validation_type,
                     "run_type": ExpArgs.run_type}, file)


def init_exp():
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(ExpArgs.seed)
    seed_everything(ExpArgs.seed)


def set_hp(hp: dict):
    for k, v in hp.items():
        setattr(ExpArgs, k, v)