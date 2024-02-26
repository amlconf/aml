import time
from typing import List

import torch
from torch import Tensor

from config.config import ExpArgs
from config.types_enums import ModelBackboneTypes


def conv_class_to_dict(item):
    obj = {}
    for key in item.keys():
        obj[key] = item[key]
    return obj


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def calculate_num_of_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_num_of_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def calculate_percentage_of_trainable_params(model) -> str:
    return f"{round(calculate_num_of_trainable_params(model) / calculate_num_of_params(model), 2) * 100}%"


def print_number_of_trainable_and_not_trainable_params(model) -> None:
    print(
        f"Number of params: {calculate_num_of_params(model)}, Number of trainable params: {calculate_num_of_trainable_params(model)}")


def get_current_time():
    return int(round(time.time()))


def run_model(model, model_backbone, input_ids: Tensor = None, attention_mask: Tensor = None,
              inputs_embeds: Tensor = None, is_return_logits: bool = False, is_return_output: bool = False):
    if model_backbone == ModelBackboneTypes.LLAMA.value:
        torch.cuda.empty_cache()
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(torch.float16)
            model_output = model(inputs_embeds = inputs_embeds).logits[:, -1, :]
        else:
            model_output = model(input_ids = input_ids).logits[:, -1, :]
        logits = model_output[:, ExpArgs.labels_tokens_opt]  # logits = torch.clip(logits, -10e10, 10e10)
    elif model_backbone in [ModelBackboneTypes.ROBERTA.value, ModelBackboneTypes.BERT.value,
                            ModelBackboneTypes.DISTILBERT.value]:
        model_output = model(input_ids = input_ids, attention_mask = attention_mask, inputs_embeds = inputs_embeds)
        logits = model_output.logits
    else:
        raise ValueError("LLAMA works here only!")

    if is_return_logits and is_return_output:
        return model_output, logits

    elif is_return_logits:
        return logits

    elif is_return_output:
        return model_output

    else:
        raise ValueError(f"must choose model output option")


def model_seq_cls_merge_inputs(inputs: List[Tensor], task_prompt_embeds: List[Tensor], label_prompt_embeds: Tensor):
    if ExpArgs.explained_model_backbone != ModelBackboneTypes.LLAMA.value:
        if type(inputs) == list:
            return torch.stack(inputs)
        return inputs

    merged_inputs = []
    for item_idx in range(len(task_prompt_embeds)):
        merged_inputs.append(
            torch.concat([task_prompt_embeds[item_idx], inputs[item_idx], label_prompt_embeds], axis = 0))
    return torch.stack(merged_inputs)
