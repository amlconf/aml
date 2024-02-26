import torch
from torch import nn

from config.config import ExpArgs
from config.types_enums import AddLabelTokenToAttrGetType, ActivationFunctionTypes


def add_label_handler(last_hidden_states, labels_new_indices):
    add_label_type = ExpArgs.add_label_token_to_attr_get_type

    if add_label_type == AddLabelTokenToAttrGetType.NONE.value:
        return last_hidden_states

    if add_label_type == AddLabelTokenToAttrGetType.FIRST_TOKEN.value:
        return last_hidden_states[:, 1:]

    elif add_label_type == AddLabelTokenToAttrGetType.LAST_TOKEN.value:
        return last_hidden_states[:, -1:]

    elif add_label_type == AddLabelTokenToAttrGetType.AFTER_LAST_SEP.value:
        return torch.stack(
            [torch.cat((row[:idx], row[idx + 1:])) for row, idx in zip(last_hidden_states, labels_new_indices)])

    else:
        raise ValueError(f"post_model_pre_classifier ERROR!")


def post_model_pre_classifier(last_hidden_states, labels_new_indices):
    last_hidden_states = add_label_handler(last_hidden_states, labels_new_indices)
    return last_hidden_states


def get_act_layer():
    act_type = ExpArgs.attr_gen_token_classifier_mid_activation_function
    if act_type == ActivationFunctionTypes.RELU.value:
        return nn.ReLU()
    elif act_type == ActivationFunctionTypes.TANH.value:
        return nn.Tanh()
    else:
        raise ValueError("Activation function not supported")


def get_token_classifier(hidden_size):
    if ExpArgs.attr_gen_token_classifier_size == 2:
        return nn.Sequential(

            nn.Linear(in_features = hidden_size, out_features = hidden_size),

            get_act_layer(),

            nn.Linear(hidden_size, 1)

        )

    elif ExpArgs.attr_gen_token_classifier_size == 1:
        return nn.Sequential(nn.Linear(hidden_size, 1))

    else:
        raise ValueError("attr_gen_token_classifier_size number not supported")
