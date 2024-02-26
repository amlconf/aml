from typing import Tuple

import torch
from torch import Tensor
from transformers import BertTokenizer, RobertaTokenizer, DistilBertTokenizer, AutoTokenizer

from config.config import ExpArgs, BackbonesMetaData
from config.constants import HF_CACHE
from config.types_enums import ModelBackboneTypes, AddLabelTokenToAttrGetType
from models.tokens_attr_class.bert_tokens_attr_generation import BertTokensAttrGeneration
from models.tokens_attr_class.distilbert_tokens_attr_generation import DistilBertAttrGeneration
from models.tokens_attr_class.roberta_tokens_attr_generation import RobertaTokensAttrGeneration
from utils.dataclasses import Task


def get_transformers_seq_cls_models():
    task = ExpArgs.task
    if ExpArgs.explained_model_backbone == ModelBackboneTypes.BERT.value:
        from transformers import BertForSequenceClassification
        return BertForSequenceClassification.from_pretrained(task.bert_fine_tuned_model, cache_dir = HF_CACHE)
    elif ExpArgs.explained_model_backbone == ModelBackboneTypes.ROBERTA.value:
        from transformers import RobertaForSequenceClassification
        return RobertaForSequenceClassification.from_pretrained(task.roberta_fine_tuned_model, cache_dir = HF_CACHE)
    elif ExpArgs.explained_model_backbone == ModelBackboneTypes.DISTILBERT.value:
        from transformers import DistilBertForSequenceClassification
        return DistilBertForSequenceClassification.from_pretrained(task.distilbert_fine_tuned_model,
                                                                   cache_dir = HF_CACHE)
    elif ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
        from transformers import LlamaForCausalLM, LlamaForSequenceClassification
        model_class = LlamaForSequenceClassification if task.llama_is_for_seq_class else LlamaForCausalLM
        if ExpArgs.llama_f16:
            return model_class.from_pretrained(task.llama_model, torch_dtype = torch.float16, device_map = "auto",
                                               # load_in_8bit=True,
                                               cache_dir = HF_CACHE)
        else:
            return model_class.from_pretrained(task.llama_model,  # torch_dtype = torch.float16,
                                               device_map = "auto",  # load_in_8bit=True,
                                               cache_dir = HF_CACHE)
    else:
        raise ValueError("unsupported model backbone selected")


def get_tokens_attr_generation_model_path(task: Task):
    if ExpArgs.fine_tuned_model_for_tokens_attr_generation is not None:
        return ExpArgs.fine_tuned_model_for_tokens_attr_generation
    elif ExpArgs.explainer_model_backbone == ModelBackboneTypes.BERT.value:
        return task.bert_fine_tuned_model
    elif ExpArgs.explainer_model_backbone == ModelBackboneTypes.ROBERTA.value:
        return task.roberta_fine_tuned_model
    elif ExpArgs.explainer_model_backbone == ModelBackboneTypes.DISTILBERT.value:
        return task.distilbert_fine_tuned_model
    else:
        raise ValueError("unsupported model backbone selected - get_tokens_attr_generation_model_model_path")


def get_tokens_attr_generation_model():
    task = ExpArgs.task
    model_path: str = get_tokens_attr_generation_model_path(task)
    explainer_model_backbone = ExpArgs.explainer_model_backbone
    if explainer_model_backbone is None:
        explainer_model_backbone = ExpArgs.explained_model_backbone
    if explainer_model_backbone == ModelBackboneTypes.BERT.value:
        return BertTokensAttrGeneration.from_pretrained(model_path, cache_dir = HF_CACHE)
    elif explainer_model_backbone == ModelBackboneTypes.ROBERTA.value:
        return RobertaTokensAttrGeneration.from_pretrained(model_path, cache_dir = HF_CACHE)
    elif explainer_model_backbone == ModelBackboneTypes.DISTILBERT.value:
        return DistilBertAttrGeneration.from_pretrained(model_path, cache_dir = HF_CACHE)
    else:
        raise ValueError("unsupported model backbone selected")


def get_models_tokenizer(model_backbone):
    task = ExpArgs.task
    if model_backbone == ModelBackboneTypes.BERT.value:
        return BertTokenizer.from_pretrained(task.bert_fine_tuned_model, cache_dir = HF_CACHE)
    elif model_backbone == ModelBackboneTypes.ROBERTA.value:
        return RobertaTokenizer.from_pretrained(task.roberta_fine_tuned_model, cache_dir = HF_CACHE)
    elif model_backbone == ModelBackboneTypes.DISTILBERT.value:
        return DistilBertTokenizer.from_pretrained(task.distilbert_fine_tuned_model, cache_dir = HF_CACHE)
    elif model_backbone == ModelBackboneTypes.LLAMA.value:
        new_tokenizer = AutoTokenizer.from_pretrained(task.llama_model, cache_dir = HF_CACHE, padding_side = 'left')
        # new_tokenizer.pad_token_id = new_tokenizer.eos_token_id
        new_tokenizer.pad_token_id = new_tokenizer.eos_token_id
        return new_tokenizer
    else:
        raise ValueError("unsupported model type selected")


def get_warmup_steps_and_total_training_steps(n_epochs: int, train_samples_length: int, batch_size: int,
                                              warmup_ratio: int) -> Tuple[int, int]:
    steps_per_epoch = (train_samples_length // batch_size) + 1
    total_training_steps = int(steps_per_epoch * n_epochs)
    warmup_steps = int(total_training_steps * warmup_ratio)
    return warmup_steps, total_training_steps


def construct_word_embedding(model, model_backbone: ModelBackboneTypes, input_ids: Tensor):
    backbone_name = BackbonesMetaData.name[model_backbone]
    model = getattr(model, backbone_name)
    if model_backbone in [ModelBackboneTypes.LLAMA.value]:
        return model.get_input_embeddings()(input_ids)
    else:
        return model.embeddings.word_embeddings(input_ids)


def is_use_add_label():
    return ExpArgs.add_label_token_to_attr_get_type != AddLabelTokenToAttrGetType.NONE.value
