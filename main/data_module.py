from typing import List

import pytorch_lightning as pl
import tokenizations
import torch
from datasets import load_dataset, Dataset
from torch import Tensor
from torch.utils.data import DataLoader

from config.config import ExpArgs
from config.constants import (LABELS_NAME, INPUT_IDS_NAME, ATTENTION_MASK_NAME, EXPLAINER_ATTENTION_MASK_NAME,
                              EXPLAINER_INPUT_IDS_NAME, EXPLAINER_PREFIX, MAP_TOKENS, TEXT_PROMPT, LABEL_PROMPT,
                              TASK_PROMPT_KEY, LABEL_PROMPT_KEY)
from config.types_enums import ValidationType, ModelBackboneTypes, ModelPromptType
from models.train_models_utils import get_models_tokenizer


class TextSeqDataModule(pl.LightningDataModule):
    def __init__(self, train_sample: int = -1, test_sample: int = -1, explained_tokenizer = None,
                 explainer_tokenizer = None, data = None, task_prompt_input_ids = None, label_prompt_input_ids = None,
                 val_type: ValidationType = ValidationType.TEST.value):
        super().__init__()
        self.task = ExpArgs.task
        self.seed = ExpArgs.seed
        self.train_dataset, self.val_dataset = None, None
        self.val_type = val_type
        self.train_sample = train_sample
        self.test_sample = test_sample
        self.task_prompt = None
        self.input_prompt = None
        self.pre_label_prompt = None
        self.task_prompt_input_ids = task_prompt_input_ids
        self.label_prompt_input_ids = label_prompt_input_ids
        if explained_tokenizer is not None:
            self.explainer_tokenizer = explainer_tokenizer
            self.explained_tokenizer = explained_tokenizer
        else:
            self.explainer_tokenizer = get_models_tokenizer(ExpArgs.explainer_model_backbone)
            self.explained_tokenizer = get_models_tokenizer(ExpArgs.explained_model_backbone)
            self.set_labels_tokens_opt()
        # SET MAX LENGTH
        if self.task.is_llama_set_max_len and ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
            self.explained_tokenizer.model_max_length = self.task.llama_explained_tokenizer_max_length
            self.explainer_tokenizer.model_max_length = self.task.llama_explainer_tokenizer_max_length

        if data:
            for k, v in data.items():
                data[k] = data[k].unsqueeze(0)
            self.train_dataset = Dataset.from_dict(data)
            self.val_dataset = Dataset.from_dict(data)
            self.train_dataset.set_format(type = 'torch', columns = self.train_dataset.features)
            self.val_dataset.set_format(type = 'torch', columns = self.val_dataset.features)
        else:
            self.dataset = load_dataset(self.task.dataset_name)
            self.dataset_column_text = self.task.dataset_column_text
            self.dataset_column_label = self.task.dataset_column_label
            self.setup()

    def setup(self, stage = None):

        if not self.train_dataset:
            self.setup_train_ds()
            self.setup_test_ds()

    def setup_train_ds(self):
        tmp_train_ds = self.dataset[self.task.dataset_train].shuffle(seed = self.seed)
        if self.train_sample:
            tmp_train_ds = tmp_train_ds.train_test_split(train_size = self.train_sample, seed = self.seed,
                                                         stratify_by_column = self.dataset_column_label)
            tmp_train_ds = tmp_train_ds["train"]
        if ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
            self.set_prompt()
        self.train_dataset = self.handle_ds(tmp_train_ds)

    def setup_test_ds(self):
        if self.val_type == ValidationType.VAL.value:
            tmp_test_ds = self.dataset[self.task.dataset_val].shuffle(seed = self.seed)
            if self.test_sample:
                tmp_test_ds = tmp_test_ds.train_test_split(test_size = self.test_sample, seed = self.seed,
                                                           stratify_by_column = self.dataset_column_label)
                tmp_test_ds = tmp_test_ds["test"]
        else:
            tmp_test_ds = self.dataset[self.task.dataset_test].shuffle(seed = self.seed)
            if self.test_sample:
                tmp_test_ds = tmp_test_ds.train_test_split(train_size = self.test_sample, seed = self.seed,
                                                           stratify_by_column = self.dataset_column_label)
                tmp_test_ds = tmp_test_ds["train"]

        self.val_dataset = self.handle_ds(tmp_test_ds)

    def handle_ds(self, ds):
        ds = ds.map(self.tokenize, batched = False)
        ds = ds.remove_columns(self.dataset_column_text)
        ds = ds.rename_column(self.dataset_column_label, LABELS_NAME)
        # features = ['input_ids', 'attention_mask', LABELS_NAME]
        # if 'token_type_ids' in list(ds.features):
        #     ds = ds.remove_columns("token_type_ids")
        ds.set_format(type = 'torch', columns = ds.features)
        return ds

    def set_labels_tokens_opt(self):
        if ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
            labels_tokens = [self.explained_tokenizer.encode(str(l), return_tensors = "pt", add_special_tokens = False)
                             for l in list(ExpArgs.task.labels_int_str_maps.keys())]
            ExpArgs.labels_tokens_opt = torch.stack(labels_tokens).squeeze()[:, -1]

    def tokenize(self, example):
        inputs_txt = example[self.dataset_column_text]
        if ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
            inputs_txt = self.input_prompt + inputs_txt.strip()
            explained_tokenized_input = self.explained_tokenizer.encode_plus(inputs_txt, truncation = True,
                                                                             add_special_tokens = False)
        else:
            explained_tokenized_input = self.explained_tokenizer.encode_plus(inputs_txt, truncation = True,
                                                                             add_special_tokens = True)
        tokenized_input = explained_tokenized_input

        # explainer

        explainer_tokenized_input = self.explainer_tokenizer.encode_plus(inputs_txt, truncation = True, padding = True,
                                                                         add_special_tokens = True)
        explainer_tokenized_input = {f"{EXPLAINER_PREFIX}{k}": v for k, v in explainer_tokenized_input.items()}
        tokenized_input = {**tokenized_input, **explainer_tokenized_input}
        if ExpArgs.explained_model_backbone in [ModelBackboneTypes.LLAMA.value]:
            decoder_label = self.task.labels_int_str_maps[example[self.dataset_column_label]]
            tokenized_decoder_label = self.explained_tokenizer.encode_plus(decoder_label, truncation = True,
                                                                           padding = True, add_special_tokens = True)
            labels_dict = {f"tokenized_label_{k}": v for k, v in tokenized_decoder_label.items()}
            return {**tokenized_input, **labels_dict}
        return tokenized_input

    def train_dataloader(self):
        # data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        return DataLoader(dataset = self.train_dataset, batch_size = ExpArgs.batch_size, shuffle = True,
                          collate_fn = self.collate_fn)

    def val_dataloader(self):
        # data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)  # TODO change
        return DataLoader(self.val_dataset, batch_size = ExpArgs.eval_batch_size, shuffle = False,
                          collate_fn = self.collate_fn)

    def set_prompt(self):
        task_prompt = self.task.llama_task_prompt
        few_shots_prompt = self.task.llama_few_shots_prompt

        self.task_prompt = "\n\n".join([task_prompt, few_shots_prompt, TEXT_PROMPT])
        self.input_prompt = ""

        if ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
            self.task_prompt_input_ids = self.explained_tokenizer.encode(self.task_prompt, return_tensors = "pt",
                                                                         truncation = True,
                                                                         add_special_tokens = True).squeeze()
            self.label_prompt_input_ids = self.explained_tokenizer.encode("\n" + LABEL_PROMPT, return_tensors = "pt",
                                                                          truncation = True,
                                                                          add_special_tokens = False).squeeze()

            self.task_prompt_input_ids = self.task_prompt_input_ids = self.task_prompt_input_ids[:20]

    def pad_sequences(self, key, batch, tokenizer, model_backbone, is_inputs_ids):
        if tokenizer is None:
            return None
        sequences = [item[key].tolist() for item in batch]
        # llama inputs are not padded - just the task prompt!!
        if model_backbone == ModelBackboneTypes.LLAMA.value:
            return [torch.tensor(i) for i in sequences]
        max_len = max([len(seq) for seq in sequences])
        pad = tokenizer.pad_token_id if is_inputs_ids else 0  # 0 for attention_mask
        if model_backbone == ModelBackboneTypes.LLAMA.value:
            padded_sequences = [[pad] * (max_len - len(seq)) + seq for seq in sequences]
        else:
            padded_sequences = [seq + [pad] * (max_len - len(seq)) for seq in sequences]

        return torch.tensor(padded_sequences)

    def pad_task_prompts_sequences(self, batch, tokenizer):
        if (tokenizer is None) or (ExpArgs.explained_model_backbone != ModelBackboneTypes.LLAMA.value):
            return None
        sequences = [self.task_prompt_input_ids.tolist() + item[INPUT_IDS_NAME].tolist() for item in batch]
        max_len = max([len(seq) for seq in sequences])
        padded_task_prompts = [[tokenizer.pad_token_id] * (max_len - len(seq)) + self.task_prompt_input_ids.tolist() for
                               seq in sequences]
        padded_task_prompts = [torch.tensor(i) for i in padded_task_prompts]
        return padded_task_prompts

    def collate_fn(self, batch):
        explained_input_ids = self.pad_sequences(INPUT_IDS_NAME, batch, self.explained_tokenizer,
                                                 ExpArgs.explained_model_backbone, is_inputs_ids = True)

        explainer_input_ids = self.pad_sequences(EXPLAINER_INPUT_IDS_NAME, batch, self.explainer_tokenizer,
                                                 ExpArgs.explainer_model_backbone, is_inputs_ids = True)
        # LLAMA inputs are not padded
        maps = self.get_tokenizers_map([item[INPUT_IDS_NAME].tolist() for item in batch], explainer_input_ids)

        padded_task_prompts = self.pad_task_prompts_sequences(batch, self.explained_tokenizer)
        return {INPUT_IDS_NAME: explained_input_ids,
                ATTENTION_MASK_NAME: self.pad_sequences(ATTENTION_MASK_NAME, batch, self.explained_tokenizer,
                                                        ExpArgs.explained_model_backbone, is_inputs_ids = False),
                EXPLAINER_INPUT_IDS_NAME: explainer_input_ids,
                EXPLAINER_ATTENTION_MASK_NAME: self.pad_sequences(EXPLAINER_ATTENTION_MASK_NAME, batch,
                                                                  self.explainer_tokenizer,
                                                                  ExpArgs.explainer_model_backbone,
                                                                  is_inputs_ids = False),
                LABELS_NAME: [i[LABELS_NAME] for i in batch], MAP_TOKENS: maps, TASK_PROMPT_KEY: padded_task_prompts,
                LABEL_PROMPT_KEY: self.label_prompt_input_ids}

    def get_tokenizers_map(self, explained_input_ids: List, explainer_input_ids: Tensor):
        maps = []
        if ExpArgs.explained_model_backbone != ModelBackboneTypes.LLAMA.value:
            return maps
        for batch_idx in range(len(explained_input_ids)):
            a2b, b2a = tokenizations.get_alignments(
                [self.explained_tokenizer.convert_ids_to_tokens(i) for i in explained_input_ids[batch_idx]],
                [self.explainer_tokenizer.convert_ids_to_tokens(i.item()) for i in explainer_input_ids[batch_idx]])
            maps.append(a2b)
        return maps
