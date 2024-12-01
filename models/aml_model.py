import math
import os
import shutil
import time
from typing import List

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import AdamW
from transformers import (get_linear_schedule_with_warmup, AutoTokenizer, get_cosine_schedule_with_warmup,
                          get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule_with_warmup)

from config.config import ExpArgs, MetricsMetaData, BackbonesMetaData, RegularizationTypes, LossCoefficients
from config.constants import (EXPLAINED_INPUT_IDS_NAME, EXPLAINED_ATTENTION_MASK_NAME, NEW_ADDED_TRAINABLE_PARAMS,
                              INTERPRETER_ATTENTION_MASK_NAME, INTERPRETER_INPUT_IDS_NAME, MAP_TOKENS, NAN_FLOAT,
                              LABEL_PROMPT_INPUT_IDS, TASK_PROMPT_INPUT_IDS, LABEL_PROMPT_ATTENTION_MASK,
                              TASK_PROMPT_ATTENTION_MASK)
from config.types_enums import (TokenTransformationTypes, DirectionTypes, SchedulerTypes, LabelTokenPosition,
                                CrossTokenizersPooling)
from evaluations.evaluations import evaluate_tokens_attributions
from main.explained_model_utils import (save_checkpoint, encourage_token_attr_to_prior_loss, l1_loss,
                                        calculate_prediction_loss, calculate_inverse_loss)
from models.train_models_utils import construct_word_embedding, is_add_label_embedding
from utils.dataclasses import AmlOutput, LossOutput, StepOutput
from utils.dataclasses.evaluations import DataForEvaluation, DataForEvaluationInputs
from utils.utils_functions import (conv_class_to_dict, get_current_time, run_model, merge_prompts,
                                   is_model_encoder_only, is_use_prompt, get_model_special_tokens, get_device)


class AmlModel(pl.LightningModule):
    def __init__(self, explained_model, interpreter_model, interpreter_tokenizer: AutoTokenizer,
                 explained_tokenizer: AutoTokenizer, total_training_steps: int, experiment_path: str,
                 checkpoints_path: str, warmup_steps: int, trainable_embeddings, label_embedding_index, ref_token_id):
        super().__init__()
        self.log_round_digits = 4
        self.task = ExpArgs.task
        self.warmup_steps = warmup_steps
        self.explained_model = explained_model
        self.interpreter_model = interpreter_model
        self.trainable_embeddings = trainable_embeddings
        self.label_embedding_index = label_embedding_index
        self.freeze_layers()
        self.interpreter_tokenizer = interpreter_tokenizer
        self.explained_tokenizer = explained_tokenizer

        self.interpreter_special_tokens = torch.tensor(
            get_model_special_tokens(ExpArgs.interpreter_model_backbone, self.interpreter_tokenizer))
        self.explained_special_tokens = torch.tensor(
            get_model_special_tokens(ExpArgs.explained_model_backbone, self.explained_tokenizer))

        self.ref_token_id = ref_token_id
        self.n_training_steps = total_training_steps
        self.experiment_path = experiment_path
        self.checkpoints_path = checkpoints_path

        self.training_step_outputs: List[StepOutput] = []
        self.val_step_outputs: List[StepOutput] = []
        self.prev_metric_result = None

    # TODO: merge forward and forwad_paml_inference functions
    def forward(self, explained_model_inputs):
        batch = conv_class_to_dict(explained_model_inputs)
        explained_model_inputs, explained_model_attention_mask = merge_prompts(  #
            inputs = batch[EXPLAINED_INPUT_IDS_NAME],  #
            attention_mask = batch[EXPLAINED_ATTENTION_MASK_NAME],  #
            task_prompt = batch[TASK_PROMPT_INPUT_IDS],  #
            label_prompt = batch[LABEL_PROMPT_INPUT_IDS],  #
            task_prompt_attention_mask = batch[TASK_PROMPT_ATTENTION_MASK],  #
            label_prompt_attention_mask = batch[LABEL_PROMPT_ATTENTION_MASK]  #
        )

        explained_model_predicted_logits = run_model(model = self.explained_model,
                                                     model_backbone = ExpArgs.explained_model_backbone,
                                                     input_ids = explained_model_inputs,
                                                     attention_mask = explained_model_attention_mask,
                                                     is_return_logits = True)
        explained_model_predicted_probabilities = torch.softmax(explained_model_predicted_logits, dim = 1)
        explained_model_predicted_class = torch.argmax(explained_model_predicted_logits, dim = 1)

        token_score_attributions = self.calculate_tokens_attribution(explained_model_predicted_probabilities, batch)

        _tokens_attribution_ = token_score_attributions.clone()

        interpreter_tokens_attribution, interpreter_special_tokens_indices = self.special_tokens_handler(
            batch[INTERPRETER_INPUT_IDS_NAME], _tokens_attribution_, self.interpreter_special_tokens,
            _is_return_special_tokens_indices = True, _replacement_value = 0)

        explained_tokens_attribution = self.transform_tokens_attr_handler(interpreter_tokens_attribution.clone(), batch)


        if not is_model_encoder_only(ExpArgs.explained_model_backbone):
            for _item_tokens_attribution in explained_tokens_attribution:
                nan_indices = torch.isnan(_item_tokens_attribution)
                _item_tokens_attribution[nan_indices] = 0 # shouldnt happen


        explained_tokens_attribution, explained_special_tokens_indices = self.special_tokens_handler(
            batch[EXPLAINED_INPUT_IDS_NAME], explained_tokens_attribution, self.explained_special_tokens,
            _is_return_special_tokens_indices = True, _replacement_value = 1)

        explained_model_perturbed_inputs_logits, inverse_explained_model_perturbed_inputs_logits = self.forward_with_token_attributions(
            batch, explained_tokens_attribution, explained_special_tokens_indices = explained_special_tokens_indices)
        loss_output = self.calculate_loss(explained_model_logits = explained_model_perturbed_inputs_logits,
                                          inverse_explained_model_logits = inverse_explained_model_perturbed_inputs_logits,
                                          target_probabilities = explained_model_predicted_probabilities,
                                          explained_tokens_attribution = explained_tokens_attribution,
                                          interpreter_tokens_attribution = interpreter_tokens_attribution,
                                          special_tokens_indices = explained_special_tokens_indices)

        return AmlOutput(loss_output = loss_output, explained_model_predicted_class = explained_model_predicted_class,
                         explained_model_predicted_logits = explained_model_predicted_logits,
                         tokens_attr = explained_tokens_attribution)

    def forwad_paml_inference(self, batch, is_evaluate):
        with torch.no_grad():
            batch = {k: v.to(get_device()) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            begin = time.time()

            if is_use_prompt():
                for k in [TASK_PROMPT_INPUT_IDS, TASK_PROMPT_ATTENTION_MASK, EXPLAINED_INPUT_IDS_NAME,
                    EXPLAINED_ATTENTION_MASK_NAME]:
                    batch[k] = [item.to(get_device()) for item in batch[k]]


            explained_model_inputs, explained_model_attention_mask = merge_prompts(  #
                inputs = batch[EXPLAINED_INPUT_IDS_NAME],  #
                attention_mask = batch[EXPLAINED_ATTENTION_MASK_NAME],  #
                task_prompt = batch[TASK_PROMPT_INPUT_IDS],  #
                label_prompt = batch[LABEL_PROMPT_INPUT_IDS],  #
                task_prompt_attention_mask = batch[TASK_PROMPT_ATTENTION_MASK],  #
                label_prompt_attention_mask = batch[LABEL_PROMPT_ATTENTION_MASK]  #
            )

            explained_model_predicted_logits = run_model(model = self.explained_model,
                                                         model_backbone = ExpArgs.explained_model_backbone,
                                                         input_ids = explained_model_inputs,
                                                         attention_mask = explained_model_attention_mask,
                                                         is_return_logits = True)
            explained_model_predicted_probabilities = torch.softmax(explained_model_predicted_logits, dim = 1)
            explained_model_predicted_class = torch.argmax(explained_model_predicted_logits, dim = 1)

            token_score_attributions = self.calculate_tokens_attribution(explained_model_predicted_probabilities, batch)

            _tokens_attribution_ = token_score_attributions.clone()

            interpreter_tokens_attribution, interpreter_special_tokens_indices = self.special_tokens_handler(
                batch[INTERPRETER_INPUT_IDS_NAME], _tokens_attribution_, self.interpreter_special_tokens,
                _is_return_special_tokens_indices = True, _replacement_value = 0)

            explained_tokens_attribution = self.transform_tokens_attr_handler(interpreter_tokens_attribution.clone(),
                                                                              batch)

            if not is_model_encoder_only(ExpArgs.explained_model_backbone):
                for _item_tokens_attribution in explained_tokens_attribution:
                    nan_indices = torch.isnan(_item_tokens_attribution)
                    _item_tokens_attribution[nan_indices] = 0  # The case no source found

            explained_tokens_attribution, explained_special_tokens_indices = self.special_tokens_handler(
                batch[EXPLAINED_INPUT_IDS_NAME], explained_tokens_attribution, self.explained_special_tokens,
                _is_return_special_tokens_indices = True, _replacement_value = 1)


            if len(explained_tokens_attribution) != 1:
                raise ValueError("inference use with batch=1 only")

            tokens_attribution = explained_tokens_attribution[0]

            evaluation_item = None
            if is_evaluate:
                item_data = DataForEvaluation(tokens_attr = tokens_attribution.detach().squeeze(),
                                              explained_model_predicted_class = explained_model_predicted_class,
                                              explained_model_predicted_logits = explained_model_predicted_logits.squeeze(),
                                              input = DataForEvaluationInputs(  #
                                                  input_ids = batch[EXPLAINED_INPUT_IDS_NAME],  #
                                                  attention_mask = batch[EXPLAINED_ATTENTION_MASK_NAME],
                                                  task_prompt_input_ids = batch[TASK_PROMPT_INPUT_IDS],
                                                  label_prompt_input_ids = batch[LABEL_PROMPT_INPUT_IDS],
                                                  task_prompt_attention_mask = batch[TASK_PROMPT_ATTENTION_MASK],
                                                  label_prompt_attention_mask = batch[LABEL_PROMPT_ATTENTION_MASK]))

                evaluation_result, evaluation_item = evaluate_tokens_attributions(model = self.explained_model,
                                                                                  explained_tokenizer = self.explained_tokenizer,
                                                                                  ref_token_id = self.ref_token_id,
                                                                                  data = item_data, step = -1,
                                                                                  epoch = -1, item_index = "",
                                                                                  experiment_path = self.experiment_path)

                end = time.time()
                duration = end - begin

        return tokens_attribution, evaluation_item, duration

    def calculate_tokens_attribution(self, explained_model_probabilities, batch):
        input_ids = batch[INTERPRETER_INPUT_IDS_NAME].clone()
        attention_mask = batch[INTERPRETER_ATTENTION_MASK_NAME].clone()

        input_ids, attention_mask, inputs_embeds, inserted_label_token_indices, long_vectors = self.insert_label_token_embeddings_to_interpreter(
            explained_model_probabilities, input_ids, attention_mask)

        # try:
        #     print(f"self.interpreter_model(input_ids = input_ids: {input_ids}. {inputs_embeds.shape}")
        # except Exception as e:
        #     print(f"self.interpreter_model(input_ids = input_ids: {input_ids}")

        attribution_scores = self.interpreter_model(input_ids = input_ids, attention_mask = attention_mask,
                                                    inputs_embeds = inputs_embeds,
                                                    inserted_label_token_indices = inserted_label_token_indices)
        attribution_scores = self.re_swap_last_tokens(attribution_scores, long_vectors)
        return attribution_scores

    def _remove_special_tokens(self, _input_ids, _tokens_attr, _special_tokens, _special_tokens_indices = None,
                               _replacement_value = 1):
        if _special_tokens_indices is None:
            _special_tokens_indices = torch.isin(_input_ids, _special_tokens.to(self.device))
        _tokens_attr[_special_tokens_indices] = _replacement_value
        return _tokens_attr, _special_tokens_indices

    def special_tokens_handler(self, _input_ids, _tokens_attr, _special_tokens, _replacement_value = 1,
                               _is_return_special_tokens_indices = False, _special_tokens_indices = None):
        if isinstance(_tokens_attr, list):
            new_tokens_attr, new_special_tokens_indices = [], []
            for i in range(len(_tokens_attr)):
                current_special_tokens_indices = _special_tokens_indices[
                    i] if _special_tokens_indices is not None else None
                current_token_attr, current_special_token = self._remove_special_tokens(_input_ids[i], _tokens_attr[i],
                                                                                        _special_tokens,
                                                                                        _special_tokens_indices = current_special_tokens_indices,
                                                                                        _replacement_value = _replacement_value)
                new_tokens_attr.append(current_token_attr)
                new_special_tokens_indices.append(current_special_token)
        else:
            new_tokens_attr, new_special_tokens_indices = self._remove_special_tokens(_input_ids, _tokens_attr,
                                                                                      _special_tokens,
                                                                                      _special_tokens_indices = None,
                                                                                      _replacement_value = _replacement_value)
        if _is_return_special_tokens_indices:
            return new_tokens_attr, new_special_tokens_indices
        return new_tokens_attr

    def transform_tokens_attr_handler(self, tokens_attr, batch):
        if is_model_encoder_only(ExpArgs.explained_model_backbone):
            return tokens_attr
        new_tokens_attr = []
        for batch_idx in range(len(batch[EXPLAINED_INPUT_IDS_NAME])):
            new_tokens_attr_lst = []
            for indices in batch[MAP_TOKENS][batch_idx]:
                if -1 in indices: # padding
                    new_tokens_attr_lst.append(torch.tensor(0).cuda())
                    continue

                scores = tokens_attr[batch_idx][indices]
                scores = [v for v in scores if not math.isnan(v)]
                pooled_score = torch.tensor(NAN_FLOAT).to(self.device)
                if len(scores) > 0:
                    scores = torch.stack(scores)
                    if ExpArgs.cross_tokenizers_pooling == CrossTokenizersPooling.MEAN.value:
                        pooled_score = scores.mean()
                    elif ExpArgs.cross_tokenizers_pooling == CrossTokenizersPooling.MAX.value:
                        pooled_score = scores.max()
                    elif ExpArgs.cross_tokenizers_pooling == CrossTokenizersPooling.MIN.value:
                        pooled_score = scores.min()
                    else:
                        raise ValueError(f"cross_tokenizers_pooling is not supported")

                new_tokens_attr_lst.append(pooled_score)
            new_tokens_attr.append(torch.stack(new_tokens_attr_lst))
        return new_tokens_attr

    def re_swap_last_tokens(self, attribution_scores, long_vectors):
        if not is_add_label_embedding():
            return attribution_scores

        # after remove label token
        if long_vectors is not None:
            batch_size = attribution_scores.shape[0]
            zeros_vec = torch.zeros(batch_size, device = attribution_scores.device).unsqueeze(-1)
            attribution_scores = torch.cat((attribution_scores, zeros_vec), dim = 1)

            self.swap(attribution_scores, long_vectors, index_1 = -1, index_2 = -2)

        return attribution_scores

    def swap_last_tokens(self, input_ids: Tensor, attention_mask: Tensor):
        last_index = -1  # last token
        second_to_last_index = last_index - 1  # before last
        indices = None
        if input_ids.shape[-1] == self.interpreter_tokenizer.model_max_length:
            sep_indices = input_ids[:, last_index] == self.interpreter_tokenizer.sep_token_id
            indices = torch.nonzero(sep_indices).squeeze()

            self.swap(input_ids, indices, second_to_last_index, last_index)
            self.swap(attention_mask, indices, second_to_last_index, last_index)

            # Remove the token immediately preceding the SEP token
            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, :-1]
        return input_ids, attention_mask, indices

    @staticmethod
    def swap(input_tensor, vectors_idices, index_1, index_2):
        try:
            if not isinstance(vectors_idices, torch.Tensor):
                vectors_idices = torch.tensor(vectors_idices)
            if vectors_idices.dim() == 0:
                vectors_idices = vectors_idices.unsqueeze(0)
            input_tensor[vectors_idices, index_1], input_tensor[vectors_idices, index_2] = input_tensor[
                vectors_idices, index_2], input_tensor[vectors_idices, index_1]
        except Exception as e:
            print(f"vectors_idx: {vectors_idices}")
            print(f"t: {input_tensor}")
            raise ValueError(e)

    def get_trainable_embed_vec(self, i):
        vec = self.trainable_embeddings(torch.tensor(i).to(self.device))
        if ExpArgs.is_include_general_label_token:
            label_vec = self.trainable_embeddings(self.label_embedding_index.to(self.device))
            return vec + label_vec
        return vec

    def insert_label_token_embeddings_to_interpreter(self, explained_model_probabilities, input_ids, attention_mask):
        interpreter_label_token_position = ExpArgs.interpreter_label_token_position
        inserted_label_token_indices = []

        if interpreter_label_token_position == LabelTokenPosition.NONE.value:
            inputs_embeds, inserted_label_token_indices, long_vectors = None, None, None
            return input_ids, attention_mask, inputs_embeds, inserted_label_token_indices, long_vectors

        new_attention_vec = torch.ones(attention_mask.shape[0], device = attention_mask.device).unsqueeze(-1)
        input_ids, attention_mask, long_vectors = self.swap_last_tokens(input_ids, attention_mask)

        eps = 1e-30
        # eps = 0
        prediction_embeddings: List[Tensor] = []
        for current_probability in explained_model_probabilities.tolist():
            item_prob_embed: List[Tensor] = [(p - eps) * self.get_trainable_embed_vec(i) for i, p in
                                             enumerate(current_probability)]
            item_prob_embed_mean = torch.stack(item_prob_embed).mean(dim = 0)
            prediction_embeddings.append(item_prob_embed_mean)
        prediction_embeddings = torch.stack(prediction_embeddings).unsqueeze(1)

        inputs_embeds = construct_word_embedding(self.interpreter_model, ExpArgs.interpreter_model_backbone,
                                                 input_ids).to(self.device)

        if interpreter_label_token_position == LabelTokenPosition.FIRST_TOKEN.value:
            attention_mask = torch.cat((new_attention_vec, attention_mask), dim = -1)
            inputs_embeds = torch.cat((prediction_embeddings, inputs_embeds), dim = 1)


        elif interpreter_label_token_position == LabelTokenPosition.LAST_TOKEN.value:
            attention_mask = torch.cat((attention_mask, new_attention_vec), dim = -1)
            inputs_embeds = torch.cat((inputs_embeds, prediction_embeddings), dim = 1)


        elif interpreter_label_token_position == LabelTokenPosition.AFTER_LAST_SEP.value:
            batch_size = attention_mask.shape[0]
            pad = self.interpreter_tokenizer.pad_token_id
            pad_vec = torch.tensor([pad] * batch_size, device = attention_mask.device).unsqueeze(-1)
            pad_vec_embeds = construct_word_embedding(self.interpreter_model, ExpArgs.interpreter_model_backbone,
                                                      pad_vec).to(self.device)
            inputs_embeds = torch.cat((inputs_embeds, pad_vec_embeds), dim = 1)
            zeros_vec = torch.zeros(batch_size, device = attention_mask.device).unsqueeze(-1)
            attention_mask = torch.cat((attention_mask, zeros_vec), dim = -1)

            sep_token_indices = (input_ids == self.interpreter_tokenizer.sep_token_id).nonzero(as_tuple = False).cpu()

            # support one sep only
            last_sep_indices = sep_token_indices[:, 1]
            inserted_label_token_indices = last_sep_indices + 1

            batch_indices = sep_token_indices[:, 0]

            inputs_embeds[batch_indices, inserted_label_token_indices.unsqueeze(1)] = prediction_embeddings[
                batch_indices].squeeze(1)
            attention_mask[batch_indices, inserted_label_token_indices] = 1

            # support one SEP only  # sep_counts = sep_token_indices[:, 0].bincount()  # if not (sep_counts == 1).all():  #     raise ValueError(  #         f"Issue with interpreter_label_token_position after the last [SEP] token. Items with incorrect [SEP] count: {(sep_counts != 1).nonzero(as_tuple = False).tolist()}")

        # using input embed instead of input ids
        input_ids = None
        return input_ids, attention_mask, inputs_embeds, inserted_label_token_indices, long_vectors

    def forward_with_token_attributions(self, batch, tokens_attributions, explained_special_tokens_indices):
        task_prompt_embeds, label_prompt_embeds = None, None
        if is_model_encoder_only(ExpArgs.explained_model_backbone):
            inputs_embeds = construct_word_embedding(self.explained_model, ExpArgs.explained_model_backbone,
                                                     batch[EXPLAINED_INPUT_IDS_NAME]).to(self.device)
        else:
            inputs_embeds = [
                construct_word_embedding(self.explained_model, ExpArgs.explained_model_backbone, i).to(self.device) for
                i in batch[EXPLAINED_INPUT_IDS_NAME]]
            if is_use_prompt():
                task_prompt_embeds = [
                    construct_word_embedding(self.explained_model, ExpArgs.explained_model_backbone, i).to(self.device)
                    for i in batch[TASK_PROMPT_INPUT_IDS]]

                label_prompt_embeds = construct_word_embedding(self.explained_model, ExpArgs.explained_model_backbone,
                                                               batch[LABEL_PROMPT_INPUT_IDS]).to(self.device)

        explained_model_perturbed_inputs_logits = self.integrate_tokens_attributions_with_explained_model_input_embeds(
            tokens_attributions, inputs_embeds, batch, task_prompt_embeds, label_prompt_embeds)
        inverse_tokens_attributions = [1 - item for item in tokens_attributions]
        inverse_tokens_attributions = self.special_tokens_handler(batch[EXPLAINED_INPUT_IDS_NAME],
                                                                  inverse_tokens_attributions,
                                                                  self.explained_special_tokens,
                                                                  _special_tokens_indices = explained_special_tokens_indices,
                                                                  _replacement_value = 1)

        inverse_explained_model_perturbed_inputs_logits = self.integrate_tokens_attributions_with_explained_model_input_embeds(
            inverse_tokens_attributions, inputs_embeds, batch, task_prompt_embeds, label_prompt_embeds)
        return explained_model_perturbed_inputs_logits, inverse_explained_model_perturbed_inputs_logits

    def integrate_tokens_attributions_with_explained_model_input_embeds(self, tokens_attr, inputs_embeds, batch,
                                                                        task_prompt_embeds, label_prompt_embeds):
        input_ids = batch[EXPLAINED_INPUT_IDS_NAME]
        tokens_attr_function = ExpArgs.tokens_attr_with_ref_token_function_type
        if inputs_embeds[0].dim() != tokens_attr[0].dim():
            tokens_attr = [i.unsqueeze(1) for i in tokens_attr]

        if tokens_attr_function == TokenTransformationTypes.SCALE.value:
            embedding_output = [tokens_attr[i] * inputs_embeds[i] for i in range(len(tokens_attr))]
        elif tokens_attr_function == TokenTransformationTypes.BLEND.value:
            embedding_output = []
            for i in range(len(tokens_attr)):
                mask_inputs = input_ids[i].detach().clone().fill_(self.ref_token_id)
                mask_embedding_output = construct_word_embedding(self.explained_model, ExpArgs.explained_model_backbone,
                                                                 mask_inputs)
                res = ((1 - tokens_attr[i]) * mask_embedding_output) + (tokens_attr[i] * inputs_embeds[i])
                embedding_output.append(res)
        else:
            raise ValueError("forbidden mask function")
        embedding_output, attention_mask = merge_prompts(inputs = embedding_output,
                                                         attention_mask = batch[EXPLAINED_ATTENTION_MASK_NAME],
                                                         task_prompt = task_prompt_embeds,
                                                         label_prompt = label_prompt_embeds,
                                                         task_prompt_attention_mask = batch[TASK_PROMPT_ATTENTION_MASK],
                                                         label_prompt_attention_mask = batch[
                                                             LABEL_PROMPT_ATTENTION_MASK])

        return run_model(model = self.explained_model, model_backbone = ExpArgs.explained_model_backbone,
                         inputs_embeds = embedding_output, attention_mask = attention_mask, is_return_logits = True)

    def training_step(self, batch, batch_idx):
        self.explained_model.eval()
        self.interpreter_model.train()

        output = self.forward(explained_model_inputs = batch)

        results = StepOutput(loss = output.loss_output.loss, prediction_loss = output.loss_output.prediction_loss,
                             prediction_loss_weighted = output.loss_output.prediction_loss_weighted,
                             regularization_loss = output.loss_output.regularization_loss,
                             explained_model_predicted_class = output.explained_model_predicted_class,
                             explained_model_logits = output.explained_model_predicted_logits,
                             regularization_weighted = output.loss_output.regularization_loss_weighted,
                             inverse_loss = output.loss_output.inverse_loss,
                             inverse_loss_weighted = output.loss_output.inverse_loss_weighted,
                             tokens_attr = output.tokens_attr, input = batch)

        log_dict = dict(loss = results.loss.item(), prediction_loss = results.prediction_loss.item(),
                        inverse_loss = results.inverse_loss.item(),
                        regularization_loss = results.regularization_loss.item(),
                        inverse_loss_weighted = results.inverse_loss_weighted.item(),
                        prediction_loss_weighted = results.prediction_loss_weighted.item(),
                        regularization_weighted = results.regularization_weighted.item())
        log_dict = {"Train_step/" + key: round(value, self.log_round_digits) for key, value in log_dict.items()}
        self.log_dict(log_dict, on_step = True, on_epoch = False)
        # self.logger.log_metrics(log_dict)
        self.training_step_outputs.append(results)
        return dict(loss = results.loss)

    def validation_step(self, batch, batch_idx):

        self.explained_model.eval()
        self.interpreter_model.eval()

        with torch.no_grad():
            output: AmlOutput = self.forward(explained_model_inputs = batch)

            results = StepOutput(loss = output.loss_output.loss, prediction_loss = output.loss_output.prediction_loss,
                                 prediction_loss_weighted = output.loss_output.prediction_loss_weighted,
                                 inverse_loss = output.loss_output.inverse_loss,
                                 inverse_loss_weighted = output.loss_output.inverse_loss_weighted,
                                 regularization_loss = output.loss_output.regularization_loss,
                                 explained_model_predicted_class = output.explained_model_predicted_class,
                                 explained_model_logits = output.explained_model_predicted_logits,
                                 regularization_weighted = output.loss_output.regularization_loss_weighted,
                                 tokens_attr = output.tokens_attr, input = batch)

            self.val_step_outputs.append(results)
            return results

    def on_train_epoch_end(self):
        loss, prediction_loss, regularization_loss, prediction_loss_weighted, regularization_loss_weighted, inverse_loss, inverse_loss_weighted = self.get_mean_results(
            self.training_step_outputs)
        log_dict = dict(loss = loss.item(), prediction_loss = prediction_loss.item(),
                        regularization_loss = regularization_loss.item(), inverse_loss = inverse_loss.item(),
                        inverse_loss_weighted = inverse_loss_weighted.item(),
                        prediction_loss_weighted = prediction_loss_weighted.item(),
                        regularization_loss_weighted = regularization_loss_weighted.item())
        log_dict = {"Train_epoch_end/" + key: round(value, self.log_round_digits) for key, value in log_dict.items()}
        self.log_dict(log_dict, on_step = False, on_epoch = True)
        # self.logger.log_metrics(log_dict)

        self.training_step_outputs.clear()
        return {'avg_train_loss': loss}

    def on_validation_epoch_end(self):
        loss, prediction_loss, regularization_loss, prediction_loss_weighted, regularization_loss_weighted, inverse_loss, inverse_loss_weighted = self.get_mean_results(
            self.val_step_outputs)
        log_dict = dict(loss = loss.item(), prediction_loss = prediction_loss.item(),
                        regularization_loss = regularization_loss.item(), inverse_loss = inverse_loss.item(),
                        inverse_loss_weighted = inverse_loss_weighted.item(),
                        prediction_loss_weighted = prediction_loss_weighted.item(),
                        regularization_loss_weighted = regularization_loss_weighted.item())
        log_dict = {"Val_epoch_end/" + key: round(value, self.log_round_digits) for key, value in log_dict.items()}
        self.log_dict(log_dict)

        metric_results = []
        for step_output in self.val_step_outputs:
            if len(step_output.tokens_attr) != 1:
                raise ValueError(f"val use batch_size 1 only!")
            # validate for bert - step_output.tokens_attr[0]
            tokens_attr = step_output.tokens_attr[0].detach().squeeze()
            if tokens_attr.dim() == 0:
                tokens_attr = tokens_attr.unsqueeze(0)
            item_data = DataForEvaluation(tokens_attr = tokens_attr,
                                          explained_model_predicted_class = step_output.explained_model_predicted_class.squeeze(),
                                          explained_model_predicted_logits = step_output.explained_model_logits.squeeze(),
                                          input = DataForEvaluationInputs(
                                              input_ids = step_output.input[EXPLAINED_INPUT_IDS_NAME],
                                              attention_mask = step_output.input[EXPLAINED_ATTENTION_MASK_NAME],
                                              task_prompt_input_ids = step_output.input[TASK_PROMPT_INPUT_IDS],
                                              label_prompt_input_ids = step_output.input[LABEL_PROMPT_INPUT_IDS],
                                              task_prompt_attention_mask = step_output.input[
                                                  TASK_PROMPT_ATTENTION_MASK],
                                              label_prompt_attention_mask = step_output.input[
                                                  LABEL_PROMPT_ATTENTION_MASK]))
            evaluation_result, evaluation_item = evaluate_tokens_attributions(model = self.explained_model,
                                                                              explained_tokenizer = self.explained_tokenizer,
                                                                              ref_token_id = self.ref_token_id,
                                                                              data = item_data, step = self.global_step,
                                                                              epoch = self.current_epoch,
                                                                              item_index = "",
                                                                              experiment_path = self.experiment_path, )
            metric_results.append(evaluation_result)
        metric_results_mean = sum(metric_results) / len(metric_results)
        metric_results_dict = {ExpArgs.eval_metric: metric_results_mean}

        new_logs = {f"Val_metric/{k}": v for k, v in metric_results_dict.items()}
        self.log_dict(new_logs, on_step = False, on_epoch = True)
        evaluation_result = metric_results_dict[ExpArgs.eval_metric]
        self.val_step_outputs.clear()

        if ExpArgs.is_save_model:
            direction = MetricsMetaData.directions[ExpArgs.eval_metric]
            if (self.prev_metric_result is None) or (
                    ((evaluation_result < self.prev_metric_result) and (direction == DirectionTypes.MIN.value)) or (
                    (evaluation_result > self.prev_metric_result) and (direction == DirectionTypes.MAX.value))):
                ckp_path = self.checkpoints_path
                if os.path.exists(ckp_path):
                    shutil.rmtree(ckp_path)
                save_checkpoint(model = self.interpreter_model, tokenizer = self.interpreter_tokenizer,
                                path_dir = ckp_path)
                if is_add_label_embedding():
                    torch.save({NEW_ADDED_TRAINABLE_PARAMS: self.trainable_embeddings.state_dict(), },
                               f'{ckp_path}/{NEW_ADDED_TRAINABLE_PARAMS}.pth')

                pd.DataFrame(dict(epoch = [self.current_epoch], step = [self.global_step])).to_pickle(
                    f"{ckp_path}/MORE_INFO_{get_current_time()}.pkl")
                self.prev_metric_result = evaluation_result

        return {**dict(loss = loss.cpu().item()), **metric_results_dict}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr = ExpArgs.lr)

        if ExpArgs.scheduler_type == SchedulerTypes.LINEAR_SCHEDULE_WITH_WARMUP.value:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = self.warmup_steps,
                                                        num_training_steps = self.n_training_steps)

        elif ExpArgs.scheduler_type == SchedulerTypes.COSINE_SCHEDULE_WITH_WARMUP.value:
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = self.warmup_steps,
                                                        num_training_steps = self.n_training_steps, num_cycles = 0.5,
                                                        last_epoch = -1)
        elif ExpArgs.scheduler_type == SchedulerTypes.COSINE_WITH_HARD_RESTARTS_SCHEDULE_WITH_WARMUP.value:
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                           num_warmup_steps = self.warmup_steps,
                                                                           num_training_steps = self.n_training_steps,
                                                                           num_cycles = 0.5, last_epoch = -1)
        elif ExpArgs.scheduler_type == SchedulerTypes.CONSTANT_SCHEDULE_WITH_WARMUP.value:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps = self.warmup_steps,
                                                          num_training_steps = self.n_training_steps)
        else:
            raise ValueError(f"unsupported scheduler type")
        return dict(optimizer = optimizer, lr_scheduler = dict(scheduler = scheduler, interval = "step"))

    def freeze_layers(self):
        backbone_name = BackbonesMetaData.name[ExpArgs.interpreter_model_backbone]
        c_model = getattr(self.interpreter_model, backbone_name)
        modules = [c_model.embeddings]

        # if ExpArgs.interpreter_model_n_first_layers_to_freeze == -1:
        #     modules.append(c_model)
        # modules = [c_model.embeddings, c_model.encoder.layer[:ExpArgs.interpreter_model_n_first_layers_to_freeze]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    @staticmethod
    def get_mean_results(all_outputs: List[StepOutput]):
        loss = torch.mean(torch.stack([output.loss for output in all_outputs]))
        prediction_loss = torch.mean(torch.stack([output.prediction_loss for output in all_outputs]))
        regularization_loss = torch.mean(torch.stack([output.regularization_loss for output in all_outputs]))
        inverse_loss = torch.mean(torch.stack([output.inverse_loss for output in all_outputs]))

        prediction_loss_weighted = torch.mean(torch.stack([output.prediction_loss_weighted for output in all_outputs]))
        regularization_loss_weighted = torch.mean(
            torch.stack([output.regularization_weighted for output in all_outputs]))
        inverse_loss_weighted = torch.mean(torch.stack([output.inverse_loss_weighted for output in all_outputs]))
        return loss, prediction_loss, regularization_loss, prediction_loss_weighted, regularization_loss_weighted, inverse_loss, inverse_loss_weighted

    def calculate_regularization(self, tokens_attr, special_tokens_indices: Tensor):
        regularization_tokens_attr = tokens_attr
        # regularization_tokens_attr = tokens_attr.clone()
        # regularization_tokens_attr[special_tokens_indices] = 0
        if ExpArgs.regularization_type == RegularizationTypes.BCE.value:
            if is_model_encoder_only(ExpArgs.explained_model_backbone):
                regularization_loss = encourage_token_attr_to_prior_loss(tokens_attr = regularization_tokens_attr,
                                                                         prior = 0)
            else:
                all_results = []
                for t in regularization_tokens_attr:
                    all_results.append(encourage_token_attr_to_prior_loss(tokens_attr = t, prior = 0))
                regularization_loss = torch.stack(all_results).mean()
        elif ExpArgs.regularization_type == RegularizationTypes.L1.value:
            regularization_loss = l1_loss(regularization_tokens_attr)
        else:
            raise f"Value of self.mask_loss_type is not recognized"

        return regularization_loss

    def calculate_loss(self, explained_model_logits: Tensor, inverse_explained_model_logits: Tensor,
                       target_probabilities: Tensor, explained_tokens_attribution: List[Tensor],
                       interpreter_tokens_attribution: List[Tensor], special_tokens_indices: Tensor) -> LossOutput:
        regularization_loss = self.calculate_regularization(interpreter_tokens_attribution, special_tokens_indices)
        prediction_loss = calculate_prediction_loss(output = explained_model_logits, target = target_probabilities)
        inverse_loss = calculate_inverse_loss(logits = inverse_explained_model_logits, target = target_probabilities)

        prediction_loss_weighted = LossCoefficients.prediction_loss_weight * prediction_loss
        regularization_loss_weighted = LossCoefficients.regularization_loss_weight * regularization_loss
        inverse_loss_weighted = LossCoefficients.inverse_loss_weight * inverse_loss

        loss = prediction_loss_weighted + regularization_loss_weighted + inverse_loss_weighted

        return LossOutput(loss = loss, prediction_loss_weighted = prediction_loss_weighted,
                          inverse_loss_weighted = inverse_loss_weighted,
                          regularization_loss_weighted = regularization_loss_weighted,
                          prediction_loss = prediction_loss, inverse_loss = inverse_loss,
                          regularization_loss = regularization_loss)


    # def on_before_optimizer_step(self, optimizer):
    #     for idx, param_group in enumerate(optimizer.param_groups):
    #         lr = param_group['lr']
    #         log_dict = {}
    #         log_dict[f"Lr/learning_rate_{idx}"] = float(lr)
    #         # print(f'LR at step {self.global_step} for group {idx}: {lr}')
    #         self.log_dict(log_dict, on_step = True, on_epoch = False)


    # def on_after_backward(self):
        # for name, param in self.named_parameters():
        #     log_dict = {}
        #
        #     if param.grad is not None:
        #         log_dict[f'Params/grad_norm_{name}'] = float(param.grad.norm().item())
        #     else:
        #         log_dict[f'Params/grad_norm_{name}'] = float(-1_000)
        #
        #     if param.grad is not None and torch.isnan(param.grad).any():
        #         print(f"NaN detected in gradients of {name}")
        #     print(log_dict)
        #
        #     self.log_dict(log_dict, on_step = True, on_epoch = False)


