import math
import os
import shutil
from typing import List

import pandas as pd
import pytorch_lightning as pl
import torch
from lightning_fabric import seed_everything
from torch import Tensor, nn
from torch.optim import AdamW
from transformers import (get_linear_schedule_with_warmup, AutoTokenizer, get_cosine_schedule_with_warmup,
                          get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule_with_warmup)

from config.config import ExpArgs, MetricsMetaData, BackbonesMetaData, TokenAttrSparseLossType
from config.constants import (LABELS_NAME, INPUT_IDS_NAME, ATTENTION_MASK_NAME, NEW_ADDED_TRAINABLE_PARAMS,
                              EXPLAINER_ATTENTION_MASK_NAME, EXPLAINER_INPUT_IDS_NAME, MAP_TOKENS, NAN_FLOAT,
                              LABEL_PROMPT_KEY, TASK_PROMPT_KEY)
from config.types_enums import (RefTokenNameTypes, TokensAttrWithRefTokenFunctionTypes, DirectionTypes,
                                ModelBackboneTypes, SchedulerTypes, AddLabelTokenToAttrGetType, CrossTokenizersPooling)
from evaluations.evaluations import evaluate_tokens_attr
from main.seq_cls_utils import (save_checkpoint, encourage_token_attr_to_prior_loss, l1_loss, prediction_loss,
                                cal_inverse_tokens_attr_pred_loss)
from models.train_models_utils import construct_word_embedding, is_use_add_label
from utils.dataclasses import EeaOutput, LossOutput
from utils.dataclasses.trainer_outputs import DataForEval
from utils.utils_functions import conv_class_to_dict, get_current_time, run_model, model_seq_cls_merge_inputs


class ClassificationWithTokenClassificationModel(pl.LightningModule):
    def __init__(self, model_for_sequence_classification, model_for_tokens_attr_generation,
                 explainer_tokenizer: AutoTokenizer, explained_tokenizer: AutoTokenizer, total_training_steps: int,
                 experiment_path: str, checkpoints_path: str, lr: float, warmup_steps: int):
        super().__init__()
        self.log_round_digits = 4
        self.task = ExpArgs.task
        self.warmup_steps = warmup_steps
        self.model_seq_cls = model_for_sequence_classification
        self.model_tokens_attr_gen = model_for_tokens_attr_generation
        self.freeze_layers()
        self.explainer_tokenizer = explainer_tokenizer
        self.explained_tokenizer = explained_tokenizer
        self.ref_token_id = self.get_ref_token_name()
        self.n_training_steps = total_training_steps
        self.experiment_path = experiment_path
        self.checkpoints_path = checkpoints_path
        self.lr = lr

        self.training_step_outputs: List[DataForEval] = []
        self.val_step_outputs: List[DataForEval] = []
        self.prev_metric_result = None

        torch.manual_seed(ExpArgs.seed)
        seed_everything(ExpArgs.seed)
        if is_use_add_label():
            trainable_embeddings_len = len(self.task.labels_str_int_maps.keys())
            if ExpArgs.add_label_token_with_label_token:
                trainable_embeddings_len = trainable_embeddings_len + 1
                self.label_id = torch.tensor(trainable_embeddings_len - 1)  # last item index
            self.trainable_embeddings = nn.Embedding(trainable_embeddings_len,
                                                     self.model_tokens_attr_gen.config.hidden_size,
                                                     padding_idx = self.model_tokens_attr_gen.config.pad_token_id)
            self.trainable_embeddings.weight.data.normal_(mean = 0.0,
                                                          std = self.model_tokens_attr_gen.config.initializer_range)

    def get_ref_token_name(self):
        if ExpArgs.ref_token_name == RefTokenNameTypes.MASK.value:
            return self.explained_tokenizer.mask_token_id
        elif ExpArgs.ref_token_name == RefTokenNameTypes.PAD.value:
            return self.explained_tokenizer.pad_token_id
        elif ExpArgs.ref_token_name == RefTokenNameTypes.UNK.value:
            return self.explained_tokenizer.unk_token_id
        else:
            raise ValueError("ref name invalid")

    def forward(self, inputs, gt_target = None):
        batch = conv_class_to_dict(inputs)
        inputs = model_seq_cls_merge_inputs(batch[INPUT_IDS_NAME], batch[TASK_PROMPT_KEY], batch[LABEL_PROMPT_KEY]).to(
            self.model_seq_cls.device)
        cls_output_logits = run_model(model = self.model_seq_cls, model_backbone = ExpArgs.explained_model_backbone,
                                      input_ids = inputs, attention_mask = batch[ATTENTION_MASK_NAME],
                                      is_return_logits = True)
        model_pred_origin = torch.argmax(cls_output_logits, dim = 1)

        tokens_attr = self.cal_tokens_attr(model_pred_origin, batch)
        tokens_attr = self.map_tokens_attr_handler(tokens_attr, batch)
        tokens_attr_for_multiply = [i.clone() for i in tokens_attr]
        for item in tokens_attr_for_multiply:
            item[torch.isnan(item)] = 1

        pertub_logits, inverse_pertub_output_logits = self.forward_with_tokens_attr_using_input_embed(batch,
                                                                                                      tokens_attr_for_multiply)

        loss_output = self.cal_loss(seq_cls_logits = pertub_logits,
                                    inverse_tokens_attr_seq_cls_logits = inverse_pertub_output_logits,
                                    target = cls_output_logits.argmax(dim = -1), tokens_attr = tokens_attr)

        return EeaOutput(loss_output = loss_output, pred_origin = model_pred_origin,
                         pred_origin_logits = cls_output_logits, tokens_attr = tokens_attr)

    def cal_tokens_attr(self, model_pred_origin, batch):
        input_ids = batch[EXPLAINER_INPUT_IDS_NAME].clone()
        attention_mask = batch[EXPLAINER_ATTENTION_MASK_NAME].clone()

        input_ids, attention_mask, inputs_embeds, labels_new_indices, long_vectors = self.add_token_gen_label_handler(
            model_pred_origin, input_ids, attention_mask)
        attr_scores = self.model_tokens_attr_gen(input_ids = input_ids, attention_mask = attention_mask,
                                                 inputs_embeds = inputs_embeds, labels_new_indices = labels_new_indices)
        attr_scores = self.re_swap_last_tokens(attr_scores, long_vectors)
        return attr_scores

    def map_tokens_attr_handler(self, tokens_attr, batch):
        if ExpArgs.explained_model_backbone != ModelBackboneTypes.LLAMA.value:
            return tokens_attr
        new_tokens_attr = []
        for batch_idx in range(len(batch[INPUT_IDS_NAME])):
            new_tokens_attr_lst = []
            for indices in batch[MAP_TOKENS][batch_idx]:
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

    def re_swap_last_tokens(self, attr_scores, long_vectors):
        if not is_use_add_label():
            return attr_scores

        # after remove label token
        if long_vectors is not None:
            batch_size = attr_scores.shape[0]
            zeros_vec = torch.zeros(batch_size, device = attr_scores.device).unsqueeze(-1)
            attr_scores = torch.cat((attr_scores, zeros_vec), dim = 1)

            self.swap(attr_scores, long_vectors, x = -1, y = -2)

        return attr_scores

    def swap_last_tokens(self, input_ids, attention_mask):
        l = -1  # last token
        bl = l - 1  # before last
        indices = None
        if input_ids.shape[-1] == self.explainer_tokenizer.model_max_length:
            indices = torch.nonzero(input_ids[:, l] == self.explainer_tokenizer.sep_token_id).squeeze()

            self.swap(input_ids, indices, bl, l)
            self.swap(attention_mask, indices, bl, l)

            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, :-1]
        return input_ids, attention_mask, indices

    @staticmethod
    def swap(t, vectors_idx, x, y):
        try:
            if not isinstance(vectors_idx, torch.Tensor):
                vectors_idx = torch.tensor(vectors_idx)
            if vectors_idx.dim() == 0:
                vectors_idx = vectors_idx.unsqueeze(0)
            t[vectors_idx, x], t[vectors_idx, y] = t[vectors_idx, y], t[vectors_idx, x]
        except Exception as e:
            print(f"vectors_idx: {vectors_idx}")
            print(f"t: {t}")
            raise ValueError(e)

    def get_trainable_embed_vec(self, i):
        vec = self.trainable_embeddings(torch.tensor(i).to(self.device))
        if ExpArgs.add_label_token_with_label_token:
            label_vec = self.trainable_embeddings(self.label_id.to(self.device))
            return vec + label_vec
        return vec

    def add_token_gen_label_handler(self, model_pred_origin, input_ids, attention_mask):
        add_label_type = ExpArgs.add_label_token_to_attr_get_type
        labels_new_indices = []

        if add_label_type == AddLabelTokenToAttrGetType.NONE.value:
            inputs_embeds, labels_new_indices, long_vectors = None, None, None
            return input_ids, attention_mask, inputs_embeds, labels_new_indices, long_vectors

        new_attention_vec = torch.ones(attention_mask.shape[0], device = attention_mask.device).unsqueeze(-1)
        input_ids, attention_mask, long_vectors = self.swap_last_tokens(input_ids, attention_mask)

        pred_embeds = [self.get_trainable_embed_vec(i) for i in model_pred_origin.tolist()]
        pred_embeds = torch.stack(pred_embeds).unsqueeze(1)

        inputs_embeds = construct_word_embedding(self.model_tokens_attr_gen, ExpArgs.explainer_model_backbone,
                                                 input_ids).to(self.device)

        if add_label_type == AddLabelTokenToAttrGetType.FIRST_TOKEN.value:
            attention_mask = torch.cat((new_attention_vec, attention_mask), dim = -1)
            inputs_embeds = torch.cat((pred_embeds, inputs_embeds), dim = 1)


        elif add_label_type == AddLabelTokenToAttrGetType.LAST_TOKEN.value:
            attention_mask = torch.cat((attention_mask, new_attention_vec), dim = -1)
            inputs_embeds = torch.cat((inputs_embeds, pred_embeds), dim = 1)


        elif add_label_type == AddLabelTokenToAttrGetType.AFTER_LAST_SEP.value:
            batch_size = attention_mask.shape[0]
            pad = self.explainer_tokenizer.pad_token_id
            pad_vec = torch.tensor([pad] * batch_size, device = attention_mask.device).unsqueeze(-1)
            pad_vec_embeds = construct_word_embedding(self.model_tokens_attr_gen, ExpArgs.explainer_model_backbone,
                                                      pad_vec).to(self.device)
            inputs_embeds = torch.cat((inputs_embeds, pad_vec_embeds), dim = 1)
            zeros_vec = torch.zeros(batch_size, device = attention_mask.device).unsqueeze(-1)
            attention_mask = torch.cat((attention_mask, zeros_vec), dim = -1)

            for item_idx, item in enumerate(input_ids):
                indices = (item == self.explainer_tokenizer.sep_token_id).nonzero(as_tuple = False).cpu()
                if len(indices) != 1:
                    raise ValueError(f"add_label_type after last sep. issue. len(indices): {len(indices)}")
                last_occurrence_idx = indices[0, -1].item()
                new_token_idx = last_occurrence_idx + 1
                inputs_embeds[item_idx, new_token_idx] = pred_embeds[item_idx]
                attention_mask[item_idx, new_token_idx] = 1
                labels_new_indices.append(new_token_idx)

        input_ids = None
        return input_ids, attention_mask, inputs_embeds, labels_new_indices, long_vectors

    def forward_with_tokens_attr_using_input_embed(self, batch, tokens_attr):
        if ExpArgs.explained_model_backbone != ModelBackboneTypes.LLAMA.value:
            inputs_embeds = construct_word_embedding(self.model_seq_cls, ExpArgs.explained_model_backbone,
                                                     batch[INPUT_IDS_NAME]).to(self.device)
            task_prompt_embeds, label_prompt_embeds = None, None
        else:
            inputs_embeds = [
                construct_word_embedding(self.model_seq_cls, ExpArgs.explained_model_backbone, i).to(self.device) for i
                in batch[INPUT_IDS_NAME]]
            task_prompt_embeds = [
                construct_word_embedding(self.model_seq_cls, ExpArgs.explained_model_backbone, i).to(self.device) for i
                in batch[TASK_PROMPT_KEY]]

            label_prompt_embeds = construct_word_embedding(self.model_seq_cls, ExpArgs.explained_model_backbone,
                                                           batch[LABEL_PROMPT_KEY]).to(self.device)

        pertub_output_logits = self.tokens_attr_using_input_embed(tokens_attr, inputs_embeds, batch, task_prompt_embeds,
                                                                  label_prompt_embeds)
        inverse_tokens_attr = [1 - i for i in tokens_attr]
        inverse_tokens_attr_pertub_output_logits = self.tokens_attr_using_input_embed(inverse_tokens_attr,
                                                                                      inputs_embeds, batch,
                                                                                      task_prompt_embeds,
                                                                                      label_prompt_embeds)
        return pertub_output_logits, inverse_tokens_attr_pertub_output_logits

    def tokens_attr_using_input_embed(self, tokens_attr, inputs_embeds, batch, task_prompt_embeds, label_prompt_embeds):
        input_ids = batch[INPUT_IDS_NAME]
        attention_mask = batch[ATTENTION_MASK_NAME]
        tokens_attr_function = ExpArgs.tokens_attr_with_ref_token_function_type
        if inputs_embeds[0].dim() != tokens_attr[0].dim():
            tokens_attr = [i.unsqueeze(1) for i in tokens_attr]

        if tokens_attr_function == TokensAttrWithRefTokenFunctionTypes.MUL.value:
            embedding_output = [tokens_attr[i] * inputs_embeds[i] for i in range(len(tokens_attr))]
        elif tokens_attr_function == TokensAttrWithRefTokenFunctionTypes.LINEAR.value:
            embedding_output = []
            for i in range(len(tokens_attr)):
                mask_inputs = input_ids[i].detach().clone().fill_(self.ref_token_id)
                mask_embedding_output = construct_word_embedding(self.model_seq_cls, ExpArgs.explained_model_backbone,
                                                                 mask_inputs)
                res = ((1 - tokens_attr[i]) * mask_embedding_output) + (tokens_attr[i] * inputs_embeds[i])
                embedding_output.append(res)
        else:
            raise ValueError("forbidden mask function")
        embedding_output = model_seq_cls_merge_inputs(embedding_output, task_prompt_embeds, label_prompt_embeds).to(
            self.model_seq_cls.device)
        return run_model(model = self.model_seq_cls, model_backbone = ExpArgs.explained_model_backbone,
                         inputs_embeds = embedding_output, attention_mask = attention_mask, is_return_logits = True)

    def training_step(self, batch, batch_idx):
        self.model_seq_cls.eval()
        self.model_tokens_attr_gen.train()

        gt_target = batch[LABELS_NAME]
        output = self.forward(inputs = batch, gt_target = gt_target)

        results = DataForEval(loss = output.loss_output.loss, pred_loss = output.loss_output.pred_loss,
                              pred_loss_mul = output.loss_output.prediction_loss_multiplied,
                              tokens_attr_sparse_loss = output.loss_output.tokens_attr_sparse_loss,
                              pred_origin = output.pred_origin, pred_origin_logits = output.pred_origin_logits,
                              tokens_attr_sparse_loss_mul = output.loss_output.mask_loss_multiplied,
                              tokens_attr = output.tokens_attr, input = batch, gt_target = gt_target)

        log_dict = dict(loss = results.loss.item(), prediction_loss = results.pred_loss.item(),
                        mask_loss = results.tokens_attr_sparse_loss.item(),
                        prediction_loss_mul = results.pred_loss_mul.item(),
                        mask_loss_mul = results.tokens_attr_sparse_loss_mul.item())
        log_dict = {"Train_step/" + key: round(value, self.log_round_digits) for key, value in log_dict.items()}
        self.log_dict(log_dict, on_step = True, on_epoch = False)
        # self.logger.log_metrics(log_dict)
        self.training_step_outputs.append(results)
        return dict(loss = results.loss)

    def validation_step(self, batch, batch_idx):

        self.model_seq_cls.eval()
        self.model_tokens_attr_gen.eval()

        with torch.no_grad():
            gt_target = batch[LABELS_NAME]
            output: EeaOutput = self.forward(inputs = batch, gt_target = gt_target)

            for t in output.tokens_attr:
                t[torch.isnan(t)] = -10e10

            results = DataForEval(loss = output.loss_output.loss, pred_loss = output.loss_output.pred_loss,
                                  pred_loss_mul = output.loss_output.prediction_loss_multiplied,
                                  tokens_attr_sparse_loss = output.loss_output.tokens_attr_sparse_loss,
                                  pred_origin = output.pred_origin, pred_origin_logits = output.pred_origin_logits,
                                  tokens_attr_sparse_loss_mul = output.loss_output.mask_loss_multiplied,
                                  tokens_attr = output.tokens_attr, input = batch, gt_target = gt_target)

            self.val_step_outputs.append(results)
            return results

    def on_train_epoch_end(self):
        loss, pred_loss, mask_loss, pred_loss_mul, mask_loss_mul = self.get_mean_results(self.training_step_outputs)
        log_dict = dict(loss = loss.item(), prediction_loss = pred_loss.item(), mask_loss = mask_loss.item(),
                        prediction_loss_mul = pred_loss_mul.item(), mask_loss_mul = mask_loss_mul.item())
        log_dict = {"Train_epoch_end/" + key: round(value, self.log_round_digits) for key, value in log_dict.items()}
        self.log_dict(log_dict, on_step = False, on_epoch = True)
        # self.logger.log_metrics(log_dict)

        self.training_step_outputs.clear()
        return {'avg_train_loss': loss}

    def on_validation_epoch_end(self):
        loss, pred_loss, mask_loss, pred_loss_mul, mask_loss_mul = self.get_mean_results(self.val_step_outputs)
        log_dict = dict(loss = loss.item(), prediction_loss = pred_loss.item(), mask_loss = mask_loss.item(),
                        prediction_loss_mul = pred_loss_mul.item(), mask_loss_mul = mask_loss_mul.item())
        log_dict = {"Val_epoch_end/" + key: round(value, self.log_round_digits) for key, value in log_dict.items()}
        self.log_dict(log_dict)

        metric_results = evaluate_tokens_attr(model = self.model_seq_cls,
                                              explained_tokenizer = self.explained_tokenizer,
                                              ref_token_id = self.ref_token_id, outputs = self.val_step_outputs,
                                              stage = "Val", step = self.global_step, epoch = self.current_epoch,
                                              item_index = -1, experiment_path = self.experiment_path,
                                              verbose = ExpArgs.verbose, is_sequel = False)
        metric_results_dict = {ExpArgs.eval_metric: metric_results.item()}

        new_logs = {f"Val_metric/{k}": v for k, v in metric_results_dict.items()}
        self.log_dict(new_logs, on_step = False, on_epoch = True)
        new_metric_result = metric_results_dict[ExpArgs.eval_metric]
        self.val_step_outputs.clear()

        if ExpArgs.is_save_model:
            direction = MetricsMetaData.directions[ExpArgs.eval_metric]
            if (self.prev_metric_result is None) or (
                    ((new_metric_result < self.prev_metric_result) and (direction == DirectionTypes.MIN.value)) or (
                    (new_metric_result > self.prev_metric_result) and (direction == DirectionTypes.MAX.value))):
                ckp_path = self.checkpoints_path
                if os.path.exists(ckp_path):
                    shutil.rmtree(ckp_path)
                save_checkpoint(model = self.model_tokens_attr_gen, tokenizer = self.explainer_tokenizer,
                                path_dir = ckp_path)
                if is_use_add_label():
                    torch.save({NEW_ADDED_TRAINABLE_PARAMS: self.trainable_embeddings.state_dict(), },
                               f'{ckp_path}/{NEW_ADDED_TRAINABLE_PARAMS}.pth')

                pd.DataFrame(dict(epoch = [self.current_epoch], step = [self.global_step])).to_pickle(
                    f"{ckp_path}/MORE_INFO_{get_current_time()}.pkl")
                self.prev_metric_result = new_metric_result

        return {**dict(loss = loss.cpu().item()), **metric_results_dict}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr = self.lr)

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
        for param in self.model_seq_cls.parameters():
            param.requires_grad = False

        backbone_name = BackbonesMetaData.name[ExpArgs.explainer_model_backbone]
        c_model = getattr(self.model_tokens_attr_gen, backbone_name)
        modules = [c_model.embeddings]

        if ExpArgs.explainer_model_n_first_layers_to_freeze == -1:
            modules.append(c_model)
        # modules = [c_model.embeddings, c_model.encoder.layer[:ExpArgs.explainer_model_n_first_layers_to_freeze]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    @staticmethod
    def get_mean_results(all_outputs: List[DataForEval]):
        loss = torch.mean(torch.stack([output.loss for output in all_outputs]))
        pred_loss = torch.mean(torch.stack([output.pred_loss for output in all_outputs]))
        mask_loss = torch.mean(torch.stack([output.tokens_attr_sparse_loss for output in all_outputs]))
        pred_loss_mul = torch.mean(torch.stack([output.pred_loss_mul for output in all_outputs]))
        mask_loss_mul = torch.mean(torch.stack([output.tokens_attr_sparse_loss_mul for output in all_outputs]))
        return loss, pred_loss, mask_loss, pred_loss_mul, mask_loss_mul

    def cal_loss(self, seq_cls_logits: Tensor, inverse_tokens_attr_seq_cls_logits: Tensor, target: Tensor,
                 tokens_attr: Tensor) -> LossOutput:
        if ExpArgs.tokens_attr_sparse_loss_type == TokenAttrSparseLossType.BCE.value:
            if ExpArgs.explained_model_backbone != ModelBackboneTypes.LLAMA.value:
                mask_loss = encourage_token_attr_to_prior_loss(tokens_attr = tokens_attr, prior = 0)
            else:
                all_results = []
                for t in tokens_attr:
                    nan_mask = torch.isnan(t)
                    tensor_without_nans = t[~nan_mask]
                    all_results.append(encourage_token_attr_to_prior_loss(tokens_attr = tensor_without_nans, prior = 0))
                mask_loss = torch.stack(all_results).mean()
        elif ExpArgs.tokens_attr_sparse_loss_type == TokenAttrSparseLossType.L1.value:
            mask_loss = l1_loss(tokens_attr)
        else:
            raise (f"Value of self.mask_loss_type is not recognized")

        pred_loss = prediction_loss(output = seq_cls_logits, target = target)
        inverse_tokens_attr_pred_loss = cal_inverse_tokens_attr_pred_loss(logits = inverse_tokens_attr_seq_cls_logits,
                                                                          target = target)

        prediction_loss_multiplied = ExpArgs.prediction_loss_mul * pred_loss
        mask_loss_multiplied = ExpArgs.tokens_attr_loss_mul * mask_loss
        opp_mask_prediction_loss_multiplied = ExpArgs.inverse_token_attr_function_type_mul * inverse_tokens_attr_pred_loss
        loss = prediction_loss_multiplied + mask_loss_multiplied + opp_mask_prediction_loss_multiplied

        return LossOutput(loss = loss, prediction_loss_multiplied = prediction_loss_multiplied,
                          opp_mask_prediction_loss_multiplied = opp_mask_prediction_loss_multiplied,
                          mask_loss_multiplied = mask_loss_multiplied, pred_loss = pred_loss,
                          inverse_tokens_attr_pred_loss = inverse_tokens_attr_pred_loss,
                          tokens_attr_sparse_loss = mask_loss)
