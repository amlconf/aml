from pathlib import Path
from typing import List

import torch

from config.config import ExpArgs
from config.constants import LABELS_NAME, NEW_ADDED_TRAINABLE_PARAMS
from evaluations.evaluations import evaluate_tokens_attr
from models.classification_with_token_classification_model import ClassificationWithTokenClassificationModel
from models.train_models_utils import is_use_add_label
from utils.dataclasses.trainer_outputs import DataForEval


class ClassificationWithTokenClassificationModelFineTune(ClassificationWithTokenClassificationModel):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.MAX_EXAMPLES_TO_PRINT = 5
        self.item_idx = -1
        self.training_step_outputs: List[DataForEval] = []
        self.load_trainable_embeddings()

    def init_data(self, item_idx: int):
        self.item_idx = item_idx

    def load_trainable_embeddings(self):
        file_path = f"{ExpArgs.fine_tuned_model_for_tokens_attr_generation}/{NEW_ADDED_TRAINABLE_PARAMS}.pth"
        if is_use_add_label() and Path(file_path).is_file():
            checkpoint = torch.load(file_path)
            self.trainable_embeddings.load_state_dict(checkpoint[NEW_ADDED_TRAINABLE_PARAMS])

            for p in self.trainable_embeddings.parameters():
                p.requires_grad = False

    def training_step(self, batch, batch_idx):
        self.model_seq_cls.eval()
        if ExpArgs.is_fine_tune_attr_gen_train:
            self.model_tokens_attr_gen.train()
        else:
            self.model_tokens_attr_gen.eval()
        gt_target = batch[LABELS_NAME]
        output = self.forward(inputs=batch, gt_target=gt_target)

        results = DataForEval(loss=output.loss_output.loss, pred_loss=output.loss_output.pred_loss,
                              pred_loss_mul=output.loss_output.prediction_loss_multiplied,
                              tokens_attr_sparse_loss=output.loss_output.tokens_attr_sparse_loss,
                              pred_origin=output.pred_origin, pred_origin_logits=output.pred_origin_logits,
                              tokens_attr_sparse_loss_mul=output.loss_output.mask_loss_multiplied,
                              tokens_attr=output.tokens_attr, input=batch, gt_target=gt_target)
        self.training_step_outputs.append(results)
        return dict(loss=results.loss)

    def validation_step(self, batch, batch_idx):
        pass

    def on_train_epoch_end(self):
        loss = torch.mean(torch.stack([output.loss for output in self.training_step_outputs]))
        if self.current_epoch >= ExpArgs.start_epoch_to_evaluate:
            self.run_perturbation_test(self.training_step_outputs)
        self.training_step_outputs.clear()
        return dict(loss=loss)

    def on_validation_epoch_end(self):
        pass

    def run_perturbation_test(self, outputs: List[DataForEval]):
        verbose = ExpArgs.verbose
        if self.item_idx > self.MAX_EXAMPLES_TO_PRINT:
            verbose = False

        metric_results = evaluate_tokens_attr(model=self.model_seq_cls,
                                              explained_tokenizer=self.explained_tokenizer,
                                              ref_token_id=self.ref_token_id, outputs=outputs, stage="Fine_tuned",
                                              experiment_path=self.experiment_path, verbose=verbose,
                                              item_index=self.item_idx, step=self.global_step, epoch=self.current_epoch,
                                              is_sequel=True)
        metric_results_dict = {ExpArgs.eval_metric: metric_results.item()}

        new_logs = {f"Val_metric/{k}": v for k, v in metric_results_dict.items()}
        self.log_dict(new_logs, on_step=False, on_epoch=True)