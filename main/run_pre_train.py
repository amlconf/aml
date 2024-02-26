import os
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from config.config import ExpArgs
from main.data_module import TextSeqDataModule
from main.seq_cls_utils import init_exp, set_hp, save_running_time
from models.classification_with_token_classification_model import (ClassificationWithTokenClassificationModel)
from models.train_models_utils import (get_tokens_attr_generation_model, get_transformers_seq_cls_models,
                                       get_warmup_steps_and_total_training_steps)


class PreTrain:

    def __init__(self, hp: dict, exp_name: str):
        init_exp()
        set_hp(hp)
        ExpArgs.is_save_model = True
        ExpArgs.is_save_results = True
        ExpArgs.verbose = False

        self.exp_name = exp_name
        self.pretrain_path = f"{ExpArgs.default_root_dir}/PRE_TRAIN"

    def run(self):
        begin = time.time()

        os.makedirs(self.pretrain_path, exist_ok = True)

        model_for_tokens_attr_generation = get_tokens_attr_generation_model()
        model_for_sequence_classification = get_transformers_seq_cls_models()
        data_module = TextSeqDataModule(train_sample = ExpArgs.task.train_sample,
                                        test_sample = ExpArgs.task.test_sample, val_type = ExpArgs.validation_type)

        warmup_steps, total_training_steps = get_warmup_steps_and_total_training_steps(
            n_epochs = ExpArgs.n_epochs_for_pre_train, train_samples_length = len(data_module.train_dataset),
            batch_size = ExpArgs.batch_size, warmup_ratio = ExpArgs.warmup_ratio)

        perturbation_results_path = Path(self.pretrain_path, "RESULTS_DF", self.exp_name).__str__()
        checkpoints_path = Path(self.pretrain_path, "CHECKPOINTS", self.exp_name).__str__()
        tb_logger = TensorBoardLogger(Path(self.pretrain_path, "TB_LOGS", self.exp_name))

        model = ClassificationWithTokenClassificationModel(
            model_for_sequence_classification = model_for_sequence_classification,
            model_for_tokens_attr_generation = model_for_tokens_attr_generation,
            explained_tokenizer = data_module.explained_tokenizer,
            explainer_tokenizer = data_module.explainer_tokenizer, total_training_steps = total_training_steps,
            experiment_path = perturbation_results_path, checkpoints_path = checkpoints_path, lr = ExpArgs.lr,
            warmup_steps = warmup_steps)

        # print_number_of_trainable_and_not_trainable_params(model)

        device = "gpu" if torch.cuda.is_available() else "cpu"
        trainer = pl.Trainer(accelerator = device, max_epochs = ExpArgs.n_epochs_for_pre_train, logger = tb_logger,
                             enable_progress_bar = False, num_sanity_val_steps = ExpArgs.num_sanity_val_steps,
                             default_root_dir = ExpArgs.default_root_dir,
                             val_check_interval = ExpArgs.val_check_interval,
                             log_every_n_steps = ExpArgs.log_every_n_steps,
                             accumulate_grad_batches = ExpArgs.accumulate_grad_batches,
                             enable_checkpointing = ExpArgs.enable_checkpointing)

        trainer.fit(model = model, datamodule = data_module)

        end = time.time()

        save_running_time(end, begin, self.exp_name, file_type = "PreTrain")

        return checkpoints_path
