import copy
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from config.config import ExpArgs
from main.data_module import TextSeqDataModule
from main.hp_search import HpSearch
from main.seq_cls_utils import init_exp, set_hp, save_running_time
from models.classification_with_token_classification_model_fine_tune import \
    ClassificationWithTokenClassificationModelFineTune
from models.train_models_utils import get_tokens_attr_generation_model, get_transformers_seq_cls_models


class FineTune:

    def __init__(self, hp: dict, exp_name: str):
        init_exp()
        set_hp(hp)
        self.exp_name = exp_name
        self.pretrain_path = f"{ExpArgs.default_root_dir}/FINE_TUNE"

    def run(self):

        begin = time.time()
        ExpArgs.scheduler_type = ExpArgs.fine_tune_scheduler_type

        task = ExpArgs.task

        model_for_tokens_attr_generation = get_tokens_attr_generation_model()
        model_for_sequence_classification = get_transformers_seq_cls_models()

        data_module = TextSeqDataModule(train_sample=ExpArgs.task.train_sample, test_sample=ExpArgs.task.test_sample)

        perturbation_results_path = str(Path(self.pretrain_path, "RESULTS_DF", self.exp_name))
        tb_logger = TensorBoardLogger(Path(self.pretrain_path, "TB_LOGS", self.exp_name))

        device = "gpu" if torch.cuda.is_available() else "cpu"
        warmup_steps = 0
        for idx, item in enumerate(data_module.val_dataset):
            current_model = ClassificationWithTokenClassificationModelFineTune(
                model_for_sequence_classification=model_for_sequence_classification,
                model_for_tokens_attr_generation=copy.deepcopy(model_for_tokens_attr_generation),
                explained_tokenizer=data_module.explained_tokenizer,
                explainer_tokenizer=data_module.explainer_tokenizer, total_training_steps=warmup_steps,
                experiment_path=perturbation_results_path, checkpoints_path="", lr=ExpArgs.lr,
                warmup_steps=warmup_steps)
            item_module = TextSeqDataModule(data=item, explained_tokenizer=data_module.explained_tokenizer,
                                            explainer_tokenizer=data_module.explainer_tokenizer,
                                            task_prompt_input_ids=data_module.task_prompt_input_ids,
                                            label_prompt_input_ids=data_module.label_prompt_input_ids)

            ExpArgs.is_save_model = False
            ExpArgs.is_save_results = True
            ExpArgs.verbose = True

            current_model.init_data(item_idx=idx)
            trainer = pl.Trainer(accelerator=device, max_epochs=ExpArgs.n_epochs_for_fine_tune, logger=tb_logger,
                                 num_sanity_val_steps=ExpArgs.num_sanity_val_steps, enable_progress_bar=False,
                                 enable_checkpointing=False)
            trainer.fit(model=current_model, datamodule=item_module)

            del current_model

        end = time.time()

        save_running_time(end, begin, self.exp_name, file_type="FineTune")