import os
import pickle
import time
from pathlib import Path

import optuna
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from config.config import ExpArgs, MetricsMetaData
from config.types_enums import (InverseTokenAttrFunctionTypes, DirectionTypes, ModelBackboneTypes)
from main.data_module import TextSeqDataModule
from main.seq_cls_utils import init_exp, save_running_time
from models.classification_with_token_classification_model import (ClassificationWithTokenClassificationModel)
from models.classification_with_token_classification_model_fine_tune import \
    ClassificationWithTokenClassificationModelFineTune
from models.train_models_utils import (get_tokens_attr_generation_model, get_transformers_seq_cls_models,
                                       get_warmup_steps_and_total_training_steps)


class HpSearch:

    def __init__(self, exp_name: str, model_for_tokens_attr_generation = None, model_for_sequence_classification = None,
                 data_module = None, is_run_all_hp_search = True, n_trials: int = 0, item_idx: int = -1,
                 warmup_steps = None, total_training_steps = None):
        init_exp()
        ExpArgs.is_save_model = False
        ExpArgs.is_save_results = False
        ExpArgs.verbose = False
        self.exp_name = exp_name
        self.conf_path = f"{ExpArgs.default_root_dir}/CONFIG"
        self.monitor = f"Val_metric/{ExpArgs.eval_metric}"
        is_direction_max = MetricsMetaData.directions[ExpArgs.eval_metric] == DirectionTypes.MAX.value
        self.direction = "maximize" if is_direction_max else "minimize"

        self.item_idx = item_idx
        self.model_for_tokens_attr_generation = model_for_tokens_attr_generation
        self.model_for_sequence_classification = model_for_sequence_classification
        self.data_module = data_module
        self.is_run_all_hp_search = is_run_all_hp_search
        self.n_trials = n_trials
        self.warmup_steps = warmup_steps
        self.total_training_steps = total_training_steps
        if model_for_sequence_classification:
            self.model_for_sequence_classification = model_for_sequence_classification
        else:
            self.model_for_sequence_classification = get_transformers_seq_cls_models()

    def objective(self, trial):
        ExpArgs.lr = ExpArgs.task.default_lr
        if ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
            ExpArgs.lr = ExpArgs.task.llama_lr

        ExpArgs.prediction_loss_mul = trial.suggest_int('prediction_loss_mul', low = 1, high = 100)
        ExpArgs.tokens_attr_loss_mul = trial.suggest_int('tokens_attr_loss_mul', low = 1, high = 100)
        ExpArgs.inverse_token_attr_function_type_mul = trial.suggest_int('inverse_token_attr_function_type_mul',
                                                                         low = 1, high = 100)

        if self.item_idx > -1:
            model_for_tokens_attr_generation = self.model_for_tokens_attr_generation
            data_module = self.data_module
        else:
            model_for_tokens_attr_generation = get_tokens_attr_generation_model()
            data_module = TextSeqDataModule(train_sample = ExpArgs.task.hp_search_train_sample,
                                            test_sample = ExpArgs.task.hp_search_test_sample,
                                            val_type = ExpArgs.validation_type)

        if self.total_training_steps:
            warmup_steps = self.warmup_steps
            total_training_steps = self.total_training_steps
        else:
            warmup_steps, total_training_steps = get_warmup_steps_and_total_training_steps(
                n_epochs = ExpArgs.n_epochs_for_pre_train, train_samples_length = len(data_module.train_dataset),
                batch_size = ExpArgs.batch_size, warmup_ratio = ExpArgs.warmup_ratio)
        # print(f"total_training_steps: {total_training_steps}")

        if self.item_idx > -1:
            model = ClassificationWithTokenClassificationModelFineTune(
                model_for_sequence_classification = self.model_for_sequence_classification,
                model_for_tokens_attr_generation = model_for_tokens_attr_generation,
                explained_tokenizer = data_module.explained_tokenizer,
                explainer_tokenizer = data_module.explainer_tokenizer, total_training_steps = total_training_steps,
                experiment_path = "", checkpoints_path = "", lr = ExpArgs.lr, warmup_steps = warmup_steps)
        else:
            model = ClassificationWithTokenClassificationModel(
                model_for_sequence_classification = self.model_for_sequence_classification,
                model_for_tokens_attr_generation = model_for_tokens_attr_generation,
                explained_tokenizer = data_module.explained_tokenizer,
                explainer_tokenizer = data_module.explainer_tokenizer, total_training_steps = total_training_steps,
                experiment_path = "", checkpoints_path = "", lr = ExpArgs.lr, warmup_steps = warmup_steps)

        tb_logger = TensorBoardLogger(Path(self.conf_path, "TB_LOGS", self.exp_name))
        device = "gpu" if torch.cuda.is_available() else "cpu"
        trainer = pl.Trainer(accelerator = device, max_epochs = 1, logger = tb_logger, enable_progress_bar = False,
                             enable_model_summary = False, default_root_dir = self.conf_path,
                             num_sanity_val_steps = ExpArgs.num_sanity_val_steps,
                             val_check_interval = ExpArgs.val_check_interval,
                             accumulate_grad_batches=ExpArgs.accumulate_grad_batches,
                             callbacks = [ModelCheckpoint(save_top_k = 0, monitor = self.monitor),
                                          EarlyStopping(monitor = self.monitor, patience = 2)])

        trainer.fit(model = model, datamodule = data_module)

        return trainer.callback_metrics[self.monitor].item()

    def run(self):

        begin = time.time()

        os.makedirs(self.conf_path, exist_ok = True)
        result_path = f"{self.conf_path}/OPTUNA_RESULTS"
        os.makedirs(result_path, exist_ok = True)

        exp_args_path = f"{self.conf_path}/EXPERIMENT_ARGUMENTS"
        os.makedirs(exp_args_path, exist_ok = True)

        study = optuna.create_study(direction = self.direction,
                                    sampler = optuna.samplers.TPESampler(seed = ExpArgs.seed))
        study.optimize(self.objective, n_trials = self.n_trials)

        best_trial = study.best_trial
        print("Best trial:")
        print("Value: ", best_trial.value)
        print("Params: ", best_trial.params)

        file_name = f'{self.exp_name}.pkl'
        if self.item_idx:
            file_name = f'{self.exp_name}_item_{self.item_idx}.pkl'

        with open(f"{result_path}/{file_name}", 'wb') as file:
            pickle.dump({**{"_best_trial_value_": best_trial.value}, **best_trial.params,
                         **{"_is_run_all_hp_search": self.is_run_all_hp_search, "_eval_metric": ExpArgs.eval_metric,
                            "_explainer_model_backbone": ExpArgs.explainer_model_backbone,
                            "_explained_model_backbone": ExpArgs.explained_model_backbone,
                            "_validation_type": ExpArgs.validation_type,
                            "_run_type": ExpArgs.run_type}}, file)

        if self.is_run_all_hp_search:
            exp_args_dict = {}
            for k, v in vars(ExpArgs).items():
                if "__" not in k:
                    exp_args_dict[k] = v
            with open(f"{exp_args_path}/{file_name}", 'wb') as file:
                pickle.dump(exp_args_dict, file)

        end = time.time()

        save_running_time(end, begin, self.exp_name, file_type = "HpSearch")

        return best_trial.params