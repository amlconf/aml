import sys

sys.path.append("..")

from config.config import ExpArgs
from config.tasks import RTN_TASK
from main.hp_search import HpSearch
from main.run_fine_tune import FineTune
from main.run_pre_train import PreTrain
from utils.utils_functions import get_current_time
from config.types_enums import (EvalMetric, ModelBackboneTypes, RefTokenNameTypes)

ExpArgs.task = RTN_TASK
ExpArgs.explained_model_backbone = ModelBackboneTypes.BERT.value
ExpArgs.explainer_model_backbone = ModelBackboneTypes.BERT.value
ExpArgs.ref_token_name = RefTokenNameTypes.MASK.value
ExpArgs.eval_metric = EvalMetric.COMPREHENSIVENESS.value

time_str = get_current_time()
exp_name_prefix = f"explained_model_backbone_{ExpArgs.explained_model_backbone}_metric_{ExpArgs.eval_metric}"

# ------------------- start running -----------------
# Run hyper params search
hp_exp_name = f"hp_{exp_name_prefix}_{time_str}"
hp = HpSearch(hp_exp_name, n_trials = ExpArgs.task.hp_search_n_trials).run()

# Run pretrain
pre_train_exp_name = f"pt_{exp_name_prefix}_{time_str}"
pretrain_model_path = PreTrain(hp, pre_train_exp_name).run()

ExpArgs.fine_tuned_model_for_tokens_attr_generation = pretrain_model_path

# Run fine tune
fine_tune_exp_name = f"ft_{exp_name_prefix}_{time_str}"
FineTune(hp, fine_tune_exp_name).run()

