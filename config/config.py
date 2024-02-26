import torch

from config.types_enums import *
from config.types_enums import SchedulerTypes
from utils.dataclasses import Task


class ExpArgs:
    seed = 42
    n_epochs_for_pre_train = 5
    warmup_ratio = 0
    enable_checkpointing = False
    default_root_dir = "OUT"
    tokens_attr_with_ref_token_function_type: TokensAttrWithRefTokenFunctionTypes = TokensAttrWithRefTokenFunctionTypes.LINEAR.value
    tokens_attr_sparse_loss_type: TokenAttrSparseLossType = TokenAttrSparseLossType.BCE.value
    ref_token_name: RefTokenNameTypes = RefTokenNameTypes.MASK.value
    tokens_attr_loss_mul = 10
    prediction_loss_mul = 1
    inverse_token_attr_function_type_mul = 0
    lr = 10e-05
    start_epoch_to_evaluate = 0
    verbose = True
    n_epochs_for_fine_tune = 5
    batch_size = 20
    accumulate_grad_batches = 1
    eval_batch_size = 1
    num_sanity_val_steps = -1
    val_check_interval = 0.32
    is_save_model = False
    log_every_n_steps = 40
    inverse_token_attr_function = InverseTokenAttrFunctionTypes.NEGATIVE_PROB_LOSS.value
    eval_metric: EvalMetric = EvalMetric.POS_AUC_WITH_REFERENCE_TOKEN.value
    is_save_results = False
    task: Task = None
    explained_model_backbone: ModelBackboneTypes = ModelBackboneTypes.ROBERTA.value
    explainer_model_backbone: ModelBackboneTypes = ModelBackboneTypes.ROBERTA.value
    validation_type = ValidationType.VAL.value
    fine_tuned_model_for_tokens_attr_generation = None
    run_type = None
    labels_tokens_opt = None
    scheduler_type = SchedulerTypes.LINEAR_SCHEDULE_WITH_WARMUP.value
    fine_tune_scheduler_type = SchedulerTypes.LINEAR_SCHEDULE_WITH_WARMUP.value
    explainer_model_n_first_layers_to_freeze = None
    is_fine_tune_attr_gen_train = False
    add_label_token_to_attr_get_type = AddLabelTokenToAttrGetType.AFTER_LAST_SEP.value
    attr_gen_token_classifier_size: int = 2
    attr_gen_token_classifier_mid_activation_function: ActivationFunctionTypes = ActivationFunctionTypes.TANH.value
    add_label_token_with_label_token = True
    llama_prompt_type: ModelPromptType = ModelPromptType.FEW_SHOT.value
    cross_tokenizers_pooling = CrossTokenizersPooling.MAX.value
    llama_f16 = True


ExpArgsDefault = type('ClonedExpArgs', (), vars(ExpArgs).copy())


class MetricsMetaData:
    directions = {EvalMetric.SUFFICIENCY.value: DirectionTypes.MIN.value,
                  EvalMetric.COMPREHENSIVENESS.value: DirectionTypes.MAX.value,
                  EvalMetric.EVAL_LOG_ODDS.value: DirectionTypes.MIN.value,
                  EvalMetric.AOPC_SUFFICIENCY.value: DirectionTypes.MIN.value,
                  EvalMetric.AOPC_COMPREHENSIVENESS.value: DirectionTypes.MAX.value,
                  EvalMetric.POS_AUC_WITH_REFERENCE_TOKEN.value: DirectionTypes.MIN.value,
                  EvalMetric.NEG_AUC_WITH_REFERENCE_TOKEN.value: DirectionTypes.MAX.value}

    top_k = {EvalMetric.SUFFICIENCY.value: [20], EvalMetric.COMPREHENSIVENESS.value: [20],
             EvalMetric.EVAL_LOG_ODDS.value: [20], EvalMetric.AOPC_SUFFICIENCY.value: torch.arange(20, 100, 20),
             EvalMetric.AOPC_COMPREHENSIVENESS.value: torch.arange(20, 100, 20),
             EvalMetric.POS_AUC_WITH_REFERENCE_TOKEN.value: torch.arange(20, 100, 20),
             EvalMetric.NEG_AUC_WITH_REFERENCE_TOKEN.value: torch.arange(20, 100, 20)}


class BackbonesMetaData:
    name = {  #
        ModelBackboneTypes.BERT.value: "bert",  #
        ModelBackboneTypes.ROBERTA.value: "roberta",  #
        ModelBackboneTypes.DISTILBERT.value: "distilbert",  #
        ModelBackboneTypes.LLAMA.value: "model"  #
    }
