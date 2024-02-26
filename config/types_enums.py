from enum import Enum


class ModelBackboneTypes(Enum):
    BERT = 'BERT'
    ROBERTA = 'ROBERTA'
    DISTILBERT = 'DISTILBERT'
    LLAMA = 'LLAMA'


class TokensAttrWithRefTokenFunctionTypes(Enum):
    MUL = 'MUL'
    LINEAR = 'LINEAR'


class InverseTokenAttrFunctionTypes(Enum):
    MAX_ENTROPY = 'MAX_ENTROPY'
    PROBABILITY = 'PROBABILITY'
    NEGATIVE_PROB_LOSS = 'NEGATIVE_PROB_LOSS'
    NEGATIVE_LOSS = 'NEGATIVE_LOSS'  # NONE = 'NONE'


class EvalMetric(Enum):
    SUFFICIENCY = 'SUFFICIENCY'
    COMPREHENSIVENESS = 'COMPREHENSIVENESS'
    EVAL_LOG_ODDS = 'EVAL_LOG_ODDS'
    AOPC_SUFFICIENCY = 'AOPC_SUFFICIENCY'
    AOPC_COMPREHENSIVENESS = 'AOPC_COMPREHENSIVENESS'
    POS_AUC_WITH_REFERENCE_TOKEN = 'POS_AUC_WITH_REFERENCE_TOKEN'
    NEG_AUC_WITH_REFERENCE_TOKEN = 'NEG_AUC_WITH_REFERENCE_TOKEN'  # POS_AUC_NO_REFERENCE_TOKEN = 8  # NEG_AUC_NO_REFERENCE_TOKEN = 9


class DirectionTypes(Enum):
    MAX = 'MAX'
    MIN = 'MIN'


class TokenAttrSparseLossType(Enum):
    BCE = "BCE"
    L1 = "L1"


class RefTokenNameTypes(Enum):
    MASK = 'MASK'
    MASK_TEXT = 'MASK_TEXT'
    PAD = 'PAD'
    NEW_TOKEN = 'NEW_TOKEN'
    EOS = 'eos'
    UNK = 'unk'


class SchedulerTypes(Enum):
    LINEAR_SCHEDULE_WITH_WARMUP = 'LINEAR_SCHEDULE_WITH_WARMUP'
    COSINE_SCHEDULE_WITH_WARMUP = 'COSINE_SCHEDULE_WITH_WARMUP'
    COSINE_WITH_HARD_RESTARTS_SCHEDULE_WITH_WARMUP = 'COSINE_WITH_HARD_RESTARTS_SCHEDULE_WITH_WARMUP'
    CONSTANT_SCHEDULE_WITH_WARMUP = 'CONSTANT_SCHEDULE_WITH_WARMUP'


class AddLabelTokenToAttrGetType(Enum):
    FIRST_TOKEN = 'first_token'
    LAST_TOKEN = 'last_token'
    AFTER_LAST_SEP = 'after_last_sep'
    NONE = "none"


class ValidationType(Enum):
    VAL = 'VAL'
    TEST = 'TEST'


class ActivationFunctionTypes(Enum):
    TANH = 'tanh'
    RELU = 'relu'
    SIGMOID = 'sigmoid'
    NONE = 'none'


class ModelPromptType(Enum):
    ZERO_SHOT = 'zero_shot'
    FEW_SHOT = 'few_shot'
    FEW_SHOT_CONTENT = 'few_shot_content'  # few-shot but map just content


class CrossTokenizersPooling(Enum):
    MAX = 'max'
    MIN = 'min'
    MEAN = 'mean'
