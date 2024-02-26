from typing import Optional, List

import torch
import torch.utils.checkpoint
from torch import nn, Tensor
from transformers.models.distilbert.modeling_distilbert import (DistilBertPreTrainedModel, PretrainedConfig,
                                                                DistilBertModel)

from models.tokens_attr_class.tokens_attr_utils import post_model_pre_classifier, get_token_classifier


class DistilBertAttrGeneration(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.token_classifier = get_token_classifier(config.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None, output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None,
                 labels_new_indices: List[int] = None) -> Tensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask,
                                  inputs_embeds=inputs_embeds, output_attentions=output_attentions,
                                  output_hidden_states=output_hidden_states, return_dict=return_dict, )

        sequence_output = outputs[0]
        sequence_output = post_model_pre_classifier(sequence_output, labels_new_indices)

        sequence_output = self.dropout(sequence_output)
        logits = self.token_classifier(sequence_output)

        tokens_attr = logits.squeeze(-1)  # [batch, seq]
        tokens_attr = torch.sigmoid(tokens_attr)

        return tokens_attr