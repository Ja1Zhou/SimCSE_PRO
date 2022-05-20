import torch
import torch.nn as nn
import importlib
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last", "mask"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs, where_mask=None):
        last_hidden = outputs.last_hidden_state # batch_size x seq_len x hidden_size
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "mask":
            return last_hidden[where_mask[0], where_mask[1],:]
        else:
            raise NotImplementedError

class PromptBertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.model_args = model_kargs["model_args"]
        self.prefix_length = self.model_args.prefix_length
        self.wrap_length = self.model_args.wrap_length
        self.prompt_length = self.model_args.prefix_length+self.model_args.wrap_length
        self.bert = BertModel(config, add_pooling_layer=False)
        self.mask_token_id = model_kargs["mask_token_id"]
        if self.model_args.freeze_backbone:
            # freeze bert parameters
            for param in self.bert.parameters():
                param.requires_grad = False
        self.prompt_range = torch.arange(self.prompt_length,dtype=torch.long)
        self.prompt1 = nn.Embedding(self.prompt_length, self.config.num_hidden_layers * 2 * self.config.hidden_size)
        self.prompt2 = nn.Embedding(self.prompt_length, self.config.num_hidden_layers * 2 * self.config.hidden_size)        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)
        self.pooler_type = self.model_args.pooler_type
        self.pooler = Pooler(self.model_args.pooler_type)
        if self.model_args.pooler_type == "cls":
            self.mlp = MLPLayer(config)
        self.sim = Similarity(temp=self.model_args.temp)
        init_function = getattr(importlib.import_module(f"..{self.model_args.init_function}", package="simcse.models.init_functions.subpkg"), self.model_args.init_function)
        init_function(self, config)
        self.train_forward_function = getattr(importlib.import_module(f"..{self.model_args.forward_function}", package="simcse.models.forward_functions.subpkg"), "train_forward_function")
        self.inference_forward_function = getattr(importlib.import_module(f"..{self.model_args.forward_function}", package="simcse.models.forward_functions.subpkg"), "inference_forward_function")
    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,        
    ):
        if sent_emb:
            return self.inference_forward_function(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return self.train_forward_function(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )