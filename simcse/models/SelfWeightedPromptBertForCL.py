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




class SelfWeightedPromptBertForCL(BertPreTrainedModel):
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
        self.get_mask = ZzjGetMask(config)
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
                self.get_mask,
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
                self.get_mask,
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
class ZzjGetMask(nn.Module):
    def __init__(self, config):
        super().__init__()
        #inputs should be [batch_size, seq_len, hidden_size]

        self.hidden_size = config.hidden_size
        self.output_size = 1 # a probability for each word
        self.layer = nn.Linear(self.hidden_size*2, self.output_size)
        self.activation = nn.Tanh()
        self.get_prob = nn.Softmax(dim=-1)
        # self.seq_len = config.max_position_embeddings
    def forward(self,inputs,attention_masks, sent_embed):
        # inputs should be [batch, num_sent, seq_len, 2*hidden_size]
        # inference inputs should be [batch, seq_len, 2*hidden_size]
        if sent_embed:
            batch, seq_len, _ = inputs.shape
        else:
            batch, num_sent, seq_len, _ = inputs.shape
        # inputs = inputs * attention_masks.unsqueeze(dim=-1).expand(-1, -1, -1, inputs.shape[-1])
        masks = self.layer(inputs)
        masks = self.activation(masks)
        # masks = self.get_prob(masks).view(-1,self.output_size)[:,0]
        masks = masks.view(batch, seq_len) if sent_embed else masks.view(batch,num_sent, seq_len)
        masks = masks - (attention_masks==0).clone().detach()*10000
        masks = self.get_prob(masks)# get probs along seq_len dimension
        return masks