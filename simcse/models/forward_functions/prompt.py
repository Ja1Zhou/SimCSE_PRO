import torch
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
import torch.distributed as dist
import torch.nn as nn
def train_forward_function(
    cls,
    encoder,
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
    mlm_input_ids=None,
    mlm_labels=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    # ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    batch_len = input_ids.size(-1)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    where_mask = torch.nonzero(input_ids==cls.mask_token_id,as_tuple=True)
    assert len(where_mask[1]) == batch_size * num_sent
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
    # input is shape like the following:
    # prefix-length + [X] + wrap-length + [MASK]
    # position ids for x should plus prefix length
    # position ids for x should plus prefix length and wrap length
    position_ids = torch.arange(input_ids.size(-1), dtype=torch.int, device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1).clone().detach()
    position_ids = attention_mask.clone().detach() * cls.prefix_length + position_ids
    position_ids[where_mask[0],where_mask[1]] = position_ids[where_mask[0],where_mask[1]] + cls.wrap_length
    position_ids[where_mask[0],where_mask[1]+1] = position_ids[where_mask[0],where_mask[1]+1] + cls.wrap_length
    # token_type_ids = torch.cat([token_type_ids, torch.zeros(token_type_ids.size(0),cls.prompt_length,dtype=token_type_ids.dtype,device=input_ids.device)],dim=-1)
    tmp_arange = cls.prompt_range.unsqueeze(0).expand(input_ids.size(0), -1).to(input_ids.device)
    past_key_values_1 = cls.prompt1(tmp_arange).view(input_ids.size(0),cls.n_layer*2,cls.n_head, cls.prompt_length,cls.n_embd)
    past_key_values_1 = cls.dropout(past_key_values_1)
    past_key_values_1 = past_key_values_1.transpose(0,1).split(2)
    past_key_values_2 = cls.prompt2(tmp_arange).view(input_ids.size(0),cls.n_layer*2,cls.n_head, cls.prompt_length,cls.n_embd) if not cls.model_args.self_cl else cls.prompt1(tmp_arange).view(input_ids.size(0),cls.n_layer*2,cls.n_head, cls.prompt_length,cls.n_embd)
    past_key_values_2 = cls.dropout(past_key_values_2)
    past_key_values_2 = past_key_values_2.transpose(0,1).split(2)

    # Get raw embeddings
    outputs_1 = encoder(
        input_ids,
        attention_mask = torch.cat([attention_mask, torch.ones(attention_mask.size(0),cls.prompt_length,dtype=attention_mask.dtype,device=input_ids.device)],dim=-1),
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
        past_key_values=past_key_values_1
    )
    outputs_2 = encoder(
        input_ids,
        attention_mask = torch.cat([attention_mask, torch.ones(attention_mask.size(0),cls.prompt_length,dtype=attention_mask.dtype,device=input_ids.device)],dim=-1),
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
        past_key_values=past_key_values_2
    )   
    # last_hidden -> batch_size * num_sent , seq_len, hidden_size
    pooler_output_1 = cls.pooler(attention_mask,outputs_1, where_mask)
    pooler_output_1 = pooler_output_1.view(batch_size, num_sent, pooler_output_1.size(-1))
    pooler_output_2 = cls.pooler(attention_mask,outputs_2, where_mask)
    pooler_output_2 = pooler_output_2.view(batch_size, num_sent, pooler_output_2.size(-1))
    # should be [ batch*num_sent, len, hidden_size]
    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )
    
    # Pooling
    # pooler_output = cls.pooler(attention_mask, outputs)
    # pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)
    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    # if cls.pooler_type == "cls":
    #     pooler_output = cls.mlp(pooler_output)
    attention_mask = attention_mask.view(batch_size,num_sent,batch_len)
    # Separate representation
    z1, z2 = pooler_output_1[:,0], pooler_output_1[:,1]
    z1_, z2_ = pooler_output_2[:,0], pooler_output_2[:,1]
    # Hard negative
    if num_sent == 3:
        z3 = pooler_output_1[:, 2]
        z3_ = pooler_output_2[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            z3__list = [torch.zeros_like(z3_) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            dist.all_gather(tensor_list=z3__list, tensor=z3_.contiguous())
            z3_list[dist.get_rank()] = z3
            z3__list[dist.get_rank()] = z3_
            z3 = torch.cat(z3_list, 0)
            z3_ = torch.cat(z3__list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z1__list = [torch.zeros_like(z2_) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        z2__list = [torch.zeros_like(z2_) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z1__list, tensor=z1_.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
        dist.all_gather(tensor_list=z2__list, tensor=z2_.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z1__list[dist.get_rank()] = z1_
        z2_list[dist.get_rank()] = z2
        z2__list[dist.get_rank()] = z2_
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z1_ = torch.cat(z1__list, 0)
        z2 = torch.cat(z2_list, 0)
        z2_ = torch.cat(z2__list, 0)

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    cos_sim_ = cls.sim(z1_.unsqueeze(1), z2_.unsqueeze(0))
    inter_cos = cls.sim(z1.unsqueeze(1), z1_.unsqueeze(0))
    # dis_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        z1_z3__cos = cls.sim(z1_.unsqueeze(1), z3_.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)
        cos_sim_ = torch.cat([cos_sim_, z1_z3__cos], 1)
        inter_z1_z3 = cls.sim(z1.unsqueeze(1), z3_.unsqueeze(0))
        inter_cos = torch.cat([inter_cos, inter_z1_z3], 1)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.model_args.hard_negative_weight
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device)
        cos_sim = cos_sim + weights
        cos_sim_ = cos_sim_ + weights
        inter_cos = inter_cos + weights

    loss = loss_fct(cos_sim, labels) + loss_fct(cos_sim_, labels) + loss_fct(inter_cos, labels)
    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs_1[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs_1.hidden_states,
        attentions=outputs_1.attentions,
    )   
def inference_forward_function(
    cls,
    encoder,
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
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    # ori_input_ids = input_ids
    # for inference, input should be [batch, len]
    batch_size = input_ids.size(0)
    # batch_len = input_ids.size(-1)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    where_mask = torch.nonzero(input_ids==cls.mask_token_id,as_tuple=True)
    assert len(where_mask[1]) == batch_size
    # attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    # if token_type_ids is not None:
    # token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
    # input is shape like the following:
    # prefix-length + [X] + wrap-length + [MASK]
    # position ids for x should plus prefix length
    # position ids for x should plus prefix length and wrap length
    position_ids = torch.arange(input_ids.size(-1), dtype=torch.int, device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1).clone().detach()
    position_ids = attention_mask.clone().detach() * cls.prefix_length + position_ids
    position_ids[where_mask[0],where_mask[1]] = position_ids[where_mask[0],where_mask[1]] + cls.wrap_length
    position_ids[where_mask[0],where_mask[1]+1] = position_ids[where_mask[0],where_mask[1]+1] + cls.wrap_length
    tmp_arange = cls.prompt_range.unsqueeze(0).expand(input_ids.size(0), -1).to(input_ids.device)
    past_key_values_1 = cls.prompt1(tmp_arange) if cls.model_args.inference_prompt == 1 else cls.prompt2(tmp_arange)
    past_key_values_1 = past_key_values_1.view(input_ids.size(0),cls.n_layer*2,cls.n_head, cls.prompt_length,cls.n_embd)
    # past_key_values_1 = cls.prompt2(tmp_arange).view(input_ids.size(0),cls.n_layer*2,cls.n_head, cls.prompt_length,cls.n_embd)
    # past_key_values_1 = cls.dropout(past_key_values_1)
    past_key_values_1 = past_key_values_1.transpose(0,1).split(2)

    # mlm_outputs = None
    # Flatten input for encoding
    # input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    # attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    # if token_type_ids is not None:
        # token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
    # Get static embeddings
    # static_embeddings =  encoder.embeddings.word_embeddings(input_ids) # (bs, len, dim)
    # static_embeddings = static_embeddings.view((batch_size, num_sent, -1, static_embeddings.size(-1))) # (bs, num_sent, len, dim)
    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=torch.cat([attention_mask, torch.ones(attention_mask.size(0),cls.prompt_length,dtype=attention_mask.dtype,device=input_ids.device)], dim=-1),
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args is None or (cls.model_args is not None and cls.model_args.pooler_type in ['avg_top2', 'avg_first_last']) else False,
        return_dict=True,
        past_key_values=past_key_values_1
    )
    # should be [ batch, len, hidden_size]
    # MLM auxiliary objective
    # if mlm_input_ids is not None:
    #     mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
    #     mlm_outputs = encoder(
    #         mlm_input_ids,
    #         attention_mask=attention_mask,
    #         token_type_ids=token_type_ids,
    #         position_ids=position_ids,
    #         head_mask=head_mask,
    #         inputs_embeds=inputs_embeds,
    #         output_attentions=output_attentions,
    #         output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
    #         return_dict=True,
    #     )
    
    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs, where_mask)
    pooler_output = pooler_output.view((batch_size, pooler_output.size(-1))) # (bs, num_sent, hidden)
    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    # if cls.pooler_type == "cls":
    #     pooler_output = cls.mlp(pooler_output)
    # attention_mask = attention_mask.view(batch_size,num_sent,batch_len)
    # Separate representation

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]
    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=past_key_values_1
    )
