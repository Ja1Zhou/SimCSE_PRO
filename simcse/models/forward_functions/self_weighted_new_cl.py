import torch
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
import torch.distributed as dist
import torch.nn as nn
def new_cl_util(cls, num_sent, batch_size, hidden_size, z1, z2, z3=None):   
    if num_sent == 3:
        z2 = torch.cat([z2, z3], dim=0) # 2 * batch_size, hidden
    z2 = z2.unsqueeze(0).expand(batch_size, -1, -1) # batch_size, negative_number + 1, hidden
    z2 = z2.permute(2,0,1) # hidden * batch_size, negative_number + 1
    positive_mask = torch.eye(batch_size, device=z1.device, dtype=torch.bool)
    if num_sent == 3:
        mask_helper = torch.zeros(batch_size,batch_size, device=z1.device, dtype=torch.bool)
        positive_mask = torch.cat([positive_mask, mask_helper], dim=1) # batch_size, negative_number + 1
    anchor_vectors = z1.permute(1,0).unsqueeze(-1) # hidden * batch_size, 1
    positive_vectors = torch.masked_select(z2, positive_mask).view(hidden_size, batch_size, -1) # hidden * batch_size * 1
    negative_vectors = torch.masked_select(z2, ~positive_mask).view(hidden_size, batch_size, -1) # hidden * batch_size, negative_number
    positive_distance = torch.sqrt(torch.sum((anchor_vectors - positive_vectors)**2, dim=0)) # batch_size * 1
    negative_distance = torch.sqrt(torch.sum((anchor_vectors - negative_vectors)**2, dim=0)) # batch_size * negative_number
    if cls.loss_record is not None and cls.loss_record[-1] == 100:
        rescaler = torch.pow(negative_distance / (positive_distance + 1e-6),torch.mean(torch.exp(torch.tensor(cls.loss_record[0]))).item())
        rescaled_negative_vectors = (anchor_vectors + rescaler * (negative_vectors - anchor_vectors)).clone()
        # positive_distance < negative_distance -> batch_size * negative_number
        to_change_index = (positive_distance < negative_distance).unsqueeze(0).expand(hidden_size, -1, -1)
        negative_vectors[to_change_index] = rescaled_negative_vectors[to_change_index].to(negative_vectors.dtype)
    negative_vectors = negative_vectors.permute(1,2,0) # batch_size, negative_number, hidden
    positive_vectors = positive_vectors.permute(1,2,0) # batch_size, 1, hidden
    z2 = torch.cat([positive_vectors, negative_vectors], dim=1) # batch_size, negative_number + 1, hidden
    return cls.sim(z1.unsqueeze(1), z2) # batch_size, negative_number + 1
def train_forward_function(
    cls,
    encoder,
    masker,
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
    sent_emb=False
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
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
    # Get static embeddings
    static_embeddings =  encoder.embeddings.word_embeddings(input_ids) # (bs * num_sent, len, dim)
    static_embeddings = static_embeddings.view((batch_size, num_sent, -1, static_embeddings.size(-1))) # (bs, num_sent, len, dim)
    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )
    zzj_outputs = outputs.last_hidden_state
    hidden_size = zzj_outputs.size(-1)
    # should be [ batch*num_sent, len, hidden_size]
    zzj_outputs = zzj_outputs.view((batch_size, num_sent, -1, zzj_outputs.size(-1))) # (bs, num_sent, len, dim)
    all_embeddings = torch.cat([static_embeddings, zzj_outputs],dim = -1) # (bs, num_sent, len, 2*dim) 
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
    masked = masker(all_embeddings, attention_mask, sent_emb) # (bs, num_sent, len)
    if cls.model_args.add_entropy_loss:
        #calculate entropy of masked
        clone_mask = masked.clone()
        clone_mask[clone_mask != 0] = torch.log(clone_mask[clone_mask != 0])
        entropy = -torch.sum(masked * clone_mask, dim=-1)
        entropy = entropy.mean()
    zzj_outputs = zzj_outputs * masked.unsqueeze(-1).expand(-1,-1,-1,zzj_outputs.shape[-1]) # (bs, num_sent, len, dim)
    # Separate representation
    zzj_outputs = torch.sum(zzj_outputs, dim=-2) # (bs, num_sent, hidden_size)
    z1, z2 = zzj_outputs[:,0], zzj_outputs[:,1]

    # Hard negative
    z3 = zzj_outputs[:, 2] if num_sent == 3 else None
    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)
    cos_sim = new_cl_util(cls, num_sent, batch_size, hidden_size, z1, z2, z3)

    labels = torch.zeros(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives

    loss = loss_fct(cos_sim, labels)
    if cls.model_args.add_entropy_loss:
        loss += cls.model_args.entropy_loss_weight * entropy
    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )   
def inference_forward_function(
    cls,
    encoder,
    masker,
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
    sent_emb=True
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    # ori_input_ids = input_ids
    # for inference, input should be [batch, len]
    # batch_size = input_ids.size(0)
    # batch_len = input_ids.size(-1)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    # num_sent = input_ids.size(1)

    # mlm_outputs = None
    # Flatten input for encoding
    # input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    # attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    # if token_type_ids is not None:
        # token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
    # Get static embeddings
    static_embeddings =  encoder.embeddings.word_embeddings(input_ids) # (bs, len, dim)
    # static_embeddings = static_embeddings.view((batch_size, num_sent, -1, static_embeddings.size(-1))) # (bs, num_sent, len, dim)
    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args is None or (cls.model_args is not None and cls.model_args.pooler_type in ['avg_top2', 'avg_first_last']) else False,
        return_dict=True,
    )
    zzj_outputs = outputs.last_hidden_state
    # should be [ batch, len, hidden_size]
    # zzj_outputs = zzj_outputs.view((batch_size, num_sent, -1, zzj_outputs.size(-1))) # (bs, num_sent, len, dim)
    all_embeddings = torch.cat([static_embeddings, zzj_outputs],dim = -1) # (bs, len, 2*dim) 
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
    # pooler_output = cls.pooler(attention_mask, outputs)
    # pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)
    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    # if cls.pooler_type == "cls":
    #     pooler_output = cls.mlp(pooler_output)
    # attention_mask = attention_mask.view(batch_size,num_sent,batch_len)
    masked = masker(all_embeddings, attention_mask, sent_emb) # (bs, len)
    zzj_outputs = zzj_outputs * masked.unsqueeze(-1).expand(-1,-1,zzj_outputs.shape[-1]) # (bs, len, dim)
    # Separate representation
    zzj_outputs = torch.sum(zzj_outputs, dim=-2) # (bs, hidden_size)

    if not return_dict:
        return (outputs[0], zzj_outputs) + outputs[2:]
    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=zzj_outputs,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=masked
    )