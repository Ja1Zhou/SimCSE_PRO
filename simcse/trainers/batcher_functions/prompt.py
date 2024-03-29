import torch
def prompt(tokenizer, model_args, training_args, model):    
    def batcher(params, batch):
        for s in batch:
            s.append(tokenizer.mask_token)
        sentences = [' '.join(s) for s in batch]
        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
        )
        assert model.max_position_embeddings >= len(batch['input_ids'][0])

        for k in batch:
            batch[k] = batch[k].to(training_args.device)
        with torch.no_grad():
            output_hidden_states=True
            return_dict=True
            sent_emb=True
            model_forward_args = {}
            for k in model_args.model_forward_args:
                model_forward_args[k] = eval(k)
            batch.update(model_forward_args)
            outputs = model(**batch)
            pooler_output = outputs.pooler_output
        return pooler_output.cpu()
    return batcher