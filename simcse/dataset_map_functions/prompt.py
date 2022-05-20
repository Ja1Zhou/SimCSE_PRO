def prompt(tokenizer, data_args,sent0_cname, sent1_cname, sent2_cname=None,):
    def prepare_features(examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the 
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.
        total = len(examples[sent0_cname])# batch size

        # Avoid "None" fields 
        for idx in range(total):
            if examples[sent0_cname][idx] is None:
                examples[sent0_cname][idx] = " "
            # examples[sent0_cname][idx] += "[MASK]"
            if examples[sent1_cname][idx] is None:
                examples[sent1_cname][idx] = " "
            # if sent1_cname != sent0_cname:
            #     examples[sent1_cname][idx] += "[MASK]"
        
        sentences = examples[sent0_cname] + examples[sent1_cname] # batch_size * num_sent

        # If hard negative exists
        if sent2_cname is not None:
            for idx in range(total):
                if examples[sent2_cname][idx] is None:
                    examples[sent2_cname][idx] = " "
                # if sent2_cname != sent0_cname and sent2_cname != sent1_cname:
                #     examples[sent2_cname][idx] += "[MASK]"
            sentences += examples[sent2_cname]

        sent_features = tokenizer(
            sentences,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )
        # add [MASK] id to the end of each sentence
        for ii in range(len(sentences)):
            tmp_end = sent_features["input_ids"][ii].pop()
            if len(sent_features["input_ids"][ii]) == data_args.max_seq_length-1:
                sent_features["input_ids"][ii].pop() 
                sent_features["attention_mask"][ii].pop()
                sent_features["token_type_ids"][ii].pop()
            sent_features['input_ids'][ii].append(tokenizer.mask_token_id)
            sent_features['input_ids'][ii].append(tmp_end)
            sent_features['attention_mask'][ii].append(1)
            sent_features['token_type_ids'][ii].append(0)
        features = {}
        if sent2_cname is not None:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
        else:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
            
        return features
    return prepare_features