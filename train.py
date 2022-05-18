import logging, os, hydra
from types import SimpleNamespace
from collections import namedtuple
import glob, transformers, importlib, math
from datasets import load_dataset, Dataset
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    BertForPreTraining,
    AdamW,
    get_linear_schedule_with_warmup
)
from transformers.trainer_utils import is_main_process
logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
@hydra.main(config_path="configs", config_name="default")
def main(cfg):
    model_args = SimpleNamespace(**cfg.model_args)
    data_args = SimpleNamespace(**cfg.data_args)
    trainer_args = SimpleNamespace(**cfg.trainer_args)
    training_args = TrainingArguments(**cfg.training_args)
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if data_args.train_file is not None:
        # data_files["train"] = data_args.train_file
        data_files["train"] = sorted(glob.glob(data_args.train_file))        
    extension = data_args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    if extension == "csv":
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/", delimiter="\t" if "tsv" in data_args.train_file else ",")
    else:
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/")

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # config_kwargs = {
    #     "cache_dir": model_args.cache_dir,
    #     "revision": model_args.model_revision,
    #     "use_auth_token": True if model_args.use_auth_token else None,
    # }
    # if model_args.config_name:
    #     config = AutoConfig.from_pretrained(model_args.config_name)
    if model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # tokenizer_kwargs = {
    #     "cache_dir": model_args.cache_dir,
    #     "use_fast": model_args.use_fast_tokenizer,
    #     "revision": model_args.model_revision,
    #     "use_auth_token": True if model_args.use_auth_token else None,
    # }
    # if model_args.tokenizer_name:
    #     tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    if model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    model_class = getattr(importlib.import_module(f"..{model_args.model_class}", package="simcse.models.subpkg"), model_args.model_class)
    model_class_args = {}
    for k in model_args.model_class_args:
        model_class_args[k] = eval(k)
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        **model_class_args,
    )
    model.resize_token_embeddings(len(tokenizer))
    if model_args.do_mlm:
        pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
        model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())

    # Prepare features
    column_names = datasets["train"].column_names
    datasets = datasets["train"]
    sent2_cname = None
    if len(column_names) == 2:
        # Pair datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
    elif len(column_names) == 3:
        # Pair datasets with hard negatives
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
        sent2_cname = column_names[2]
    elif len(column_names) == 1:
        # Unsupervised datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[0]
    else:
        raise NotImplementedError
    if data_args.force_unsup:
        new_dataset_dict = {sent0_cname:[]}
        for column in column_names:
            new_dataset_dict[sent0_cname] += datasets[column]
        datasets = Dataset.from_dict(new_dataset_dict)
        sent2_cname = None
        sent1_cname = sent0_cname
    get_map_function = getattr(importlib.import_module(f"..{data_args.dataset_map_function}", package="simcse.dataset_map_functions.subpkg"), data_args.dataset_map_function)
    dataset_map_function_args={}
    for k in data_args.dataset_map_function_args:
        dataset_map_function_args[k] = eval(k)
    prepare_features = get_map_function(**dataset_map_function_args)
    if training_args.do_train:
        train_dataset = datasets.map(
            prepare_features,
            batched=True,
            batch_size=data_args.batch_size,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=datasets.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    if data_args.collator != "default":
        get_collator = getattr(importlib.import_module(f"..{data_args.collator}", package="simcse.collators.subpkg"), data_args.collator)
        collator_args = {}
        for k in data_args.collator_args:
            collator_args[k] = eval(k)
        data_collator = get_collator(**collator_args)
    else: 
        data_collator = default_data_collator
    trainer_class = getattr(importlib.import_module(f"..{trainer_args.trainer_class}", package="simcse.trainers.subpkg"), trainer_args.trainer_class)
    if trainer_args.optimizer_args.custom_optimizer:
        optimizer_args = trainer_args.optimizer_args
        optimizer_class = eval(optimizer_args.optimizer_class)
        num_update_steps_per_epoch = train_dataset.num_rows // training_args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        max_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)
        optimizer_list = []
        for param_and_lr in optimizer_args.optimizer_class_args:
            tmp_dict = {}
            for k in param_and_lr:
                tmp_dict[k] = eval(str(param_and_lr[k]))
            optimizer_list.append(tmp_dict)
        optimizer = optimizer_class(optimizer_list)
        get_scheduler = eval(optimizer_args.scheduler)
        scheduler = get_scheduler(optimizer, optimizer_args.scheduler_warmup_steps, max_steps)
        optimizers = (optimizer, scheduler)
    train_dataset = train_dataset if training_args.do_train else None
    args = training_args # note that this is hard to interpret because of code style
    trainer_class_args = {}
    for k in trainer_args.trainer_class_args:
        trainer_class_args[k] = eval(k)
    trainer = trainer_class(**trainer_class_args)

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        train_result = trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate(eval_senteval_transfer=True)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
