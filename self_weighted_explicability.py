import logging, torch, hydra, os
from prettytable import PrettyTable
from transformers import AutoTokenizer, TrainingArguments, AutoConfig
import importlib
from types import SimpleNamespace
import matplotlib.pyplot as plt
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    logger.info("\n"+str(tb))
@hydra.main(config_path="configs", config_name="default_explicability")
def main(cfg):
    model_args = SimpleNamespace(**cfg.model_args)
    # data_args = SimpleNamespace(**cfg.data_args)
    # trainer_args = SimpleNamespace(**cfg.trainer_args)
    explicability_args = SimpleNamespace(**cfg.explicability_args)
    training_args = TrainingArguments(**cfg.training_args)
    if not os.path.exists(explicability_args.output_dir):
        os.makedirs(explicability_args.output_dir)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir)
    # Load transformers' model checkpoint
    model_class = getattr(importlib.import_module(f"..{model_args.model_class}", package="simcse.models.subpkg"), model_args.model_class)
    model_class_args = {}
    mask_token_id = tokenizer.mask_token_id
    for k in model_args.model_class_args:
        model_class_args[k] = eval(k)
    model = model_class.from_pretrained(
        training_args.output_dir,
        **model_class_args,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    input_sentences = explicability_args.input_sentences
    encoded_input_sentences = tokenizer(input_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    output_hidden_states = True
    return_dict = True
    sent_emb = True
    model_forward_args = {}
    for k in model_args.model_forward_args:
        model_forward_args[k] = eval(k)
    encoded_input_sentences.update(model_forward_args)
    with torch.no_grad():
        outputs = model(**encoded_input_sentences)
        for i in len(encoded_input_sentences):
            plt.bar(tokenizer.convert_ids_to_tokens(encoded_input_sentences["input_ids"][i]),outputs.hidden_states[0].detach().numpy())
            plt.ylabel("MLP output weights")
            plt.xticks(rotation=90)
            plt.show()
            plt.savefig(f"{training_args.output_dir}/mlp_weights_{i}.png")

if __name__ == "__main__":
    main()
