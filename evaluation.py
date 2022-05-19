import sys, logging, torch, hydra
from prettytable import PrettyTable
from transformers import AutoModel, AutoTokenizer, TrainingArguments, AutoConfig
import importlib
from types import SimpleNamespace
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)
@hydra.main(config_path="configs", config_name="default_eval")
def main(cfg):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name_or_path", type=str, 
    #         help="Transformers' model name or path")
    # parser.add_argument("--pooler", type=str, 
    #         choices=['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'], 
    #         default='cls', 
    #         help="Which pooler to use")
    # parser.add_argument("--mode", type=str, 
    #         choices=['dev', 'test', 'fasttest'],
    #         default='test', 
    #         help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    # parser.add_argument("--task_set", type=str, 
    #         choices=['sts', 'transfer', 'full', 'na'],
    #         default='sts',
    #         help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    # parser.add_argument("--tasks", type=str, nargs='+', 
    #         default=['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    #                  'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
    #                  'SICKRelatedness', 'STSBenchmark'], 
    #         help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden")
    
    # args = parser.parse_args()
    model_args = SimpleNamespace(**cfg.model_args)
    data_args = SimpleNamespace(**cfg.data_args)
    trainer_args = SimpleNamespace(**cfg.trainer_args)
    eval_args = SimpleNamespace(**cfg.eval_args)
    training_args = TrainingArguments(**cfg.training_args)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    # Load transformers' model checkpoint
    model_class = getattr(importlib.import_module(f"..{model_args.model_class}", package="simcse.models.subpkg"), model_args.model_class)
    model_class_args = {}
    for k in model_args.model_class_args:
        model_class_args[k] = eval(k)
    model = model_class.from_pretrained(
        training_args.output_dir,
        **model_class_args,
    )
    tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Set up the tasks
    if eval_args.task_set == 'sts':
        eval_args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    elif eval_args.task_set == 'transfer':
        eval_args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif eval_args.task_set == 'full':
        eval_args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        eval_args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    # Set params for SentEval
    if eval_args.mode == 'dev' or eval_args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                         'tenacity': 3, 'epoch_size': 2}
    elif eval_args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return
    
    batcher_function_args = {}
    for k in trainer_args.batcher_function_args:
        batcher_function_args[k] = eval(k)
    get_batcher = getattr(importlib.import_module(f"..{trainer_args.batcher_function}", package = f"simcse.trainers.batcher_functions.subpkg"),trainer_args.batcher_function)
    batcher_function = get_batcher(**batcher_function_args)
    results = {}

    for task in eval_args.tasks:
        se = senteval.engine.SE(params, batcher_function, prepare)
        result = se.eval(task)
        results[task] = result
    
    # Print evaluation results
    if eval_args.mode == 'dev':
        print("------ %s ------" % (eval_args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

    elif eval_args.mode == 'test' or eval_args.mode == 'fasttest':
        print("------ %s ------" % (eval_args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)


if __name__ == "__main__":
    main()
