import os
import yaml
import shutil
import time
import errno
import pprint
import torch
import numpy as np
import random
import copy
import itertools
import json
import os.path as osp
from argparse import Namespace


THIS_PATH = os.path.dirname(__file__)
deep_model_list = ['mlp', 'resnet', 'ftt', 'node', 'autoint',
                   'tabpfn', 'tangos', 'saint', 'tabcaps', 'tabnet',
                   'snn', 'ptarl', 'danets', 'dcn2', 'tabtransformer',
                   'dnnr', 'switchtab', 'grownet', 'tabr', 'modernNCA',
                   'hyperfast', 'bishop', 'realmlp', 'protogate', 'mlp_plr',
                   'excelformer', 'grande','amformer','tabptm','trompt','tabm',
                   'PFN-v2', 't2gformer']
classical_model_list = ['LogReg', 'NCM', 'RandomForest', 
                        'xgboost', 'catboost', 'lightgbm',
                        'svm','knn', 'NaiveBayes',"dummy","LinearRegression"]


def mkdir(path):
    """
    Create a directory if it does not exist.

    :path: str, path to the directory
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def set_gpu(x):
    """
    Set environment variable CUDA_VISIBLE_DEVICES
    
    :x: str, GPU id
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    # print('using gpu:', x)


def ensure_path(path, remove=True):
    """
    Ensure a path exists.

    path: str, path to the directory
    remove: bool, whether to remove the directory if it exists
    """
    if os.path.exists(path):
        if remove:
            if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
                shutil.rmtree(path)
                os.mkdir(path)
    else:
        os.mkdir(path)


def merge_dicts(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for k, v in override.items():
        merged[k] = v
    return merged


#  --- criteria helper ---
class Averager():
    """
    A simple averager.

    """
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        """
        
        :x: float, value to be added
        """
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        """
        Measure the time since the last call to measure.

        :p: int, period of printing the time
        """

        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


#  ---- import from lib.util -----------
def set_seeds(base_seed: int, one_cuda_seed: bool = False) -> None:
    """
    Set random seeds for reproducibility.

    :base_seed: int, base seed
    :one_cuda_seed: bool, whether to set one seed for all GPUs
    """
    assert 0 <= base_seed < 2 ** 32 - 10000
    random.seed(base_seed)
    np.random.seed(base_seed + 1)
    torch.manual_seed(base_seed + 2)
    cuda_seed = base_seed + 3
    if one_cuda_seed:
        torch.cuda.manual_seed_all(cuda_seed)
    elif torch.cuda.is_available():
        # the following check should never succeed since torch.manual_seed also calls
        # torch.cuda.manual_seed_all() inside; but let's keep it just in case
        if not torch.cuda.is_initialized():
            torch.cuda.init()
        # Source: https://github.com/pytorch/pytorch/blob/2f68878a055d7f1064dded1afac05bb2cb11548f/torch/cuda/random.py#L109
        for i in range(torch.cuda.device_count()):
            default_generator = torch.cuda.default_generators[i]
            default_generator.manual_seed(cuda_seed + i)


def get_device() -> torch.device:
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


import sklearn.metrics as skm
def rmse(y, prediction, y_info):
    """
    
    :y: np.ndarray, ground truth
    :prediction: np.ndarray, prediction
    :y_info: dict, information about the target variable
    :return: float, root mean squared error
    """
    rmse = skm.mean_squared_error(y, prediction) ** 0.5  # type: ignore[code]
    if y_info['policy'] == 'mean_std':
        rmse *= y_info['std']
    return rmse
    
    
def load_config(args, config=None, config_name=None):
    """
    Load the config file.

    :args: argparse.Namespace, arguments
    :config: dict, config file
    :config_name: str, name of the config file
    :return: argparse.Namespace, arguments
    """
    if config is None:
        config_path = os.path.join(os.path.abspath(os.path.join(THIS_PATH, '..')), 
                                   'configs', args.dataset, 
                                   '{}.json'.format(args.model_type if args.config_name is None else args.config_name))
        with open(config_path, 'r') as fp:
            config = json.load(fp)

    # set additional parameters
    args.config = config 

    # save the config files
    with open(os.path.join(args.save_path, 
                           '{}.json'.format('config' if config_name is None else config_name)), 'w') as fp:
        args_dict = vars(args)
        if 'device' in args_dict:
            del args_dict['device']
        json.dump(args_dict, fp, sort_keys=True, indent=4)

    return args


# parameter search
def sample_parameters(trial, space, base_config):
    """
    Sample hyper-parameters.

    :trial: optuna.trial.Trial, trial
    :space: dict, search space
    :base_config: dict, base configuration
    :return: dict, sampled hyper-parameters
    """
    def get_distribution(distribution_name):
        return getattr(trial, f'suggest_{distribution_name}')

    result = {}
    for label, subspace in space.items():
        if isinstance(subspace, dict):
            result[label] = sample_parameters(trial, subspace, base_config)
        else:
            assert isinstance(subspace, list)
            distribution, *args = subspace

            if distribution.startswith('?'):
                default_value = args[0]
                result[label] = (
                    get_distribution(distribution.lstrip('?'))(label, *args[1:])
                    if trial.suggest_categorical(f'optional_{label}', [False, True])
                    else default_value
                )

            elif distribution == '$mlp_d_layers':
                min_n_layers, max_n_layers, d_min, d_max = args
                n_layers = trial.suggest_int('n_layers', min_n_layers, max_n_layers)
                suggest_dim = lambda name: trial.suggest_int(name, d_min, d_max)  # noqa
                d_first = [suggest_dim('d_first')] if n_layers else []
                d_middle = (
                    [suggest_dim('d_middle')] * (n_layers - 2) if n_layers > 2 else []
                )
                d_last = [suggest_dim('d_last')] if n_layers > 1 else []
                result[label] = d_first + d_middle + d_last

            elif distribution == '$d_token':
                assert len(args) == 2
                try:
                    n_heads = base_config['model']['n_heads']
                except KeyError:
                    n_heads = base_config['model']['n_latent_heads']

                for x in args:
                    assert x % n_heads == 0
                result[label] = trial.suggest_int('d_token', *args, n_heads)  # type: ignore[code]

            elif distribution in ['$d_ffn_factor', '$d_hidden_factor']:
                if base_config['model']['activation'].endswith('glu'):
                    args = (args[0] * 2 / 3, args[1] * 2 / 3)
                result[label] = trial.suggest_uniform('d_ffn_factor', *args)

            else:
                result[label] = get_distribution(distribution)(label, *args)
    return result


def merge_sampled_parameters(config, sampled_parameters):
    """
    Merge the sampled hyper-parameters.

    :config: dict, configuration
    :sampled_parameters: dict, sampled hyper-parameters
    """
    for k, v in sampled_parameters.items():
        if isinstance(v, dict):
            merge_sampled_parameters(config.setdefault(k, {}), v)
        else:
            # If there are parameters in the default config, the value of the parameter will be overwritten.
            config[k] = v


def fix_recipe(args, constraints):
    """
    If a constraint is violated, override the value in `args` with the first valid option
    from constraints.
    """
    model_type = args.model_type
    model_constraints = constraints.get(model_type, {})
    for key, allowed_values in model_constraints.items():
        current_value = getattr(args, key, None)
        if current_value not in allowed_values:
            # Override with the first valid choice, or any logic you prefer:
            setattr(args, key, allowed_values[0])


def fix_recipes(recipes, constraints):
    """
    For each recipe in (args, default_para, opt_space),
    fix the fields in `args` according to the model constraints.
    """
    for (args, default_para, opt_space) in recipes:
        fix_recipe(args, constraints)

        # If you also store certain fields in `args.config` (like normalization),
        # reflect that change there as well if needed:
        if hasattr(args, 'config') and 'training' in args.config:
            args.config['training']['normalization'] = args.normalization

    return recipes


def remove_duplicate_recipes(recipes):
    """
    Remove duplicates from a list of (args, default_para, opt_space) by comparing args.

    :param recipes: list of (args, default_para, opt_space)
    :return: a list with duplicates removed, preserving the original order of first occurrences
    """
    seen = set()
    unique = []
    for (args, default_para, opt_space) in recipes:
        args_dict = vars(args).copy()
        signature = json.dumps(args_dict, sort_keys=True)

        if signature not in seen:
            seen.add(signature)
            unique.append((args, default_para, opt_space))
    return unique


def expand_config(merged_config: dict, expand_keys: list):
    """
    Expand the merged_config by computing the Cartesian product of the specified keys in expand_keys.
    
    For example:
    merged_config = {
        "datasets": ["ds1", "ds2"],
        "batch_size": [32, 64],
        "model_type": "mlp",
        "lr_list": [1e-3, 1e-4]   # If you do not want to expand this key, do not include it in expand_keys.
    }
    expand_keys = ["datasets", "batch_size"]

    This will generate 2 x 2 = 4 config copies:
      1) {"datasets": "ds1", "batch_size": 32, "model_type": "mlp", "lr_list": [1e-3,1e-4]}
      2) {"datasets": "ds1", "batch_size": 64, ...}
      3) {"datasets": "ds2", "batch_size": 32, ...}
      4) {"datasets": "ds2", "batch_size": 64, ...}
    """
    result = []

    # Collect the values for each key in expand_keys; if a key's value is not a list, wrap it in a list.
    expansion_data = []
    for key in expand_keys:
        val = merged_config.get(key, None)
        if isinstance(val, list):
            expansion_data.append((key, val))
        else:
            expansion_data.append((key, [val]))
    
    # Compute the Cartesian product for the values of these keys.
    keys = [item[0] for item in expansion_data]
    vals_lists = [item[1] for item in expansion_data]
    for combo in itertools.product(*vals_lists):
        new_conf = merged_config.copy()  # shallow copy is sufficient here if nested objects are not modified
        for k, v in zip(keys, combo):
            new_conf[k] = v
        result.append(new_conf)

    return result
    

def load_recipes_from_yaml(yaml_file: str, expand_keys: list):
    """
    Read the YAML file, merge it with the corresponding default configuration (deep or classical),
    expand the configuration according to the specified keys, and construct a list of recipes.
    
    Each recipe is a tuple (args, default_conf, opt_space) where:
      - For deep models, default_conf is loaded from "configs/default/<model_type>.json" and
        opt_space from "configs/opt_space/<model_type>.json".
      - For classical models, default_conf is loaded from "configs/classical_configs.json" and
        opt_space is set to None.
    
    :param yaml_file: str, path to the YAML file.
    :param expand_keys: list, keys whose values should be expanded via Cartesian product.
    :param deep_model_list: list, model types considered as deep models.
    :param classical_model_list: list, model types considered as classical models.
    
    :return: list of tuples, each containing (args, default_conf, opt_space)
    """

    # --- Main processing ---
    # 1. Load the YAML configuration.
    with open(yaml_file, 'r', encoding='utf-8') as f:
        user_config = yaml.safe_load(f)

    # 2. Expand the user configuration using the provided keys.
    expanded_config_list = expand_config(user_config, expand_keys)

    recipes = []
    # 3. Process each expanded configuration.
    for conf in expanded_config_list:
        model_type = conf.get("model_type", "mlp")
        # Check which category the model belongs to.
        if model_type in deep_model_list:
            # --- Deep model branch ---
            # Merge with deep default configuration.
            with open("configs/deep_configs.json", 'r', encoding='utf-8') as f:
                deep_default_args = json.load(f)
            merged_conf = merge_dicts(deep_default_args, conf)
            # Load model-specific default parameters and optimization space.
            config_default_path = osp.join("configs", "default", f"{model_type}.json")
            config_opt_path = osp.join("configs", "opt_space", f"{model_type}.json")
            with open(config_default_path, 'r', encoding='utf-8') as fdef:
                default_conf = json.load(fdef)
            with open(config_opt_path, 'r', encoding='utf-8') as fopt:
                opt_space = json.load(fopt)
            if model_type not in default_conf:
                raise ValueError(f"[{model_type}] not found in {config_default_path}")
            merged_conf["config"] = default_conf[model_type]
            # Update additional training parameters.
            merged_conf["config"]["training"]["n_bins"] = merged_conf.get("n_bins", 2)
        elif model_type in classical_model_list:
            # --- Classical model branch ---
            with open("configs/classical_configs.json", 'r', encoding='utf-8') as f:
                classical_default_args = json.load(f)
            merged_conf = merge_dicts(classical_default_args, conf)
            # Load model-specific default parameters and optimization space.
            config_default_path = osp.join("configs", "default", f"{model_type}.json")
            config_opt_path = osp.join("configs", "opt_space", f"{model_type}.json")
            with open(config_default_path, 'r', encoding='utf-8') as fdef:
                default_conf = json.load(fdef)
            with open(config_opt_path, 'r', encoding='utf-8') as fopt:
                opt_space = json.load(fopt)
            if model_type not in default_conf:
                raise ValueError(f"[{model_type}] not found in {config_default_path}")
            merged_conf["config"] = default_conf[model_type]
            merged_conf["config"]["fit"]["n_bins"] = merged_conf.get("n_bins", 2)
        else:
            raise ValueError(f"Model type {model_type} is not recognized in either deep or classical model lists.")

        # Set GPU and prepare the save path.
        gpu = merged_conf.get("gpu", "0")
        set_gpu(gpu)
        dataset_name = merged_conf.get("dataset", "UnknownDataset")
        max_epoch = merged_conf.get("max_epoch", 200)
        batch_size = merged_conf.get("batch_size", 1024)
        tune = merged_conf.get("tune", False)
        # Construct the save path differently for deep and classical models.
        if model_type in deep_model_list:
            save_path1 = f"{dataset_name}-{model_type}"
            save_path2 = f"Epoch{max_epoch}BZ{batch_size}"
            if tune:
                save_path1 += "-Tune"
        else:  # classical model branch
            save_path1 = '-'.join([dataset_name, model_type])
            save_path2 = 'classical'
            if tune:
                save_path1 += "-Tune"
        model_path = merged_conf.get("model_path", "results_model")
        final_save_path = osp.join(model_path, save_path1, save_path2)
        merged_conf["save_path"] = final_save_path
        mkdir(final_save_path)

        # Convert merged configuration into a Namespace.
        args = Namespace(**merged_conf)
        # Append the tuple to recipes.
        recipe = (args, default_conf, opt_space)
        recipes.append(recipe)
    
    # 4. filter invalid recipes
    with open("configs/model_constraints.json", 'r', encoding='utf-8') as f:
        constrains = json.load(f)
    recipes = fix_recipes(recipes, constrains)
    recipes = remove_duplicate_recipes(recipes)

    return recipes


def show_results(
    args,
    info,
    metric_name,
    loss_list,
    results_list,
    time_list,
    logger=None,
    silent_detail=False
):
    """
    :param logger
    :param silent_detail: if True, do not print detailed results, otherwise print detailed results
    :return: (mean_metrics, std_metrics, metric_arrays, metric_name)
    """
    metric_arrays = {name: [] for name in metric_name}
    for result in results_list:
        for idx, name in enumerate(metric_name):
            metric_arrays[name].append(result[idx])
    metric_arrays['Time'] = time_list
    metric_name = metric_name + ('Time', )

    mean_metrics = {name: np.mean(metric_arrays[name]) for name in metric_name}
    std_metrics = {name: np.std(metric_arrays[name]) for name in metric_name}
    if loss_list[0] is not None:
        mean_loss = np.mean(np.array(loss_list))
    else:
        mean_loss = None

    lines = []
    if not silent_detail:
        lines.append(f'{args.model_type} Detailed Results (Dataset={args.dataset}):')
        for name in metric_name:
            if info['task_type'] == 'regression' and name != 'Time':
                formatted_results = ', '.join(['{:.8e}'.format(e) for e in metric_arrays[name]])
            else:
                formatted_results = ', '.join(['{:.8f}'.format(e) for e in metric_arrays[name]])
            lines.append(f'{name} per trial: {formatted_results}')

    lines.append(f'=== {args.model_type} (Dataset={args.dataset}) {args.seed_num} Trials Summary ===')
    for name in metric_name:
        if info['task_type'] == 'regression' and name != 'Time':
            lines.append(f'{name} MEAN = {mean_metrics[name]:.8e} ± {std_metrics[name]:.8e}')
        else:
            lines.append(f'{name} MEAN = {mean_metrics[name]:.8f} ± {std_metrics[name]:.8f}')

    if mean_loss is not None:
        lines.append(f'Mean Loss: {mean_loss:.8e}')

    lines.append('-' * 20 + ' GPU info ' + '-' * 20)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        lines.append(f"{num_gpus} GPU Available.")
        for i in range(num_gpus):
            gpu_info = torch.cuda.get_device_properties(i)
            lines.append(f"GPU {i}: {gpu_info.name}")
            lines.append(f"  Total Memory:          {gpu_info.total_memory / 1024**2} MB")
            lines.append(f"  Multi Processor Count: {gpu_info.multi_processor_count}")
            lines.append(f"  Compute Capability:    {gpu_info.major}.{gpu_info.minor}")
    else:
        lines.append("CUDA is unavailable.")
    lines.append('-' * 50)

    if logger:
        for line in lines:
            logger.info(line)
    else:
        for line in lines:
            print(line)

    return mean_metrics, std_metrics, metric_arrays, metric_name


def show_cross_dataset_results(all_results_summary, logger=None):
    import numpy as np
    model_aggregates = {}  # { model_name: { metric_name: [val1, val2, ...], ... }, ...}

    # 1. 收集每个模型在每个数据集的metrics
    for dataset_name, model_results in all_results_summary.items():
        for model_name, metrics_dict in model_results.items():
            if model_name not in model_aggregates:
                model_aggregates[model_name] = {}
            for met_name, val in metrics_dict.items():
                if met_name not in model_aggregates[model_name]:
                    model_aggregates[model_name][met_name] = []
                model_aggregates[model_name][met_name].append(val)

    # 2. 汇总并输出
    lines = ["\nCross-Dataset Averages for Each Model\n---------------------------------------"]
    for model_name, metrics_per_model in model_aggregates.items():
        lines.append(f"Model: {model_name}")
        for met_name, val_list in metrics_per_model.items():
            mean_val = np.mean(val_list)
            lines.append(f"  {met_name} Overall Mean: {mean_val:.4f}")
        lines.append("---------------------------------------")

    if logger:
        for line in lines:
            logger.info(line)
    else:
        for line in lines:
            print(line)


def tune_hyper_parameters(args,opt_space,train_val_data,info):
    """
    Tune hyper-parameters.

    :args: argparse.Namespace, arguments
    :opt_space: dict, search space
    :train_val_data: tuple, training and validation data
    :info: dict, information about the dataset
    :return: argparse.Namespace, arguments
    """
    import optuna
    import optuna.samplers
    import optuna.trial
    def objective(trial):
        config = {}
        try:
            opt_space[args.model_type]['training']['n_bins'] = [
                    "int",
                    2, 
                    256
            ]
        except:
            opt_space[args.model_type]['fit']['n_bins'] = [
                    "int",
                    2, 
                    256
            ]
        merge_sampled_parameters(
            config, sample_parameters(trial, opt_space[args.model_type], config)
        )    
        if args.model_type == 'xgboost' and torch.cuda.is_available():
            config['model']['tree_method'] = 'gpu_hist' 
            config['model']['gpu_id'] = args.gpu
            config['fit']["verbose"] = False
        elif args.model_type == 'catboost' and torch.cuda.is_available():
            config['fit']["logging_level"] = "Silent"
        
        elif args.model_type == 'RandomForest':
            config['model']['max_depth'] = 12
            
        if args.model_type in ['resnet']:
            config['model']['activation'] = 'relu'
            config['model']['normalization'] = 'batchnorm'    

        if args.model_type in ['ftt','t2gformer']:
            config['model'].setdefault('prenormalization', False)
            config['model'].setdefault('initialization', 'xavier')
            config['model'].setdefault('activation', 'reglu')
            config['model'].setdefault('n_heads', 8)
            config['model'].setdefault('d_token', 64)
            config['model'].setdefault('token_bias', True)
            config['model'].setdefault('kv_compression', None)
            config['model'].setdefault('kv_compression_sharing', None)    

        if args.model_type in ['excelformer']:
            config['model'].setdefault('prenormalization', False)
            config['model'].setdefault('kv_compression', None)
            config['model'].setdefault('kv_compression_sharing', None)  
            config['model'].setdefault('token_bias', True)
            config['model'].setdefault('init_scale', 0.01)
            config['model'].setdefault('n_heads', 8)

        if args.model_type in ["node"]:
            config["model"].setdefault("choice_function", "sparsemax")
            config["model"].setdefault("bin_function", "sparsemoid")

        if args.model_type in ['tabr']:
            config['model']["num_embeddings"].setdefault('type', 'PLREmbeddings')
            config['model']["num_embeddings"].setdefault('lite', True)
            config['model'].setdefault('d_multiplier', 2.0)
            config['model'].setdefault('mixer_normalization', 'auto')
            config['model'].setdefault('dropout1', 0.0)
            config['model'].setdefault('normalization', "LayerNorm")
            config['model'].setdefault('activation', "ReLU")
        
        if args.model_type in ['mlp_plr']:
            config['model']["num_embeddings"].setdefault('type', 'PLREmbeddings')
            config['model']["num_embeddings"].setdefault('lite', True)
            
        if args.model_type in ['ptarl']:
            config['model']['n_clusters'] = 20
            config['model']["regularize"]="True"
            config['general']["diversity"]="True"
            config['general']["ot_weight"]=0.25
            config['general']["diversity_weight"]=0.25
            config['general']["r_weight"]=0.25

        if args.model_type in ['modernNCA','tabm']:
            config['model']["num_embeddings"].setdefault('type', 'PLREmbeddings')
            config['model']["num_embeddings"].setdefault('lite', True)
        
        if args.model_type in ['tabm']:
            config['model']['backbone'].setdefault('type' , 'MLP')
            config['model'].setdefault("arch_type", "tabm")
            config['model'].setdefault("k", 32)
        
        
        if args.model_type in ['danets']:
            config['general']['k'] = 5
            config['general']['virtual_batch_size'] = 256
        
        if args.model_type in ['dcn2']:
            config['model']['stacked'] = False

        if args.model_type in ['grownet']:
            config["ensemble_model"]["lr"] = 1.0
            config['model']["sparse"] = False
            config["training"]['lr_scaler'] = 3

        if args.model_type in ['autoint']:
            config['model'].setdefault('prenormalization', False)
            config['model'].setdefault('initialization', 'xavier')
            config['model'].setdefault('activation', 'relu')
            config['model'].setdefault('n_heads', 8)
            config['model'].setdefault('d_token', 64)
            config['model'].setdefault('kv_compression', None)
            config['model'].setdefault('kv_compression_sharing', None)

        if args.model_type in ['protogate']:
            config['training'].setdefault('lam', 1e-3)
            config['training'].setdefault('pred_coef', 1)
            config['training'].setdefault('sorting_tau', 16)
            config['training'].setdefault('feature_selection', True)
            config['model'].setdefault('a',1)
            config['model'].setdefault('sigma',0.5)
        
        if args.model_type in ['grande']:
            config['model'].setdefault('from_logits', True)
            config['model'].setdefault('use_class_weights', True)
            config['model'].setdefault('bootstrap', False)

        if args.model_type in ['amformer']:
            config['model'].setdefault('heads', 8)
            config['model'].setdefault('groups', [54,54,54,54])
            config['model'].setdefault('sum_num_per_group', [32,16,8,4])
            config['model'].setdefault("prod_num_per_group", [6,6,6,6])
            config['model'].setdefault("cluster", True)
            config['model'].setdefault("target_mode", "mix")
            config['model'].setdefault("token_descent", False)

        if config.get('config_type') == 'trv4':
            if config['model']['activation'].endswith('glu'):
                # This adjustment is needed to keep the number of parameters roughly in the
                # same range as for non-glu activations
                config['model']['d_ffn_factor'] *= 2 / 3

        trial_configs.append(config)
        # method.fit(train_val_data, info, train=True, config=config)  
        # run with this config
        try:
            method.fit(train_val_data, info, train=True, config=config)    
            return method.trlog['best_res']
        except Exception as e:
            print(e)
            return 1e9 if info['task_type'] == 'regression' else 0.0
    
    if osp.exists(osp.join(args.save_path, '{}-tuned.json'.format(args.model_type))) and args.retune == False:
        with open(osp.join(args.save_path, '{}-tuned.json'.format(args.model_type)), 'rb') as fp:
            args.config = json.load(fp)
    else:
        # get data property
        if info['task_type'] == 'regression':
            direction = 'minimize'
            for key in opt_space[args.model_type]['model'].keys():
                if 'dropout' in key and '?' not in opt_space[args.model_type]['model'][key][0]:
                    opt_space[args.model_type]['model'][key][0] = '?'+ opt_space[args.model_type]['model'][key][0]
                    opt_space[args.model_type]['model'][key].insert(1, 0.0)
        else:
            direction = 'maximize'  
        
        method = get_method(args.model_type)(args, info['task_type'] == 'regression')      

        trial_configs = []
        study = optuna.create_study(
                direction=direction,
                sampler=optuna.samplers.TPESampler(seed=0),
            )        
        study.optimize(
            objective,
            **{'n_trials': args.n_trials},
            show_progress_bar=True,
        ) 
        # get best configs
        best_trial_id = study.best_trial.number
        # update config files        
        print('Best Hyper-Parameters')
        print(trial_configs[best_trial_id])
        args.config = trial_configs[best_trial_id]
        with open(osp.join(args.save_path, '{}-tuned.json'.format(args.model_type)), 'w') as fp:
            json.dump(args.config, fp, sort_keys=True, indent=4)
    return args


def get_method(_model):
    """
    Get the method class.

    :_model: str, model name
    :return: class, method class
    """
    if _model == "mlp":
        from model.methods.mlp import MLPMethod
        return MLPMethod
    elif _model == 'resnet':
        from model.methods.resnet import ResNetMethod
        return ResNetMethod
    elif _model == 'node':
        from model.methods.node import NodeMethod
        return NodeMethod
    elif _model == 'ftt':
        from model.methods.ftt import FTTMethod
        return FTTMethod
    elif _model == 'tabptm':
        from model.methods.tabptm import TabPTMMethod
        return TabPTMMethod
    elif _model == 'tabpfn':
        from model.methods.tabpfn import TabPFNMethod
        return TabPFNMethod
    elif _model == 'tabr':
        from model.methods.tabr import TabRMethod
        return TabRMethod
    elif _model == 'modernNCA':
        from model.methods.modernNCA import ModernNCAMethod
        return ModernNCAMethod
    elif _model == 'tabcaps':
        from model.methods.tabcaps import TabCapsMethod
        return TabCapsMethod
    elif _model == 'tabnet':
        from model.methods.tabnet import TabNetMethod
        return TabNetMethod
    elif _model == 'saint':
        from model.methods.saint import SaintMethod
        return SaintMethod
    elif _model == 'tangos':
        from model.methods.tangos import TangosMethod
        return TangosMethod    
    elif _model == 'snn':
        from model.methods.snn import SNNMethod
        return SNNMethod
    elif _model == 'ptarl':
        from model.methods.ptarl import PTARLMethod
        return PTARLMethod
    elif _model == 'danets':
        from model.methods.danets import DANetsMethod
        return DANetsMethod
    elif _model == 'dcn2':
        from model.methods.dcn2 import DCN2Method
        return DCN2Method
    elif _model == 'tabtransformer':
        from model.methods.tabtransformer import TabTransformerMethod
        return TabTransformerMethod
    elif _model == 'grownet':
        from model.methods.grownet import GrowNetMethod
        return GrowNetMethod
    elif _model == 'autoint':
        from model.methods.autoint import AutoIntMethod
        return AutoIntMethod
    elif _model == 'dnnr':
        from model.methods.dnnr import DNNRMethod
        return DNNRMethod
    elif _model == 'switchtab':
        from model.methods.switchtab import SwitchTabMethod
        return SwitchTabMethod
    elif _model == 'hyperfast':
        from model.methods.hyperfast import HyperFastMethod
        return HyperFastMethod
    elif _model == 'bishop':
        from model.methods.bishop import BiSHopMethod
        return BiSHopMethod
    elif _model == 'protogate':
        from model.methods.protogate import ProtoGateMethod
        return ProtoGateMethod
    elif _model == 'realmlp':
        from model.methods.realmlp import RealMLPMethod
        return RealMLPMethod
    elif _model == 'mlp_plr':
        from model.methods.mlp_plr import MLP_PLRMethod
        return MLP_PLRMethod
    elif _model == 'excelformer':
        from model.methods.excelformer import ExcelFormerMethod
        return ExcelFormerMethod
    elif _model == 'grande':
        from model.methods.grande import GRANDEMethod
        return GRANDEMethod
    elif _model == 'amformer':
        from model.methods.amformer import AMFormerMethod
        return AMFormerMethod
    elif _model == 'trompt':
        from model.methods.trompt import TromptMethod
        return TromptMethod
    elif _model == 'tabm':
        from model.methods.tabm import TabMMethod
        return TabMMethod
    elif _model == 'PFN-v2':
        from model.methods.PFN_v2 import TabPFNMethod
        return TabPFNMethod
    elif _model == 't2gformer':
        from model.methods.t2gformer import T2GFormerMethod
        return T2GFormerMethod
    elif _model == 'xgboost':
        from model.classical_methods.xgboost import XGBoostMethod
        return XGBoostMethod
    elif _model == 'LogReg':
        from model.classical_methods.logreg import LogRegMethod
        return LogRegMethod
    elif _model == 'NCM':
        from model.classical_methods.ncm import NCMMethod
        return NCMMethod
    elif _model == 'lightgbm':
        from model.classical_methods.lightgbm import LightGBMMethod
        return LightGBMMethod
    elif _model == 'NaiveBayes':
        from model.classical_methods.naivebayes import NaiveBayesMethod
        return NaiveBayesMethod
    elif _model == 'knn':
        from model.classical_methods.knn import KnnMethod
        return KnnMethod
    elif _model == 'RandomForest':
        from model.classical_methods.randomforest import RandomForestMethod
        return RandomForestMethod
    elif _model == 'catboost':
        from model.classical_methods.catboost import CatBoostMethod
        return CatBoostMethod
    elif _model == 'svm':
        from model.classical_methods.svm import SvmMethod
        return SvmMethod
    elif _model == 'dummy':
        from model.classical_methods.dummy import DummyMethod
        return DummyMethod
    elif _model == 'LinearRegression':
        from model.classical_methods.lr import LinearRegressionMethod
        return LinearRegressionMethod
    else:
        raise NotImplementedError("Model \"" + _model + "\" not yet implemented")
