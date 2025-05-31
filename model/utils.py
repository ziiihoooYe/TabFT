from __future__ import annotations
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
import os.path as osp
from typing import Dict, Any, Tuple
import copy
import torch
import optuna


THIS_PATH = os.path.dirname(__file__)
deep_model_list = ['mlp', 'resnet', 'ftt', 'node', 'autoint',
                   'tabpfn', 'tangos', 'saint', 'tabcaps', 'tabnet',
                   'snn', 'ptarl', 'danets', 'dcn2', 'tabtransformer',
                   'dnnr', 'switchtab', 'grownet', 'tabr', 'modernNCA',
                   'hyperfast', 'bishop', 'realmlp', 'protogate', 'mlp_plr',
                   'excelformer', 'grande','amformer','tabptm','trompt','tabm',
                   'PFN-v2', 't2gformer', 'tab']
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
    if isinstance(x, int):
        x = str(x)
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
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = merge_dicts(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
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
                underlying = distribution.lstrip('?')
                enabled = trial.suggest_categorical(f'optional_{label}', [False, True])
                if not enabled:
                    result[label] = default_value
                    continue

                # enabled – perform real sampling
                if underlying == 'categorical':
                    choices = args[1:]
                    result[label] = trial.suggest_categorical(label, choices)
                else:
                    result[label] = get_distribution(underlying)(label, *args[1:])

            elif distribution == 'categorical':
                # `args` is the list of candidate choices
                result[label] = trial.suggest_categorical(label, args)

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


# --- update_transform_list helper ---
def update_transform_list(transform_list: list, updates: dict) -> list:
    """
    Return a **new** transform_list whose internal parameter dicts are
    updated according to *updates*.

    `updates` follows the same nesting structure as transform_list, e.g.:

        updates = {
            "num_cdf": {          # block name
                "cdf_type": "uniform",
                "n_components": 50,
            },
            "norm": {
                "policy": "minmax",
            },
        }
    """
    new_list = copy.deepcopy(transform_list)

    def deep_merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                deep_merge(dst[k], v)
            else:
                dst[k] = v

    for block in new_list:
        name = next(iter(block))          # each block is {"blk_name": {...}}
        if name in updates:
            deep_merge(block[name], updates[name])
    return new_list


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
        tune = merged_conf.get("tune", False) or merged_conf.get("load_tune_config", False)
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


def load_tuned_config(args, logger):
    tune_config_path = osp.join(args.save_path, f"{args.model_type}-tuned.json")
    
    if not osp.exists(tune_config_path):
        logger.info(f"Tuned config file {tune_config_path} does not exist. Skipping loading.")
        return args

    
    with open(tune_config_path, 'r') as f:
        tune_args = json.load(f)

    args.config = merge_dicts(args.config, tune_args)

    logger.info("Successfully merged tuned config into args.config")
    return args


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

    for dataset_name, model_results in all_results_summary.items():
        for model_name, metrics_dict in model_results.items():
            if model_name not in model_aggregates:
                model_aggregates[model_name] = {}
            for met_name, val in metrics_dict.items():
                if met_name not in model_aggregates[model_name]:
                    model_aggregates[model_name][met_name] = []
                model_aggregates[model_name][met_name].append(val)

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

# ============================ Hyperparameter Search ============================
def deep_update(dst: Dict[str, Any], src: Dict[str, Any], *, overwrite: bool = True) -> None:
    """Recursively merge *src* into *dst*.
    When *overwrite* is False, existing keys in *dst* are kept.
    """
    for k, v in src.items():
        if isinstance(v, dict):
            dst.setdefault(k, {})
            deep_update(dst[k], v, overwrite=overwrite)
        else:
            if overwrite or k not in dst:
                dst[k] = v


def apply_model_defaults(config: Dict[str, Any], model_type: str, *, gpu_id: int | None = None) -> None:
    """Fill *config* with static defaults and GPU‑dependent overrides."""
    # 1) static defaults
    default_json_path = osp.join("configs", "default", f"{model_type}.json")
    if osp.exists(default_json_path):
        with open(default_json_path, 'r') as fp:
            defaults = json.load(fp)
            defaults = defaults.get(model_type, {})
    else:
        defaults = {}

    deep_update(config, defaults, overwrite=False)

    # 2) dynamic defaults
    if model_type == "xgboost" and torch.cuda.is_available():
        deep_update(config, {"model": {"tree_method": "gpu_hist", "gpu_id": gpu_id or 0}}, overwrite=True)
    if model_type == "catboost" and torch.cuda.is_available():
        deep_update(config, {"fit": {"logging_level": "Silent"}}, overwrite=False)


from typing import Dict, Any, Tuple, Optional
def tune_hyper_parameters(
    args,
    opt_space: Dict[str, Any],
    train_val_data: Tuple[Any, Any, Any],
    info: Dict[str, Any]
):
    """Run Optuna search and return *args* with the best config injected.

    Parameters
    ----------
    args        : argparse.Namespace  – must contain model_type / save_path / n_trials / gpu
    opt_space   : dict               – search‑space definition (same schema as before)
    train_val_data : tuple           – (train, valid) data used by `method.fit`
    info        : dict               – dataset metadata; needs `task_type`
    """
    # 0) tweak search space (once, not inside objective)
    if info["task_type"] == "regression":
        direction = "minimize"
        for k, v in opt_space[args.model_type]["model"].items():
            if "dropout" in k and not v[0].startswith("?"):
                opt_space[args.model_type]["model"][k] = ["?" + v[0], 0.0] + v[1:]
    else:
        direction = "maximize"

    # 1) define objective
    def objective(trial: optuna.trial.Trial):
        # skeleton config
        config = {"transform": {},"model": {}, "fit": {}, "training": {}, "general": {}, "ensemble_model": {}}

        # (a) defaults
        apply_model_defaults(config, args.model_type, gpu_id=args.gpu)
        # (b) optuna‑sampled overrides
        local_space = copy.deepcopy(opt_space[args.model_type])

        # --- 1) sample everything (model + training + transform) ---
        sampled_all = sample_parameters(trial, local_space, config)

        # --- 2) peel off transform‑related hyper‑params -------------
        sampled_transform = sampled_all.pop("transform", None)

        # --- 3) merge the rest into `config` ------------------------
        merge_sampled_parameters(config, sampled_all)

        # --- 4) clone args and apply transform updates --------------
        trial_args = copy.deepcopy(args)
        if sampled_transform:
            trial_args.transform_list = update_transform_list(
                trial_args.transform_list, sampled_transform
            )

        method = get_method(trial_args.model_type)(
            trial_args, info["task_type"] == "regression"
        )
        try:
            method.fit(copy.deepcopy(train_val_data), info, train=True, config=config, tune=True)
            score = method.trlog["best_res"]
        except Exception as e:
            print("Trial failed:", e)
            score = float("inf") if direction == "minimize" else float("-inf")

        full_cfg = copy.deepcopy(config)
        if sampled_transform:
            full_cfg["transform"] = sampled_transform
        trial.set_user_attr("config", full_cfg)
        return score

    # 2) run Optuna
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best_config = study.best_trial.user_attrs["config"]
    args.config = best_config

    best_transform = best_config.pop("transform", None)
    if best_transform:
        args.transform_list = update_transform_list(args.transform_list, best_transform)

    # 3) persist
    os.makedirs(args.save_path, exist_ok=True)
    with open(osp.join(args.save_path, f"{args.model_type}-tuned.json"), "w") as fp:
        json.dump(best_config, fp, indent=4)

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
    elif _model == 'tab':
        from model.methods.Tab import TabMethod
        return TabMethod
    else:
        raise NotImplementedError("Model \"" + _model + "\" not yet implemented")

def get_logger(logger_name: str, log_file: str = None, level: int = __import__('logging').INFO):
    import logging
    import datetime
    if log_file is None:
        log_file = f"log/defalt_log.log"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # Clear any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
    # Create console handler for printing to the terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
