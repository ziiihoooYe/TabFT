import sys
import numpy as np
from tqdm import tqdm
from model.utils import (load_recipes_from_yaml, tune_hyper_parameters, get_logger,
                         set_seeds, get_method, show_results, show_cross_dataset_results)
from transform.transform_pipeline import DataTransformPipeline
from model.lib.data import get_dataset
import warnings
import os
import json
from pathlib import Path
def get_gmmparam(dataset_name, dataset_path):
    param_dir = Path(os.path.join(dataset_path, dataset_name, 'gmm_params.json'))
    feature_map_dir = Path(os.path.join(dataset_path, dataset_name, 'feature_map.json'))
    
    params = json.load(param_dir.open('r', encoding='utf-8')) if param_dir.exists() else {}
    feature_map = json.load(feature_map_dir.open('r', encoding='utf-8')) if feature_map_dir.exists() else {}
 
    return params['DPGMMCdfTransform'] if 'DPGMMCdfTransform' in params else None, feature_map['DPGMMCdfTransform'] if 'DPGMMCdfTransform' in feature_map else None
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def GMM_filter(params, feature_map, N, sample_num):
    weight_threshold = 1/sample_num
    # weight_threshold = 0.0001

    # Filter out GMM components with low weight
    valid_idxs_all = []

    for i, feature_info in enumerate(params):
        start_idx = feature_map[i]['new_start']
        assert 'weights' in feature_info, "Each feature_info should contain 'weights' key"
        
        means = feature_info['means']
        stds = feature_info['stds']
        weights = feature_info['weights']
        # find components above threshold
        valid_idxs = [i for i, w in enumerate(weights) if w >= weight_threshold]
        
        valid_idxs_all.extend([start_idx + idx for idx in valid_idxs])
    
    average_components = int(len(valid_idxs_all) / len(params)) if len(params) > 0 else 0
    print(f"average number of valid GMM components: {average_components}")
    return N[:, valid_idxs_all], average_components
    

config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
suffix = sys.argv[2] if len(sys.argv) > 2 else 'new'
is_gmm = True if len(sys.argv) > 3 else False

recipes = load_recipes_from_yaml(config_file, ['dataset', 'model_type'])
transformed_datasets = []

avg_components = []
for dataset in tqdm(range(len(recipes))):
    args, _, _ = recipes[dataset]
    train_val_data,test_data,info = get_dataset(args.dataset,args.dataset_path)
    if is_gmm:
        gmm_params, feature_map = get_gmmparam(args.dataset, args.dataset_path)

    train_val_N, train_val_C, train_val_y = train_val_data
    test_N, test_C, test_y = test_data
    args.is_regression = (info['task_type'] == 'regression')
    args.is_preprocessed = is_gmm
    orginal_features = train_val_N['train'].shape[1] if train_val_N is not None else None
    if is_gmm and (gmm_params is not None or feature_map is not None):
        for part in train_val_N:
            train_val_N[part], _ = GMM_filter(gmm_params, feature_map, train_val_N[part], sample_num=train_val_N['train'].shape[0])
        for part in test_N:
            test_N[part], components = GMM_filter(gmm_params, feature_map, test_N[part], sample_num=train_val_N['train'].shape[0])
        avg_components.append(components)
    
    pipeline = DataTransformPipeline(args.transform_list, args, is_regression=args.is_regression)

    if is_gmm:
        pipeline._cached_N, pipeline._cached_C, pipeline._cached_y = train_val_N, train_val_C, train_val_y
    else:
        pipeline.fit_transform(train_val_N, train_val_C, train_val_y)
        if orginal_features is not None:
            avg_components.append(pipeline._cached_N['train'].shape[1] / orginal_features)
    pipeline.save_transformed_dataset(args.dataset, suffix)

    if is_gmm:
        pipeline._cached_N, pipeline._cached_C, pipeline._cached_y = test_N, test_C, test_y
    else:
        pipeline.transform(test_N, test_C, test_y)
    pipeline.save_transformed_dataset(args.dataset, suffix)
    
    transformed_datasets.append(args.dataset)
    print(f"Transformed dataset {args.dataset} saved with suffix '{suffix}'.")

print(f"Average number of valid components across datasets: {np.mean(avg_components) if avg_components else 0}")

