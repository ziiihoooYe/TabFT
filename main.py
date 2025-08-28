import json
import os
import os.path as osp
from tqdm import tqdm
import copy
from model.utils import (load_recipes_from_yaml, tune_hyper_parameters, get_logger,
                         set_seeds, get_method, show_results, show_cross_dataset_results,
                         load_tuned_config)
import logging
from model.lib.data import get_dataset
from transform.transform_pipeline import DataTransformPipeline
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import sys
import datetime
expand_keys = ['dataset', 'model_type']

def main():
    
    ### ----------- Load Config/Prepare Logger ------------
    parser = argparse.ArgumentParser(description="Run model training and evaluation with specified configurations.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration YAML file.')
    parser.add_argument('--log', type=str, default=None, help='Path to the log file. If not provided, logs will be printed to console.')
    parser.add_argument('--pre_transform', type=str, default=None)
    args = parser.parse_args() 
    force_transform = args.pre_transform is not None
    config_file = args.config
    logger = get_logger(__name__, args.log)

    recipes = load_recipes_from_yaml(config_file, expand_keys)
    all_results_summary = {}  # e.g., {dataset_name: {model_name: mean_metrics, ...}, ...}
    
    for recipe in recipes:

        try: 
            args, default_para, opt_space = recipe
            set_seeds(args.seed)
            loss_list, results_list, time_list = [], [], []
            logger.info("------------------------------------")
            logger.info(f"Dataset: {args.dataset}")
            logger.info(f"Model: {args.model_type}")
            logger.info(f"transform_list: {args.transform_list}")
            
            # get dataset
            train_val_data,test_data,info = get_dataset(args.dataset,args.dataset_path)

            # ---------- Pre‑process the dataset once ----------
            need_pretransform = not (getattr(args, "tune_transform", False) and args.tune) # only if tuning transform, do not pre-transform
            pipeline = None
            pre_transformed = False

            if need_pretransform:
                pipeline = DataTransformPipeline(args.transform_list, args, info['task_type']=='regression')
                N_trainval, C_trainval, y_trainval = pipeline.fit_transform(*train_val_data)
                train_val_data = (N_trainval, C_trainval, y_trainval)

                N_test, C_test, y_test = pipeline.transform(*test_data)
                test_data = (N_test, C_test, y_test)
                pre_transformed = True
            
            ### ----------- Tuning Hyperparameters ------------
            is_loaded = False
            if args.load_tune_config:
                args, is_loaded = load_tuned_config(args, logger)
            if args.tune and not is_loaded:
                args = tune_hyper_parameters(args,opt_space,train_val_data,info,pipeline,pre_transformed)
            
            # if not pre_transformed, we still need to transform the data here
            if not pre_transformed and not force_transform:
                pipeline = DataTransformPipeline(args.transform_list, args, info['task_type']=='regression')
                N_trainval, C_trainval, y_trainval = pipeline.fit_transform(*train_val_data)
                train_val_data = (N_trainval, C_trainval, y_trainval)

                N_test, C_test, y_test = pipeline.transform(*test_data)
                test_data = (N_test, C_test, y_test)
                pre_transformed = True


            ### ----------- Training and Testing ------------
            for seed in tqdm(range(args.seed, args.seed + args.seed_num)): 

                args.seed = seed    # update seed  
                set_seeds(args.seed)

                method = get_method(args.model_type)(args, info['task_type'] == 'regression')
                if pre_transformed:
                    method.data_transform_pipeline = copy.deepcopy(pipeline)
                    method.pre_transformed = pre_transformed

                time_cost = method.fit(train_val_data, info)    
                vl, vres, metric_name, predict_logits = method.predict(test_data, info, model_name=args.evaluate_option)

                loss_list.append(vl)
                results_list.append(vres)
                time_list.append(time_cost)

            ### ----------- Show Results ------------
            mean_metrics, std_metrics, metric_arrays, m_names = show_results(
                args,
                info,
                metric_name,
                loss_list,
                results_list,
                time_list,
                logger=logger,
                silent_detail=True
            )

            if args.dataset not in all_results_summary:
                all_results_summary[args.dataset] = {}
            all_results_summary[args.dataset][args.model_type] = mean_metrics
        except Exception as e:
            logger.error(f"Error processing dataset {args.dataset} with model {args.model_type}: {e}")
            continue

    logger.info("\nFinal Summary of All Results\n----------------------------")
    for dataset_name, model_results in all_results_summary.items():
        logger.info(f"Dataset: {dataset_name}")
        for model_name, means in model_results.items():
            logger.info(f"  Model: {model_name}")
            for m_name, m_val in means.items():
                logger.info(f"    {m_name} Mean: {m_val:.4f}")
    logger.info("----------------------------")
    show_cross_dataset_results(all_results_summary, logger=logger)

if __name__ == '__main__':
    main()
