from tqdm import tqdm
from model.utils import (load_recipes_from_yaml, tune_hyper_parameters,
                         set_seeds, get_method, show_results, show_cross_dataset_results)
import logging
from model.lib.data import get_dataset
import warnings
warnings.filterwarnings("ignore")
import sys
expand_keys = ['dataset', 'model_type']

def main():
    
    config_file = 'config.yaml'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='training_results.log',
        filemode='w'
    )
    logger = logging.getLogger(__name__)
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    recipes = load_recipes_from_yaml(config_file, expand_keys)
    all_results_summary = {}  # e.g., {dataset_name: {model_name: mean_metrics, ...}, ...}
    
    for recipe in recipes:
        args, default_para, opt_space = recipe
        loss_list, results_list, time_list = [], [], []
        
        train_val_data,test_data,info = get_dataset(args.dataset,args.dataset_path)
        if args.tune:
            args = tune_hyper_parameters(args,opt_space,train_val_data,info)
        ## Training Stage over different random seeds
        
        for seed in tqdm(range(args.seed_num)):
            args.seed = seed    # update seed  
            set_seeds(args.seed)
            method = get_method(args.model_type)(args, info['task_type'] == 'regression')
            time_cost = method.fit(train_val_data, info)    
            vl, vres, metric_name, predict_logits = method.predict(test_data, info, model_name=args.evaluate_option)

            loss_list.append(vl)
            results_list.append(vres)
            time_list.append(time_cost)

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



