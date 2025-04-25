from transform.nan_transform import NanTransform
from transform.label_transform import LabelTransform
from transform.norm_transform import NormalizationTransform
from transform.enc_transform import *
import os
import json
from pathlib import Path
import numpy as np
from copy import deepcopy

class DataTransformPipeline:
    def __init__(self, transform_list, args, is_regression=False):
        """
        :param transform_names: list of str, e.g. ['nan', 'cat_nan_new', 'num_enc_Q_PLE', ...]
        :param args: your argparse or config object
        """
        self.is_preprocessed = getattr(args, 'is_preprocessed', False)
        self.is_regression = is_regression
        self.transform_list = transform_list
        self.args = args
        self.pipeline = self._build_transforms()
        self.shared_state = {}
        # cache last transformed data for optional persistence
        self._cached_N = None
        self._cached_C = None
        self._cached_y = None

    def _build_transforms(self):
        """
        Internal helper that maps each name in self.transform_names to the actual transform object.
        You can expand with more transforms as needed.
        """
        pipeline = []
        for t in self.transform_list:
            transform_name = list(t.keys())[0]
            transform_config = t[transform_name]
            if transform_name == 'nan':
                pipeline.append(NanTransform(transform_config))
            elif transform_name == 'label':
                pipeline.append(LabelTransform(self.is_regression))
            elif transform_name == 'norm':
                pipeline.append(NormalizationTransform(transform_config, self.args.seed))
            elif transform_name == 'num_binning':
                pipeline.append(BinningTransform(transform_config, self.is_regression))
            elif transform_name == 'num_ple':
                 pipeline.append(PLETransform(transform_config))
            elif transform_name == 'num_unary':
                pipeline.append(UnaryTransform(transform_config))
            elif transform_name == 'num_bin':
                pipeline.append(BinsTransform(transform_config))
            elif transform_name == 'num_binindex':
                pipeline.append(BinIndexTransform(transform_config))
            elif transform_name == 'num_johnson':
                pipeline.append(JohnsonTransform(transform_config))
            elif transform_name == 'num_quantiletransform':
                pipeline.append(QuantileTransform(transform_config))
            elif transform_name == 'num_robustscale':
                pipeline.append(RobustScaleTransform(transform_config))
            elif transform_name == 'num_smoothclip':
                pipeline.append(SmoothClipTransform(transform_config))
            elif transform_name == 'num_dpgmmcdf':
                pipeline.append(DPGMMCdfTransform(transform_config))
            elif transform_name == 'num_clustercdf':
                pipeline.append(ClusterCdfTransform(transform_config))
            elif transform_name == 'cat_ordinal':
                pipeline.append(OrdinalTransform(transform_config))
            elif transform_name == 'cat_indice':
                pipeline.append(IndiceTransform(transform_config))
            elif transform_name == 'cat_onehot':
                pipeline.append(OneHotTransform(transform_config))
            elif transform_name == 'cat_binary':
                pipeline.append(BinaryTransform(transform_config))
            elif transform_name == 'cat_hash':
                pipeline.append(HashTransform(transform_config))
            elif transform_name == 'cat_loo':
                pipeline.append(LeaveOneOutTransform(transform_config))
            elif transform_name == 'cat_target':
                pipeline.append(TargetTransform(transform_config))
            elif transform_name == 'cat_catboost':
                pipeline.append(CatBoostTransform(transform_config))
            elif transform_name == 'cat_targetindice':
                pipeline.append(TargetRankingIndiceTransform(transform_config))
            elif transform_name == 'cat_qt':
                pipeline.append(CatQuantileTransform(transform_config))
            else:
                raise ValueError(f"Unknown transform name: {transform_name}")
        return pipeline

    def fit(self, N_data, C_data, y_data=None):
        """
        Calls fit(...) on each transform in sequence.
        """
        if self.is_preprocessed:
            # If preprocessed, we don't fit the transforms
            return self

        for transform_obj in self.pipeline:
            transform_obj.fit(N_data, C_data, y_data, self.shared_state)

        return self

    def transform(self, N_data, C_data, y_data=None):
        """
        Calls transform(...) on each transform in sequence, returning the final result.
        """
        if self.is_preprocessed:
            # If preprocessed, we don't transform the data
            return N_data, C_data, y_data

        for transform_obj in self.pipeline:
            N_data, C_data, y_data = transform_obj.transform(N_data, C_data, y_data, self.shared_state)
        # keep a copy so we can optionally dump the transformed dataset later
        self._cached_N, self._cached_C, self._cached_y = N_data, C_data, y_data
        return N_data, C_data, y_data

    def fit_transform(self, N_data, C_data, y_data=None):
        """
        Convenience method: fit on the data, then transform it in place.
        """
        if self.is_preprocessed:
            # If preprocessed, we don't transform the data
            return N_data, C_data, y_data

        for transform_obj in self.pipeline:
            # 1) Fit on the current state of the data
            transform_obj.fit(N_data, C_data, y_data, self.shared_state)
            # 2) Transform the data in place
            N_data, C_data, y_data = transform_obj.transform(N_data, C_data, y_data, self.shared_state)
        
        # keep a copy so we can optionally dump the transformed dataset later
        self._cached_N, self._cached_C, self._cached_y = N_data, C_data, y_data
        return N_data, C_data, y_data

    def save_transformed_dataset(
        self,
        dataset_dir: str | Path,
        dataset_suffix: str = "",
        data_root: str | Path = "data"  # default root for datasets
    ):
        """
        Save the most‑recently transformed dataset to disk so that it can be
        re‑loaded with ``get_dataset`` without re‑running the pipeline.

        Parameters
        ----------
        save_dir : str | Path
            Directory where the new dataset folder will be created.
        original_info : dict, optional
            Existing ``info.json`` dictionary.  If supplied, it will be copied
            and its feature counts updated before being written out.
        dataset_suffix : str, optional
            Extra text appended to the folder name so multiple variants can coexist.
        """
        original_dir = dataset_dir
        save_dir = f'{data_root}/{original_dir}_{dataset_suffix}'
        
        import json
        with open(f'{data_root}/{original_dir}/info.json', 'r') as f:
            original_info = json.load(f)
        if self._cached_N is None and self._cached_C is None and self._cached_y is None:
            raise RuntimeError(
                "No transformed data are cached.  Run `transform` or "
                "`fit_transform` first."
            )

        save_dir = Path(save_dir).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

        def _dump(arr_dict, prefix):
            if arr_dict is None:
                return
            for part, arr in arr_dict.items():
                np.save(save_dir / f"{prefix}_{part}.npy", arr, allow_pickle=False)

        _dump(self._cached_N, "N")
        _dump(self._cached_C, "C")
        _dump(self._cached_y, "y")
        if 'test' in self._cached_y:
            return self
        # ------------- feature‑map aggregation -------------
        feature_maps = {}
        for t in self.pipeline:
            fmap = getattr(t, "feature_map_", None)
            if fmap:
                feature_maps[type(t).__name__] = fmap
        if feature_maps:
            (save_dir / "feature_map.json").write_text(json.dumps(feature_maps, indent=2))
 
        # -------------- Gaussian‑param aggregation --------------
        gmm_params = {}
        for t in self.pipeline:
            # DPGMMCdf* transforms
            if hasattr(t, "active_components_info_") and getattr(t, "active_components_info_", None):
                params = []
                for info_i in t.active_components_info_:
                    if info_i is None:
                        params.append(None)
                    else:
                        param_dict = {
                            "means": info_i["means"].tolist(),
                            "stds":  info_i["stds"].tolist()
                        }
                        if "weights" in info_i:
                            param_dict["weights"] = info_i["weights"].tolist()
                        params.append(param_dict)
                gmm_params[type(t).__name__] = params
            # HybridCdfTransform
            elif hasattr(t, "gmm_info_") and getattr(t, "gmm_info_", None):
                params = []
                for info_i in t.gmm_info_:
                    if info_i is None:
                        params.append(None)
                    else:
                        param_dict = {
                            "means": info_i["means"].tolist(),
                            "stds":  info_i["stds"].tolist()
                        }
                        if "weights" in info_i:
                            param_dict["weights"] = info_i["weights"].tolist()
                        params.append(param_dict)
                gmm_params[type(t).__name__] = params
 
        if gmm_params:
            (save_dir / "gmm_params.json").write_text(json.dumps(gmm_params, indent=2))

        # build and dump info.json
        if original_info is None:
            info = {
                "task_type": "unknown",
                "n_num_features": 0 if self._cached_N is None else self._cached_N["train"].shape[1],
                "n_cat_features": 0 if self._cached_C is None else self._cached_C["train"].shape[1],
            }
        else:
            info = deepcopy(original_info)
            info["n_num_features"] = 0 if self._cached_N is None else self._cached_N["train"].shape[1]
            info["n_cat_features"] = 0 if self._cached_C is None else self._cached_C["train"].shape[1]
            info["transformed_by"] = [type(t).__name__ for t in self.pipeline]

        # record auxiliary files
        if feature_maps:
            info["feature_map_file"] = "feature_map.json"
        if gmm_params:
            info["gmm_param_file"] = "gmm_params.json"
        (save_dir / "info.json").write_text(json.dumps(info, indent=2))
        
        return self