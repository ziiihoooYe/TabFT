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
        self.dataset = getattr(args, 'dataset', None)
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
            elif transform_name == 'num_cdf':
                pipeline.append(CdfTransform(transform_config, self.dataset))
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
            elif transform_name == 'uniple':
                pipeline.append(UniPiecewiseCDFTransform(transform_config))
            elif transform_name == 'stretch_opt':
                pipeline.append(SlopeEqualizeStretchTransform(transform_config, self.is_regression))
            else:
                raise ValueError(f"Unknown transform name: {transform_name}")
        return pipeline

    def fit(self, N_data, C_data, y_data=None):
        """
        Calls fit(...) on each transform in sequence.
        """
        for transform_obj in self.pipeline:
            transform_obj.fit(N_data, C_data, y_data, self.shared_state)

        return self

    def transform(self, N_data, C_data, y_data=None):
        """
        Calls transform(...) on each transform in sequence, returning the final result.
        """
        for transform_obj in self.pipeline:
            N_data, C_data, y_data = transform_obj.transform(N_data, C_data, y_data, self.shared_state)

        return N_data, C_data, y_data

    def fit_transform(self, N_data, C_data, y_data=None):
        """
        Convenience method: fit on the data, then transform it in place.
        """
        # set seeds for reproducibility
        np.random.seed(0)
        torch.manual_seed(0)
        
        for transform_obj in self.pipeline:
            # 1) Fit on the current state of the data
            transform_obj.fit(N_data, C_data, y_data, self.shared_state)
            # 2) Transform the data in place
            N_data, C_data, y_data = transform_obj.transform(N_data, C_data, y_data, self.shared_state)
        
        return N_data, C_data, y_data
