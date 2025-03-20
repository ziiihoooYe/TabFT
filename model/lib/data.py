import dataclasses as dc
import typing as ty
from copy import deepcopy
from pathlib import Path
import os
import json
import numpy as np
import sklearn.preprocessing
import torch
from sklearn.impute import SimpleImputer
from model.lib.TData import TData
from torch.utils.data import DataLoader
import torch.nn.functional as F
import category_encoders

BINCLASS = 'binclass'
MULTICLASS = 'multiclass'
REGRESSION = 'regression'
THIS_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(THIS_PATH, '..', '..', '..'))

ArrayDict = ty.Dict[str, np.ndarray]

def raise_unknown(unknown_what: str, unknown_value: ty.Any):
    raise ValueError(f'Unknown {unknown_what}: {unknown_value}')

def load_json(path):
    return json.loads(Path(path).read_text())


@dc.dataclass
class Dataset:
    N: ty.Optional[ArrayDict]
    C: ty.Optional[ArrayDict]
    y: ArrayDict
    info: ty.Dict[str, ty.Any]

    @property
    def is_binclass(self) -> bool:
        return self.info['task_type'] == BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.info['task_type'] == MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.info['task_type'] == REGRESSION

    @property
    def n_num_features(self) -> int:
        return self.info['n_num_features']

    @property
    def n_cat_features(self) -> int:
        return self.info['n_cat_features']

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def size(self, part: str) -> int:
        """
        Return the size of the dataset partition.

        Args:

        - part: str

        Returns: int
        """
        X = self.N if self.N is not None else self.C
        assert(X is not None)
        return len(X[part])


def dataname_to_numpy(dataset_name, dataset_path):

    """
    Load the dataset from the numpy files.

    :param dataset_name: str
    :param dataset_path: str
    :return: Tuple[ArrayDict, ArrayDict, ArrayDict, Dict[str, Any]]
    """
    dir_ = Path(os.path.join(DATA_PATH, dataset_path, dataset_name))

    def load(item) -> ArrayDict:
        return {
            x: ty.cast(np.ndarray, np.load(dir_ / f'{item}_{x}.npy', allow_pickle = True))  
            for x in ['train', 'val', 'test']
        }

    return (
        load('N') if dir_.joinpath('N_train.npy').exists() else None,
        load('C') if dir_.joinpath('C_train.npy').exists() else None,
        load('y'),
        load_json(dir_ / 'info.json'),
    )


def get_dataset(dataset_name, dataset_path):
    """
    Load the dataset from the numpy files.

    :param dataset_name: str
    :param dataset_path: str
    :return: Tuple[ArrayDict, ArrayDict, ArrayDict, Dict[str, Any]]
    """
    N, C, y, info = dataname_to_numpy(dataset_name, dataset_path)
    N_trainval = None if N is None else {key: N[key] for key in ["train", "val"]} if "train" in N and "val" in N else None
    N_test = None if N is None else {key: N[key] for key in ["test"]} if "test" in N else None

    C_trainval = None if C is None else {key: C[key] for key in ["train", "val"]} if "train" in C and "val" in C else None
    C_test = None if C is None else {key: C[key] for key in ["test"]} if "test" in C else None

    y_trainval = {key: y[key] for key in ["train", "val"]}
    y_test = {key: y[key] for key in ["test"]} 
    
    # tune hyper-parameters
    train_val_data = (N_trainval,C_trainval,y_trainval)
    test_data = (N_test,C_test,y_test)
    return train_val_data,test_data,info


def data_loader_process(is_regression, X, Y, y_info, device, batch_size, is_train,is_float = False):
    """
    Process the data loader.

    :param is_regression: bool
    :param X: Tuple[ArrayDict, ArrayDict]
    :param Y: ArrayDict
    :param y_info: Dict[str, Any]
    :param device: torch.device
    :param batch_size: int
    :param is_train: bool
    :return: Tuple[ArrayDict, ArrayDict, ArrayDict, DataLoader, DataLoader, Callable]
    """
    X = tuple(None if x is None else to_tensors(x) for x in X)
    Y = to_tensors(Y)

    X = tuple(None if x is None else {k: v.to(device) for k, v in x.items()} for x in X)
    Y = {k: v.to(device) for k, v in Y.items()}

    if X[0] is not None:
        if is_float:
            X = ({k: v.float() for k, v in X[0].items()}, X[1])
        else:
            X = ({k: v.double() for k, v in X[0].items()}, X[1])

    if is_regression:
        if is_float:
            Y = {k: v.float() for k, v in Y.items()}
        else:
            Y = {k: v.double() for k, v in Y.items()}
    else:
        Y = {k: v.long() for k, v in Y.items()}
    
    loss_fn = (
        F.mse_loss
        if is_regression
        else F.cross_entropy
    )

    if is_train:
        trainset = TData(is_regression, X, Y, y_info, 'train')
        valset = TData(is_regression, X, Y, y_info, 'val')
        train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)        
        val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=False, num_workers=0) 
        return X[0], X[1], Y, train_loader, val_loader, loss_fn
    else:
        testset = TData(is_regression, X, Y, y_info, 'test')
        test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=0)        
        return X[0], X[1], Y, test_loader, loss_fn
 

def to_tensors(data: ArrayDict) -> ty.Dict[str, torch.Tensor]:
    """
    Convert the numpy arrays to torch tensors.

    :param data: ArrayDict
    :return: Dict[str, torch.Tensor]
    """
    return {k: torch.as_tensor(v) for k, v in data.items()}


def get_categories(
    X_cat: ty.Optional[ty.Dict[str, torch.Tensor]]
) -> ty.Optional[ty.List[int]]:
    """
    Get the categories for each categorical feature.

    :param X_cat: Optional[Dict[str, torch.Tensor]]
    :return: Optional[List[int]]
    """
    return (
        None
        if X_cat is None
        else [
            len(set(X_cat['train'][:, i].tolist()))
            for i in range(X_cat['train'].shape[1])
        ]
    )
