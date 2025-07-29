# tabaug_method.py

import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
from copy import deepcopy

# 假设这些是你项目中的工具函数和基类
from model.methods.base import Method
from model.utils import Averager
# 导入你的数据变换类
from transform.enc_transform import UniPiecewiseCDFTransform 

class TabAugMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        # 从配置中读取增强参数
        self.augmentation_config = self.args.config.get('augmentation', {})
        self.n_bins_list = self.augmentation_config.get('n_bins_list', [1, 2, 4, 8, 16]) # 默认值为[1, 2, 4, 8, 16]
        self.augmentations = []

    def data_format(self, is_train=True, N=None, C=None, y=None):
        # 调用父类的 data_format 来处理 NaN、类别编码等
        super().data_format(is_train, N, C, y)

        # 如果是训练阶段，则创建并拟合所有的增强变换器
        if is_train:
            print(f"Initializing {len(self.n_bins_list)} augmentation transforms...")
            train_N_numpy = self.N['train'].cpu().numpy() if torch.is_tensor(self.N['train']) else self.N['train']
            
            for n_bins in self.n_bins_list:
                transform = UniPiecewiseCDFTransform(args={'n_bins': n_bins})
                # 用训练数据拟合变换器
                transform.fit(N_data={'train': train_N_numpy})
                self.augmentations.append(transform)

    def construct_model(self, model_config=None):
            from model.models.tabaug import TabAugModel # 确保你导入的是上面修改后的 TabAugModel
            if model_config is None:
                model_config = self.args.config['model']

            # --- 修改开始 ---
            # 准备模型初始化所需的详细特征信息
            d_num = self.N['train'].shape[1] if self.N and 'train' in self.N and self.N['train'] is not None else 0
            
            # self.categories 应该由父类 Method 在 data_format 中准备好
            # 它是一个列表，包含每个类别特征的基数
            cat_cardinalities = self.categories if self.C and self.categories else []

            self.model = TabAugModel(
                n_num_features=d_num,
                cat_cardinalities=cat_cardinalities,
                n_classes=self.d_out,
                **model_config
            ).to(self.args.device)
            # --- 修改结束 ---

            if self.args.use_float:
                self.model.float()
            else:
                self.model.double()
            
    def _apply_augmentations(self, X_num_batch, X_cat_batch):
        """对一个批次的数据应用所有增强变换"""
        if X_num_batch is None: # 如果没有数值特征
            n_aug = 1 # 只有一个视图（原始视图）
            X_cat_expanded = X_cat_batch.unsqueeze(1) # (B, D_cat) -> (B, 1, D_cat)
            return X_cat_expanded, n_aug

        n_aug = len(self.augmentations)
        device = X_num_batch.device
        
        # 1. 对数值特征应用所有增强
        augmented_X_num_list = []
        X_num_numpy = X_num_batch.cpu().numpy()
        for transform in self.augmentations:
            aug_N_dict, _, _ = transform.transform(N_data={'batch': X_num_numpy})
            augmented_X_num_list.append(torch.from_numpy(aug_N_dict['batch']).to(device))
        
        # 堆叠成 (n_aug, B, D_num), 然后变为 (B, n_aug, D_num)
        X_num_augmented = torch.stack(augmented_X_num_list, dim=0).permute(1, 0, 2)

        # 2. 准备类别特征
        if X_cat_batch is not None:
            # 将类别特征扩展以匹配增强后的数值特征
            X_cat_expanded = X_cat_batch.unsqueeze(1).expand(-1, n_aug, -1)
            # 拼接数值和类别特征
            X_full_augmented = torch.cat([X_num_augmented, X_cat_expanded], dim=-1)
        else:
            X_full_augmented = X_num_augmented
            
        return X_full_augmented, n_aug

    def train_epoch(self, epoch):
        self.model.train()
        tl = Averager()
        for i, (X, y) in enumerate(self.train_loader, 1):
            if self.N is not None and self.C is not None:
                X_num, X_cat = X[0], X[1]
            elif self.C is not None and self.N is None:
                X_num, X_cat = None, X
            else:
                X_num, X_cat = X, None

            # 应用数据增强
            X_augmented, n_aug = self._apply_augmentations(X_num, X_cat)
            y = y.to(self.args.device)

            pred = self.model(X_augmented) # 输出: (B, n_aug, k, D_out)
            
            # 准备损失计算
            y_pred_flat = pred.flatten(1, 2) # 展平 aug 和 k 维度 -> (B, n_aug * k, D_out)
            y_true_repeated = y.unsqueeze(1).expand(-1, y_pred_flat.shape[1]) # (B) -> (B, n_aug * k)

            loss = self.criterion(y_pred_flat.flatten(0, 1), y_true_repeated.flatten(0, 1))

            tl.add(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (i-1) % 50 == 0 or i == len(self.train_loader):
                print('epoch {}, train {}/{}, loss={:.4f} lr={:.4g}'.format(
                    epoch, i, len(self.train_loader), loss.item(), self.optimizer.param_groups[0]['lr']))
            del loss
        tl = tl.item()
        self.trlog['train_loss'].append(tl)    

    def validate(self, epoch):
        print('best epoch {}, best val res={:.4f}'.format(
            self.trlog['best_epoch'], self.trlog['best_res']))
        
        self.model.eval()
        val_logit, val_label = [], []
        with torch.no_grad():
            for i, (X, y) in tqdm(enumerate(self.val_loader)):
                if self.N is not None and self.C is not None:
                    X_num, X_cat = X[0], X[1]
                elif self.C is not None and self.N is None:
                    X_num, X_cat = None, X
                else:
                    X_num, X_cat = X, None                            

                # 应用数据增强
                X_augmented, n_aug = self._apply_augmentations(X_num, X_cat)
                
                pred = self.model(X_augmented) # 输出: (B, n_aug, k, D_out)
                
                # 在 aug 和 k 维度上取平均，得到最终预测
                pred = pred.mean(dim=(1, 2)) # -> (B, D_out)
                
                val_logit.append(pred)
                val_label.append(y)
                
        val_logit = torch.cat(val_logit, 0)
        val_label = torch.cat(val_label, 0)
        
        vl = self.criterion(val_logit, val_label).item()   

        # ... (后续的度量计算、模型保存逻辑与 TabM 相同)
        if self.is_regression:
            task_type = 'regression'
            measure = np.less_equal
        else:
            task_type = 'classification'
            measure = np.greater_equal

        vres, metric_name = self.metric(val_logit, val_label, self.y_info)

        print('epoch {}, val, loss={:.4f} {} result={:.4f}'.format(epoch, vl, task_type, vres[0]))
        if measure(vres[0], self.trlog['best_res']) or epoch == 0:
            self.trlog['best_res'] = vres[0]
            self.trlog['best_epoch'] = epoch
            torch.save(
                dict(params=self.model.state_dict()),
                osp.join(self.args.save_path, 'best-val-{}.pth'.format(str(self.args.seed)))
            )
            self.val_count = 0
        else:
            self.val_count += 1
            if self.val_count > 20:
                self.continue_training = False
        torch.save(self.trlog, osp.join(self.args.save_path, 'trlog'))
        
    def predict(self, data, info, model_name):
        # 预测逻辑与 validate 非常相似
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, model_name + '-{}.pth'.format(str(self.args.seed))))['params'])
        print('best epoch {}, best val res={:.4f}'.format(self.trlog['best_epoch'], self.trlog['best_res']))
        
        self.model.eval()
        self.data_format(False, data[0], data[1], data[2])
        
        test_logit, test_label = [], []
        with torch.no_grad():
            for i, (X, y) in tqdm(enumerate(self.test_loader)):
                if self.N is not None and self.C is not None:
                    X_num, X_cat = X[0], X[1]
                else: # ... (处理只有数值或类别特征的情况)
                    X_num, X_cat = (X, None) if self.N is not None else (None, X)

                X_augmented, _ = self._apply_augmentations(X_num, X_cat)
                pred = self.model(X_augmented).mean(dim=(1, 2))
                test_logit.append(pred)
                test_label.append(y)

        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)
        
        vl = self.criterion(test_logit, test_label).item()     
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)
        
        return vl, vres, metric_name, test_logit