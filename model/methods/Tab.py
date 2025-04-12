from model.methods.base import Method
import torch

class TabMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)

    def construct_model(self, model_config = None):
        if model_config is None:
            model_config = self.args.config['model']
        from model.models.Tab import Tab
        self.model = Tab(
            num_continuous=self.n_num_features,
            categories=self.categories, 
            d_model=model_config.get('d_model', 256), 
            n_head=model_config.get('n_head', 8), 
            d_out=self.d_out, 
            num_enc_layers=model_config.get('num_enc_layers', 6), 
            is_regression=self.is_regression
            ).to(self.args.device)
        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()