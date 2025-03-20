from model.methods.base import Method

class BiSHopMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)


    def construct_model(self, model_config = None):
        from model.models.bishop import BiSHop
        if model_config is None:
            model_config = self.args.config['model']
        self.model = BiSHop(
            n_cat=len(self.categories)if self.categories is not None else 0,
            n_num=self.d_in,
            n_out=self.d_out,
            n_class=sum(self.categories) if self.categories is not None else 0,
            **model_config
        ).to(self.args.device)
        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()