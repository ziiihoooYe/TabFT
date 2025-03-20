from model.methods.base import Method

class AMFormerMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)

    def construct_model(self, model_config = None):
        from model.models.amformer import AMFormer
        if model_config is None:
            model_config = self.args.config['model']
        self.model = AMFormer(
                num_cont=self.d_in,
                num_cate=len(self.categories) if self.categories is not None else 0,
                categories=self.categories,
                out=self.d_out,
                **model_config
                ).to(self.args.device) 
        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()