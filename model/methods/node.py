from model.methods.base import Method

class NodeMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)

    def construct_model(self, model_config = None):
        from model.models.node import NODE
        if model_config is None:
            model_config = self.args.config['model']
        self.model = NODE(
            d_in=self.d_in,
            d_out=self.d_out,
            **model_config
        ).to(self.args.device)
        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()