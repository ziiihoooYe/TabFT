from model.methods.base import Method

class FTTMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)

    def construct_model(self, model_config=None):
        """
        Build the FTT Transformer model.

        The crucial part is determining `vector_sizes`, i.e. the length k_j of the
        (possibly‑vector) representation produced for every *numeric* feature by
        preceding transforms (e.g. CdfTransform or HomoTransform).

        These transforms populate `feature_map_` inside the *shared_state* dict of
        the `DataTransformPipeline`, so we query it **after** the pipeline has been
        fitted in `data_format()`.
        """
        from model.models.ftt import Transformer

        if model_config is None:
            model_config = self.args.config["model"]

        # -------------------------------------------------------------
        # 1) gather per‑feature vector sizes from the transform pipeline
        # -------------------------------------------------------------
        fmap = getattr(self, 'feature_map_', None)
        if fmap is not None:
            # store for later use / debugging
            self.feature_map_ = fmap
            vector_sizes = [m["size"] for m in fmap]
        else:
            # no feature expansion happened → every numeric feature is scalar
            vector_sizes = [1] * self.d_in

        # -------------------------------------------------------------
        # 2) instantiate Transformer with those sizes
        # -------------------------------------------------------------
        self.model = Transformer(
            vector_sizes=vector_sizes,
            categories=self.categories,
            d_out=self.d_out,
            **model_config,
        ).to(self.args.device)

        # keep original dtype handling
        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()

    