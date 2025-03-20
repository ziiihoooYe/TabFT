

class BaseTransform:
    """
    Base class for all data transformations.
    Each transform class should implement `fit`, `transform`, optionally `fit_transform`.
    """

    def fit(self, N_data, C_data, y_data=None):
        """
        Learn any necessary parameters from data.
        N_data: numeric features (dict of arrays or a single array)
        C_data: categorical features
        y_data: labels (may be needed for some transforms, e.g. target encoding)
        Return self.
        """
        raise NotImplementedError

    def transform(self, N_data, C_data, y_data=None, shared_state=None):
        """
        Apply the transform to data, returning the transformed data (and possibly labels).
        """
        raise NotImplementedError

    def fit_transform(self, N_data, C_data, y_data=None, shared_state=None):
        """
        Convenient method: calls fit() then transform().
        """
        self.fit(N_data, C_data, y_data)
        return self.transform(N_data, C_data, y_data)