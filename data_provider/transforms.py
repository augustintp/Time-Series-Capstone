import abc
import numpy as np
import numpy.ma as ma
import pandas as pd
import sys
import torch


class BaseScaler(abc.ABC):

    @abc.abstractmethod
    def fit(self, data, missing=None):
        """Performs feature transformation."""
        raise NotImplementedError()

    @abc.abstractmethod
    def transform(self, data):
        """Performs feature transformation."""
        raise NotImplementedError()

    @abc.abstractmethod
    def scale_back(self, data):
        """Performs feature transformation."""
        raise NotImplementedError()


class IdentityScaler(BaseScaler):

    def __init__(self, data, instance_id=None, given_steps=None, missing=None):
        self.fit(data, instance_id=instance_id)

    def fit(self, data, instance_id=None):
        self.scale = 1

    def transform(self, data, instance_id=None, time_col=None):
        return data / self.scale

    def scale_back(self, data, instance_id=None, time_col=None):
        return data * self.scale

    def get_weights(self):
        return [self.scale]


class StandardScaler(BaseScaler):

    def __init__(self, data, instance_id=None, given_steps=None, missing=None):
        self.missing = missing
        self.need_id = instance_id is not None
        self.need_time = given_steps is not None
        self.mean_std = {}
        if not self.need_time:
            self.fit(data, instance_id=instance_id)
        else:
            self.given_steps = given_steps

    def fit(self, data, instance_id=None):
        assert not self.need_time, '[StandardScaler] Normalizing by windows... Time column is required!'
        assert self.need_id or instance_id is None, '[StandardScaler] Cannot have both in fit.'
        if self.missing is not None:
            mask = data != self.missing
            data = data[mask]
            if instance_id is not None:
                instance_id = instance_id[mask]
        if instance_id is not None:
            assert self.need_id, '[StandardScaler] Not normalizing by ID, but instance_id is provided.'
            combined = pd.concat([data, instance_id], axis=1)
            self.mean = combined.groupby(instance_id.name)[data.name].mean().to_dict()
            self.std = (combined.groupby(instance_id.name)[data.name].std() + 1e-8).to_dict()
        else:
            self.need_id = False
            self.mean = np.mean(data)
            self.std = np.std(data) + 1e-8

    def update(self, data, instance_id, time_col, return_scale=False):
        data_copy = ma.masked_values(np.copy(data[:, :self.given_steps]), self.missing)
        mean = np.mean(data_copy, axis=1).data
        """ numpy.ma
        We need to stress that this behavior may not be systematic, that masked data may be affected by the operation 
        in some cases and therefore users should not rely on this data remaining unchanged.
        """
        data_copy = ma.masked_values(np.copy(data[:, :self.given_steps]), self.missing)
        std = np.std(data_copy, axis=1).data
        if self.need_time:
            self.mean_std.update({(a, b): [c, d] for a, b, c, d in zip(instance_id, time_col, mean, std)})
        if return_scale:
            return mean

    def transform(self, data, instance_id=None, time_col=None):
        assert instance_id is not None or self.need_id is False, '[StandardScaler] IDs are required!'
        assert time_col is not None or self.need_time is False, '[StandardScaler] time_col is required!'
        if time_col is not None:
            all_missing_mask = data == self.missing
            mean_std = np.array([self.mean_std[(a, b)] for a, b in zip(instance_id, time_col)])
            data = (data - mean_std[:, 1, np.newaxis]) / mean_std[:, 1, np.newaxis]
            data[all_missing_mask] = self.missing
            return data
        elif instance_id is not None:
            mean = instance_id.map(self.mean)
            std = instance_id.map(self.std)
            return (data - mean) / std
        else:
            return (data - self.mean) / self.std

    def scale_back(self, data, instance_id=None, time_col=None, kill_missing=False):
        assert instance_id is not None or self.need_id is False, '[StandardScaler] IDs are required!'
        assert time_col is not None or self.need_time is False, '[StandardScaler] time_col is required!'
        all_missing_mask = data == self.missing
        if time_col is not None:
            mean_std = torch.tensor([self.mean_std[(a, b)] for a, b in zip(instance_id, time_col)],
                                    device=data.device)
            if len(data.shape) == 2:
                data = data * mean_std[:, 1, None] + mean_std[:, 0, None]
            else:
                data = data * mean_std[:, 1, None, None] + mean_std[:, 0, None, None]
        elif instance_id is not None:
            mean = instance_id.map(self.mean)
            std = instance_id.map(self.std)
            data = data * std + mean
        else:
            data = data * self.std + self.mean

        if kill_missing:
            data[all_missing_mask] = self.missing
        return data

    def get_weights(self):
        return [self.mean, self.std]


class DeepARScaler(BaseScaler):

    def __init__(self, data, instance_id=None, given_steps=None, missing=None, month_id=None):
        self.missing = missing
        self.need_id = instance_id is not None
        self.need_time = given_steps is not None
        self.need_month = month_id is not None
        self.v = {}
        if not self.need_time:
            self.fit(data, instance_id=instance_id, month_id=month_id)
        else:
            self.given_steps = given_steps

    def fit(self, data, instance_id=None, month_id=None):
        assert not self.need_time, '[DeepARScaler] Normalizing by windows... Time column is required!'
        assert self.need_id or instance_id is None, '[DeepARScaler] Cannot have both in fit.'
        if self.missing is not None:
            mask = data != self.missing
            data = data[mask]
            if instance_id is not None:
                instance_id = instance_id[mask]
        if instance_id is not None and month_id is None:
            assert self.need_id, '[DeepARScaler] Not normalizing by ID, but instance_id is provided.'
            combined = pd.concat([data, instance_id], axis=1)
            self.v = (combined.groupby(instance_id.name)[data.name].mean() + 1 + 1e-8).to_dict()
        elif instance_id is not None and month_id is not None:
            assert self.need_month, '[DeepARScaler] Not normalizing by month, but month_id is provided.'
            combined = pd.concat([data, instance_id, month_id], axis=1)
            self.v = (combined.groupby([month_id.name, instance_id.name])[
                          data.name].mean() + 1 + 1e-8).unstack().to_dict()
        else:
            self.v = np.mean(data) + 1 + 1e-8

    def update(self, data, instance_id, time_col, return_scale=False):
        missing_mask = data[:, :self.given_steps] != self.missing
        data[:, :self.given_steps][~missing_mask] = 0
        data_sum = data[:, :self.given_steps].sum(axis=1)
        missing_sum = missing_mask.sum(axis=1)
        missing_sum[missing_sum == 0] = 1
        v = np.true_divide(data_sum, missing_sum) + 1
        if self.need_time:
            self.v.update({(a, b): c for a, b, c in zip(instance_id, time_col, v)})
        if return_scale:
            return v

    def transform(self, data, instance_id=None, time_col=None, month_id=None):
        assert instance_id is not None or self.need_id is False, '[DeepARScaler] IDs are required!'
        assert time_col is not None or self.need_time is False, '[DeepARScaler] time_col is required!'
        if time_col is not None:
            all_missing_mask = data == self.missing
            v = np.array([self.v[(a, b)] for a, b in zip(instance_id, time_col)])
            data = data / v[:, np.newaxis]
            data[all_missing_mask] = self.missing
            return data
        elif instance_id is not None and month_id is None:
            assert self.need_id, '[DeepARScaler] Not normalizing by ID, but instance_id is provided.'
            v = self.v[instance_id]
            return data / v
        elif instance_id is not None and month_id is not None:
            assert self.need_month, '[DeepARScaler] Not normalizing by month, but month_id is provided.'
            new_data = data.copy()
            for month in range(len(month_id)):
                new_data[month] = data[month] / self.v[instance_id][month_id[month]]
            return new_data
        else:
            return data / self.v

    def scale_back(self, data, instance_id=None, time_col=None, kill_missing=False, month_id=None):
        assert instance_id is not None or self.need_id is False, '[DeepARScaler] IDs are required!'
        assert time_col is not None or self.need_time is False, '[DeepARScaler] time_col is required!'
        all_missing_mask = data == self.missing
        if time_col is not None:
            v = torch.tensor([self.v[(a, b)] for a, b in zip(instance_id, time_col)], device=data.device)
            if len(data.shape) == 2:
                data = data * v[:, None]
            else:
                data = data * v[:, None, None]
        elif instance_id is not None and month_id is None:
            v = self.v[instance_id]
            data = data * v
        elif instance_id is not None and month_id is not None:
            new_data = data.copy()
            for month in range(len(month_id)):
                new_data[month] = data[month] * self.v[instance_id][month_id[month]]
            return new_data
        else:
            data = data * self.v

        if kill_missing:
            data[all_missing_mask] = self.missing
        return data

    def get_weights(self):
        return [self.v]


class OneHotScaler(BaseScaler):

    def __init__(self, data):
        self.fit(data)

    def fit(self, data, missing=None, instance_id=None):
        unique_entities = data.unique()
        self.name_to_key = {}
        for i, source in enumerate(unique_entities):
            self.name_to_key[source] = i
        self.inv_trans = {v: k for k, v in self.name_to_key.items()}
        self._num_class = len(unique_entities)

    def transform(self, data, instance_id=None):
        mapped_data = data.map(self.name_to_key)
        assert not mapped_data.isnull().any(), f'Column {data.name} has categories unseen during training!'
        return mapped_data

    def scale_back(self, data, instance_id=None):
        return data.map(self.inv_trans)

    def scale_back_one_element(self, data):
        return self.inv_trans[data]

    def get_num_class(self):
        return self._num_class

    def get_weights(self):
        return [self.name_to_key, self.inv_trans]
