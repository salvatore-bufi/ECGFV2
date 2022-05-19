import typing as t
import os
import pandas as pd
import numpy as np
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class SalvatoreAttributes(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.interactions_path = getattr(ns, "interactions", None)

        self.item_mapping = {}
        self.user_mapping = {}
        self.interactions_features_shape = None

        self.all_interactions = pd.read_csv(self.interactions_path, sep='\t')

        self.users = users
        self.items = items

    def get_mapped(self) -> t.Tuple[t.Set[int], t.Set[int]]:
        return self.users, self.items

    def filter(self, users: t.Set[int], items: t.Set[int]):
        self.users = self.users & users
        self.items = self.items & items

        self.all_interactions = self.all_interactions[self.all_interactions['user'].isin(self.users)]
        self.all_interactions = self.all_interactions[self.all_interactions['item'].isin(self.items)]

    def create_namespace(self) -> SimpleNamespace:
        ns = SimpleNamespace()
        ns.__name__ = "SalvatoreAttributes"
        ns.object = self
        ns.__dict__.update(self.__dict__)

        ns.user_mapping = self.user_mapping
        ns.item_mapping = self.item_mapping

        ns.interactions_features_shape = self.interactions_features_shape

        return ns

    def get_all_features(self):
        user_item_interactions = pd.read_csv(self.interactions_path, sep='\t', header=None)
        user_item_interactions = user_item_interactions.groupby(0).apply(
            lambda x: x.sort_values(by=[1], ascending=True)).reset_index(drop=True)
        interactions = len(user_item_interactions)
        user_item_features = np.empty((interactions, *self.interactions_features_shape))
        for i, row in user_item_interactions.iterrows():
            user_item_features[i] = np.load(self.interactions_feature_folder_path + '/' + str(row[2]) + '.npy')
        return user_item_features
