import settings
from pandas.core.frame import DataFrame
import re
from torch.utils import data
import pandas as pd
import math
import random
import torch
from torch.utils.data.sampler import Sampler
import random
from itertools import chain
import numpy as np

class Dataset(data.Dataset):
    """ A class implementing :class:`torch.utils.data.Dataset`.

    Dataset subclasses the abstract class :class:`torch.utils.data.Dataset`. The class overrides
    ``__len__``, ``__getitem__``, ``__contains__``, ``__str__``, ``__eq__`` and ``__init__``.

    Dataset is a two-dimensional immutable, potentially heterogeneous tabular data structure with
    labeled axes (rows and columns).

    Args:
        rows (list of dict): Construct a two-dimensional tabular data structure from rows.

    Attributes:
        columns (set of string): Set of column names.
    """

    def __init__(self, rows):
        self.columns = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('Row must be a dict.')
            self.columns.update(row.keys())
        self.rows = rows

    def __getitem__(self, key):
        """
        Get a column or row from the dataset.

        Args:
            key (str or int): String referencing a column or integer referencing a row
        Returns:
            :class:`list` or :class:`dict`: List of column values or a dict representing a row
        """
        # Given an column string return list of column values.
        # print("key", key)
        # print("key type", type(key))
        if isinstance(key, str):
            if key not in self.columns:
                raise AttributeError('Key not in columns.')
            return [row[key] if key in row else None for row in self.rows]
        # Given an row integer return a object of row values.
        elif isinstance(key, (int, slice)):
            return self.rows[key]
        else:
            raise TypeError('Invalid argument type.')

    def __setitem__(self, key, item):
        """
        Set a column or row for a dataset.

        Args:
            key (str or int): String referencing a column or integer referencing a row
            item (list or dict): Column or rows to set in the dataset.
        """
        if isinstance(key, str):
            column = item
            self.columns.add(key)
            if len(column) > len(self.rows):
                for i, value in enumerate(column):
                    if i < len(self.rows):
                        self.rows[i][key] = value
                    else:
                        self.rows.append({key: value})
            else:
                for i, row in enumerate(self.rows):
                    if i < len(column):
                        self.rows[i][key] = column[i]
                    else:
                        self.rows[i][key] = None
        elif isinstance(key, slice):
            rows = item
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError('Row must be a dict.')
                self.columns.update(row.keys())
            self.rows[key] = rows
        elif isinstance(key, int):
            row = item
            if not isinstance(row, dict):
                raise ValueError('Row must be a dict.')
            self.columns.update(row.keys())
            self.rows[key] = row
        else:
            raise TypeError('Invalid argument type.')

    def __len__(self):
        return len(self.rows)

    def __contains__(self, key):
        return key in self.columns

    def __str__(self):
        return str(pd.DataFrame(self.rows))

    # def __eq__(self, other):
    #     return self.columns == other.columns and self.rows == other.rows

    def __add__(self, other):
        return Dataset(self.rows + other.rows)


# class FewShotSiameseSampler(object):
#     def __init__(self, dataset):
#         '''
#         Args:
#         - labels: an iterable containing the labels (1 for positive pair, 0 for negative pair).
#         '''
#         labels = dataset['label']
#         self.num_samples = len(labels)
        
#         # Extract indexes of positive and negative pairs.
#         self.positive_idxs = np.where(labels == '1')[0]
#         self.negative_idxs = np.where(labels == '0')[0]

#     def __iter__(self):
#         # Shuffle the indices once
#         np.random.shuffle(self.positive_idxs)
#         np.random.shuffle(self.negative_idxs)
        
#         # Iterating over pairs while both sets have elements left
#         paired_iter = chain(*zip(self.positive_idxs, self.negative_idxs))

#         # Remaining items from whichever set was larger
#         remaining_pos = self.positive_idxs[len(self.negative_idxs):]
#         remaining_neg = self.negative_idxs[len(self.positive_idxs):]

#         return chain(paired_iter, remaining_pos, remaining_neg)

#     def __len__(self):
#         return self.num_samples

# class BalancedFewShotSiameseSampler:
#     def __init__(self, dataset, batch_size, shot):
#         self.dataset = dataset
#         self.shot = shot
#         self.classes = list(set([label for _, _, label in dataset]))
#         self.batch_size = batch_size
#         self.positives = [self.dataset[idx] for idx, row in enumerate(dataset) if row['label'] == '1']
#         self.negatives = [self.dataset[idx] for idx, row in enumerate(dataset) if row['label'] == '0']

#     def __iter__(self):
#         pos = random.choices(self.positives, k = int(self.batch_size / 2))
#         neg = random.choices(self.negatives, k = int(self.batch_size / 2))
#         batch = pos
#         batch.extend(neg)
#         for item in batch:
#             yield item
               
#     def __len__(self):
#         # return self.num_batches * self.way * self.shot
#         return math.ceil((self.shot - 1) * self.shot / (2 * self.batch_size))


# -----
# batch sampler 
# class FewShotSiameseSampler(object):
#     def __init__(self, dataset, batch_size):
#         '''
#         Args:
#         - labels: an iterable containing the labels (1 for positive pair, 0 for negative pair).
#         - batch_size: number of samples for each iteration (batch)
#         - iterations: number of iterations (batches) per epoch
#         '''
#         self.batch_size = batch_size
#         # print(dataset)
#         labels = dataset['label']
#         self.num_samples = len(labels)
#         self.labels = np.array(labels)
#         self.iterations = self.num_samples // batch_size
        
#         # Ensure the batch size is even for a balanced number of positive and negative pairs.
#         assert batch_size % 2 == 0, "Batch size should be even!"
        
#         # Extract indexes of positive and negative pairs.
#         self.positive_idxs = np.where(self.labels == '1')[0]
#         self.negative_idxs = np.where(self.labels == '0')[0]

#     def __iter__(self):
#         half_batch = self.batch_size // 2
        
#         for _ in range(self.iterations):
#             pos_batch = np.random.choice(self.positive_idxs, half_batch, replace=False)
#             neg_batch = np.random.choice(self.negative_idxs, half_batch, replace=False)
#             batch = np.concatenate([pos_batch, neg_batch])
#             np.random.shuffle(batch)
            
#             # yield torch.from_numpy(batch)
#             yield batch.tolist()

#     def __len__(self):
#         return self.iterations

# -----
# For smapler not optimzed 
class FewShotSiameseSampler(object):
    def __init__(self, dataset):
        '''
        Args:
        - labels: an iterable containing the labels (1 for positive pair, 0 for negative pair).
        '''
        # print("len(dataset)", len(dataset))
        labels = dataset['label']
        self.num_samples = len(labels)
        self.labels = np.array(labels)
        
        # Extract indexes of positive and negative pairs.
        self.positive_idxs = np.where(self.labels == '1')[0]
        self.negative_idxs = np.where(self.labels == '0')[0]

    def __iter__(self):
        # Making sure we go through the entire dataset
        pos_remaining = list(self.positive_idxs)
        neg_remaining = list(self.negative_idxs)
        
        while pos_remaining and neg_remaining:
            pos_sample = np.random.choice(pos_remaining, 1, replace=False).tolist()
            neg_sample = np.random.choice(neg_remaining, 1, replace=False).tolist()
            
            yield from pos_sample
            yield from neg_sample

            pos_remaining.remove(pos_sample[0])
            neg_remaining.remove(neg_sample[0])
            
        # Yield the remaining positive or negative samples if any
        while pos_remaining:
            pos_sample = np.random.choice(pos_remaining, 1, replace=False).tolist()
            yield from pos_sample
            pos_remaining.remove(pos_sample[0])
            
        while neg_remaining:
            neg_sample = np.random.choice(neg_remaining, 1, replace=False).tolist()
            yield from neg_sample
            neg_remaining.remove(neg_sample[0])

    def __len__(self):
        return self.num_samples

# class FewShotSiameseSampler(object):
#     def __init__(self, dataset, batch_size):
#         '''
#         Args:
#         - labels: an iterable containing the labels (1 for positive pair, 0 for negative pair).
#         - batch_size: number of samples for each iteration (batch)
#         - iterations: number of iterations (batches) per epoch
#         '''
#         self.batch_size = batch_size
#         # print(dataset)
#         labels = dataset['label']
#         self.num_samples = len(labels)
#         self.labels = np.array(labels)
#         self.iterations = self.num_samples // batch_size
        
#         # Ensure the batch size is even for a balanced number of positive and negative pairs.
#         assert batch_size % 2 == 0, "Batch size should be even!"
        
#         # Extract indexes of positive and negative pairs.
#         self.positive_idxs = np.where(self.labels == '1')[0]
#         self.negative_idxs = np.where(self.labels == '0')[0]

#     def __iter__(self):
#         half_batch = self.batch_size // 2
        
#         for _ in range(self.iterations):
#             pos_batch = np.random.choice(self.positive_idxs, half_batch, replace=False)
#             neg_batch = np.random.choice(self.negative_idxs, half_batch, replace=False)
#             batch = np.concatenate([pos_batch, neg_batch])
#             np.random.shuffle(batch)
            
#             # yield torch.from_numpy(batch)
#             yield batch.tolist()

#     def __len__(self):
#         return self.iterations
    
# class FewShotSiameseSampler(Sampler):
#     def __init__(self, dataset, batch_size):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.num_samples = len(dataset)
        
#         if batch_size % 2 != 0:
#             raise ValueError("Batch size should be even to have 50-50% positive and negative samples.")
        
#         self.indices = [i for i in range(self.num_samples)]

#     def __iter__(self):
#         positive_indices = []
#         negative_indices = []

#         while len(positive_indices) < self.batch_size // 2 or len(negative_indices) < self.batch_size // 2:
#             index1 = random.randint(0, self.num_samples - 1)
#             index2 = random.randint(0, self.num_samples - 1)
#             if index1 not in self.indices and index2 not in self.indices:
#                 self.indices.remove(index1)
#                 self.indices.remove(index2)
#                 if self.dataset[index1]['label'] == self.dataset[index2]['label']:
#                     if len(positive_indices) < self.batch_size // 2:
#                         positive_indices.append((index1, index2))
#                 else:
#                     if len(negative_indices) < self.batch_size // 2:
#                         negative_indices.append((index1, index2))
        
#         batch_indices = positive_indices + negative_indices
#         random.shuffle(batch_indices)  # Shuffling to mix positive and negative samples
#         return iter(batch_indices)

#     def __len__(self):
#         return self.num_samples // self.batch_size

class PPIDataset():
    """
    Loads the Dataset from the csv files passed to the parser.
    """
    def  __init__(self) -> None:
        return

    # def collate_lists(self, seqA: list, seqB: list, labels: list, protA_ids: list, protB_ids: list, ncbi_geneA_id: list, ncbi_geneB_id: list) -> list:
    def collate_lists(self, seqA: list, seqB: list, labels: list) -> list:

        """ Converts each line into a dictionary. """
        collated_dataset = []
        for i in range(len(seqA)):
            collated_dataset.append({
                "seqA": str(seqA[i]), 
                "seqB": str(seqB[i]), 
                "label": str(labels[i]),
                # "protA_ids": str(protA_ids[i]).split(","), 
                # "protB_ids": str(protB_ids[i]).split(","), 
                # "ncbi_geneA_id": str(ncbi_geneA_id[i]), 
                # "ncbi_geneB_id": str(ncbi_geneB_id[i]), 
            })
        return collated_dataset

    def _retrieve_dataframe(self, path) -> DataFrame:
        column_names = [
            'SeqA', 'SeqB',
            "Label",'IndexA', 'IndexB'
            # , 
            # "protein_A_ids", "protein_B_ids", 
            # "ncbi_gene_A_id", "ncbi_gene_B_id"
        ] 
        # df: DataFrame = pd.read_csv(path, sep = "\n", names=column_names, header=0) #type:ignore
        df: DataFrame = pd.read_csv(path, names=column_names, header=0) #type:ignore

        return df

    def calculate_stat(self,path):
        df = self._retrieve_dataframe(path)
        self.nSamples_dic = df['class'].value_counts()

    # def load_predict_dataset(self, path):
    #     column_names = [
    #         "interaction", "probability",
    #         'receptor_protein_id',
    #         "receptor_protein_label",
    #         'receptor_protein_name',
    #         'receptor_protein_sequence',
    #         'capsid_protein_sequence',
    #     ] 
    #     # df: DataFrame = pd.read_csv(path, sep = "\n", names=column_names, header=0) #type:ignore
    #     df: DataFrame = pd.read_csv(path, names=column_names, header=0) #type:ignore

    #     # interactions = list(df['interaction'])
    #     # probabilities = list(df['probability'])
    #     # receptor_protein_ids = list(df['receptor_protein_id'])
    #     # receptor_protein_labels = list(df['receptor_protein_label'])
    #     # receptor_protein_names = list(df['receptor_protein_name'])
    #     # receptor_protein_seqs = list(df['receptor_protein_sequence'])
    #     # capsid_protein_sequences = list(df['capsid_protein_sequence']) 

    #     # Make sure there is a space between every token, and map rarely amino acids
    #     receptor_protein_seqs = [" ".join("".join(sample.split())) for sample in receptor_protein_seqs]
    #     receptor_protein_seqs = [re.sub(r"[UZOB]", "X", sample) for sample in receptor_protein_seqs]
        
    #     capsid_protein_sequences = [" ".join("".join(sample.split())) for sample in capsid_protein_sequences]
    #     capsid_protein_sequences = [re.sub(r"[UZOB]", "X", sample) for sample in capsid_protein_sequences]

    #     # assert len(receptor_protein_seqs) == len(interactions)
    #     # assert len(capsid_protein_sequences) == len(interactions)
    #     # assert len(receptor_protein_ids) == len(interactions)
    #     # assert len(receptor_protein_names) == len(interactions)
    #     # assert len(receptor_protein_labels) == len(interactions)
    #     # assert len(probabilities) == len(interactions)

    #     collated_dataset = []
    #     for i in range(len(interactions)):
    #         collated_dataset.append({
    #             "label": str(interactions[i]),
    #             # "probability": str(probabilities[i]), 
    #             # "receptor_protein_id": str(receptor_protein_ids[i]), 
    #             # "receptor_protein_label": str(receptor_protein_labels[i]), 
    #             # "receptor_protein_name": str(receptor_protein_names[i]), 
    #             "seqA": str(receptor_protein_seqs[i]), 
    #             "seqB": str(capsid_protein_sequences[i]), 
    #         })

    #     return Dataset(collated_dataset)

    def load_dataset(self, path):
        df = self._retrieve_dataframe(path)[:10]

        labels = list(df['Label'])
        seqA = list(df['SeqA'])
        seqB = list(df['SeqB'])
        # protA_ids = list(df['protein_A_ids']) # TODO: Rename to protein_A_ids
        # protB_ids = list(df['protein_B_ids']) # TODO: Rename to protein_A_ids
        # ncbi_geneA_id = list(df['ncbi_gene_A_id'])
        # ncbi_geneB_id = list(df['ncbi_gene_B_id'])

        # Make sure there is a space between every token, and map rarely amino acids
        # print("labels", labels)
        # print("seqA", seqA)
        # print("seqB", seqB)
        seqA = [" ".join("".join(sample.split())) for sample in seqA]
        seqA = [re.sub(r"[UZOB]", "X", sample) for sample in seqA]
        
        seqB = [" ".join("".join(sample.split())) for sample in seqB]
        seqB = [re.sub(r"[UZOB]", "X", sample) for sample in seqB]

        assert len(seqA) == len(labels)
        assert len(seqB) == len(labels)
        # assert len(protA_ids) == len(labels)
        # assert len(protB_ids) == len(labels)
        
        # return Dataset(self.collate_lists(seqA, seqB, labels, protA_ids, protB_ids, ncbi_geneA_id, ncbi_geneB_id))
        return Dataset(self.collate_lists(seqA, seqB, labels))



