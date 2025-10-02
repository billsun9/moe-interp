import os
import torch
import heapq
import time
from typing import List, Dict, Tuple, Any, Union
from collections import defaultdict, Counter
from itertools import combinations, permutations

from moe.path_config import *
from moe.activation_extraction.activation_extractor_utils import OLMoEActivationExtractor
from moe.eda.utils import load_all_activations, compute_expert_set_counts, compute_per_dataset_token_counts

class FeatureFinder:
    def __init__(self, d, subset_sizes=(1,2,3,4)):
        self.d = d
        self.subset_sizes = subset_sizes
        self.combinations_, self. permutations_order_sensitive_, self.permutations_order_insensitive_ = compute_expert_set_counts(self.d, self.subset_sizes)
        self.per_dataset_counts = compute_per_dataset_token_counts(self.d)

    def find_dataset_specific_expert_sets(self, dataset_names, mode = "combinations", subset_sizes = None, keep_n = 10, min_threshold = 100, normalize = True):
        if type(dataset_names) == str:
            dataset_names = [dataset_names]
        if not subset_sizes:
            subset_sizes = self.subset_sizes
        assert type(subset_sizes) == tuple, "subset_sizes must be a tuple!"
        
        if mode == 'combinations':
            expert_set_counts = self.combinations_
        elif mode == 'permutations_order_sensitive':
            expert_set_counts = self.permutations_order_sensitive_
        elif mode == 'permutations_order_insensitive':
            expert_set_counts = self.permutations_order_insensitive_
        else:
            raise ValueError(f"Invalid mode: {mode}")
        res = {}
        for subset_size in subset_sizes:
            top_samples = []
            for expert_set, ds_counter in expert_set_counts.items():
                if ds_counter.total() < min_threshold or len(expert_set) != subset_size:
                    continue
                score = self.calculate_score(ds_counter, dataset_names, normalize)
                heapq.heappush(top_samples, (score, ds_counter.total(), expert_set, ds_counter))
                if len(top_samples) > keep_n:
                    heapq.heappop(top_samples)
            res[subset_size] = sorted(top_samples, key=lambda x: x[0], reverse=True)
        
        return res

    def calculate_score(self, ds_counter, dataset_names, normalize = True):
        """
        Calculates the relative frequency of each token across all datasets
        """
        ds_counts = dict(ds_counter)
        if normalize:
            for key in ds_counts:
                ds_counts[key] /= self.per_dataset_counts[key]
        total = sum(ds_counts.values())
        return sum(ds_counts.get(ds_name, 0) for ds_name in dataset_names) / total

    

datasets_to_process = [
    # False ==> Question: {Q} [Optional: {Choices} or {Reasoning}] Answer:{A}
    ("gsm8k", False),
    ("arc-easy", False),
    ("arc-challenge", False),
    ("sciq", False),
    ("mbpp", False),
    # False ==> Not a question. Just the {statement}
    ("ag_news", False),
    ("imdb_pos", False),
    ("imdb_neg", False),
    # False/True does not matter for these. Just the {statement}
    ("poetry", False),
    ("lex_glue", False),
    ("arxiv", False),
    ("personas", False)
]
start_time = time.time()
extractor = OLMoEActivationExtractor()

LAYER_IDX = 7

d = load_all_activations(
    datasets_to_process,
    LAYER_IDX,
    extractor,
    split_names = ["train"]
)

F = FeatureFinder(d, subset_sizes = (1,2,3))

for dataset_name in d.keys():
    print(f"======================================= {dataset_name} =======================================")
    output = F.find_dataset_specific_expert_sets(dataset_name)
    for k, v in output.items():
        print(k,v)