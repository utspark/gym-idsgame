from itertools import groupby
from typing import List, Tuple, Dict, Optional, Final

import joblib
import numpy as np
from hmmlearn import hmm
from bisect import bisect_left

from detector_framework import config


class LifecycleDetector:
    DEFAULT_UNIFORM_SUBSEQ_LEN: Final[int] = 3
    DEFAULT_DENSITY_SCALER: Final[float] = 0.08
    DEFAULT_PROPAGATION_SCALER: Final[float] = 0.12
    DEFAULT_PROBA_THRESHOLD: Final[float] = 0.95

    @staticmethod
    def form_lifecycle_sequence(attack_stages: Dict[str, List[str]], benign: bool = False) -> Tuple[List[str], List[int]]:
        # TODO benign sequences
        # TODO consider benign states
        #  - (states: list)
        #  - pb-lifecycle/src/classifier.py

        if benign:
            techniques = ["_b_" for _ in attack_stages]
        else:
            techniques = []
            for stage, ttp_choices in attack_stages.items():
                # TODO performance increase when setting replace=True
                ttp = np.random.choice(ttp_choices, size=1)[0]
                techniques.append(ttp)

        stage_keys = []
        stage_windows = []

        for technique in techniques:
            stage_keys.append(technique)
            # Random window sizes for each stage
            stage_windows.append(np.random.choice(np.arange(10, 100, 10)))

        return stage_keys, stage_windows

    def __init__(self,
                 syscall_clf_path: Optional[str] = None,
                 network_clf_path: Optional[str] = None,
                 hpc_clf_path: Optional[str] = None,
                 lifecycle_awareness: bool = True,
                 stage_filter: bool = False,
                 density: bool = False,
                 propagation: bool = False,
                 memory: bool = False,
                 proba_threshold: float = DEFAULT_PROBA_THRESHOLD
                 ):
        if not any([syscall_clf_path, network_clf_path, hpc_clf_path]):
            raise ValueError("At least one classifier path must be provided.")

        self.syscall_clf = self._load_clf(syscall_clf_path)
        self.network_clf = self._load_clf(network_clf_path)
        self.hpc_clf = self._load_clf(hpc_clf_path)

        self.hmm = self._get_markov()
        self.lifecycle_awareness = lifecycle_awareness
        self.stage_filter = stage_filter
        self.density = density
        self.propagation = propagation
        self.memory = memory
        self.proba_threshold = proba_threshold

    @staticmethod
    def _load_clf(path: Optional[str]):
        if path is None:
            return None
        try:
            clf = joblib.load(path)
            return clf[0] if isinstance(clf, tuple) else clf
        except Exception as e:
            # TODO: Add proper logging
            print(f"Error loading classifier from {path}: {e}")
            return None

    @staticmethod
    def _get_markov() -> hmm.CategoricalHMM:
        # Probabilities for starting at each stage
        # Favoring starting at earlier stages
        alternate_start_weights = [i for i in range(1, 5)][::-1]
        start_matrix = np.array(alternate_start_weights) / np.sum(alternate_start_weights)

        # Transition probabilities
        f_b_ratio = 2  # ratio of forward over backward transition confidence
        t_f0 = 1 / (4 * f_b_ratio + 0) * f_b_ratio
        t_f1 = 1 / (3 * f_b_ratio + 1) * f_b_ratio
        t_f2 = 1 / (2 * f_b_ratio + 2) * f_b_ratio
        t_f3 = 1 / (1 * f_b_ratio + 3) * f_b_ratio

        t_b1 = 1 / (3 * f_b_ratio + 1)
        t_b2 = 1 / (2 * f_b_ratio + 2)
        t_b3 = 1 / (1 * f_b_ratio + 3)

        transition_matrix = [
            [t_f0, t_f0, t_f0, t_f0],
            [t_b1, t_f1, t_f1, t_f1],
            [t_b2, t_b2, t_f2, t_f2],
            [t_b3, t_b3, t_b3, t_f3],
        ]

        # Emission matrix (identity matrix since observations match stages)
        emission_matrix = np.eye(4)

        model = hmm.CategoricalHMM(n_components=4, n_features=4)
        model.startprob_ = start_matrix
        model.transmat_ = np.array(transition_matrix)
        model.emissionprob_ = emission_matrix

        return model

    def cross_layer_class_preds(self, cross_layer_X: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        cross_layer_classes = []
        cross_layer_probas = []

        clfs = [self.syscall_clf, self.network_clf, self.hpc_clf]
        translations = [
            config.SYSCALL_BENIGN_MALWARE_CLASS_TRANSLATION,
            config.NETWORK_BENIGN_MALWARE_CLASS_TRANSLATION,
            config.HPC_BENIGN_MALWARE_CLASS_TRANSLATION,
        ]

        for clf, layer_data, translation in zip(clfs, cross_layer_X, translations):
            if clf is None or np.all(layer_data == -1):
                classes = layer_data[:, 0]
                probas = layer_data[:, 0]
            else:
                preds = clf.predict_proba(layer_data)
                probas = np.max(preds, axis=1)
                classes = np.argmax(preds, axis=1)
                classes[probas < self.proba_threshold] = -1

            # Vectorized translation
            vector_translate = np.vectorize(lambda x: translation.get(x, x))
            classes = vector_translate(classes)
            cross_layer_classes.append(classes)
            cross_layer_probas.append(probas)

        combined_classes = np.stack(cross_layer_classes).T
        combined_probas = np.stack(cross_layer_probas).T

        return combined_classes, combined_probas

    def filter(self, class_sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filters the sequence based on minimum subsequence length."""
        new_sequence = []
        for key, group in groupby(class_sequence):
            emission_len = len(list(group))
            new_sequence.append((key, emission_len))

        if not new_sequence:
            return np.array([]), np.array([])

        new_sequence = np.array(new_sequence)
        
        if self.stage_filter:
            # Filter out entries where the length is less than the threshold
            mask = new_sequence[:, 1].astype(int) >= self.DEFAULT_UNIFORM_SUBSEQ_LEN
            new_sequence = new_sequence[mask]

        if len(new_sequence) < 1:
            return np.array([]), np.array([])

        values = new_sequence[:, 0]
        counts = new_sequence[:, 1].astype(int)

        return values, counts

    @staticmethod
    def _collate_preds(preds: np.ndarray, probas: np.ndarray) -> np.ndarray:
        """Collates predictions from multiple layers by picking the one with highest probability."""
        results = []

        for row, proba_row in zip(preds, probas):
            mask = row != -1
            if not np.any(mask):
                continue

            valid_preds = row[mask]
            valid_probas = proba_row[mask]
            results.append(valid_preds[np.argmax(valid_probas)])

        return np.array(results)


    @staticmethod
    def _longest_increasing_subsequence(nums: List[int]) -> Tuple[List[int], List[int]]:
        """Returns the longest increasing subsequence and its indices."""
        n = len(nums)
        if n == 0:
            return [], []

        tails = []  # minimal tail value for each length
        tails_idx = []  # index in nums of that tail
        prev = [-1] * n  # predecessor index for reconstruction

        for i, x in enumerate(nums):
            j = bisect_left(tails, x)
            if j == len(tails):
                tails.append(x)
                tails_idx.append(i)
            else:
                tails[j] = x
                tails_idx[j] = i
            if j > 0:
                prev[i] = tails_idx[j - 1]

        # Reconstruct LIS
        k = tails_idx[-1]
        lis_idx = []
        while k != -1:
            lis_idx.append(k)
            k = prev[k]
        
        lis_idx.reverse()
        return [nums[i] for i in lis_idx], lis_idx

    def score_stage_sequence(self, stage_sequence: np.ndarray, clf_predictions: np.ndarray) -> float:
        score = 0.0

        if self.density:
            density_penalty = (len(stage_sequence) / len(clf_predictions)) * self.DEFAULT_DENSITY_SCALER
            if self.lifecycle_awareness:
                density_penalty *= 1 / (1 + np.exp(-len(clf_predictions) / 100))
            score += density_penalty

        if len(stage_sequence) > 0 and self.lifecycle_awareness:
            if self.propagation:
                subseq, _ = self._longest_increasing_subsequence(list(stage_sequence))
                score += len(subseq) * self.DEFAULT_PROPAGATION_SCALER

            # Filter for HMM scoring
            stage_seq_filtered, counts = self.filter(stage_sequence)

            # TODO explore this
            #   if len(counts) > 3:
            #       stage_duration_penalty = np.mean([np.square(count) for count in counts[:3]])
            #       proba += 1 / (1 + np.exp(-1 * .01 * stage_duration_penalty)) * 0.2
            #   counts = np.sort(counts)
            #   stage_presence = {i: 0 for i in range(4)}
            #   for i, stage in enumerate(stage_sequence):
            #       if counts[i] > stage_presence[stage]:
            #           stage_presence[stage] = counts[i]
            #   stage_duration_penalty = np.sum([np.square(val) for val in stage_presence.values()])
            #   stage_duration_penalty = 1 / (1 + np.exp(-1 * .0001 * stage_duration_penalty)) * 0.2
            #   proba += stage_duration_penalty

            hmm_score = np.exp(self.hmm.score(np.array(stage_seq_filtered).reshape(-1, 1)))
            # Normalization by length
            hmm_score = np.power(hmm_score, 1 / len(stage_seq_filtered))
            score += hmm_score

        return score

    def score_cross_layer(self, cross_layer_X: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
        """Calculates a global detection score across multiple data layers."""
        clf_predictions, clf_probas = self.cross_layer_class_preds(cross_layer_X)
        collated_predictions = self._collate_preds(clf_predictions, clf_probas)

        return self.score_stage_sequence(collated_predictions, clf_predictions)

    def score_single_layer(self, trace_classes: np.ndarray, trace_values: np.ndarray, translation: Dict[int, int]) -> float:
        """Calculates a global detection score for a single layer of data."""
        vectorized_translate = np.vectorize(lambda x: translation.get(x, x))
        clf_predictions = vectorized_translate(trace_classes)
        predictions = clf_predictions[trace_values >= self.proba_threshold]

        return self.score_stage_sequence(predictions, clf_predictions)



