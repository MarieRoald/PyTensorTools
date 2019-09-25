import itertools
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from subprocess import Popen

import numpy as np
from scipy.io import loadmat, savemat
from scipy.stats import ttest_ind

import tenkit
from tenkit import metrics  

from .base_evaluator import BaseEvaluator


class BaseSingleRunEvaluator(BaseEvaluator):
    def __call__(self, data_reader, h5):
        return self._evaluate(data_reader, h5)

    @abstractmethod
    def _evaluate(self, data_reader, h5):
        pass


class FinalLoss(BaseSingleRunEvaluator):
    _name = "Final loss"

    def _evaluate(self, data_reader, h5):
        return {self.name: h5["LossLogger/values"][-1]}


class ExplainedVariance(BaseSingleRunEvaluator):
    # TODO: maybe create a decomposer to not rely on logging
    _name = "Explained variance"

    def _evaluate(self, data_reader, h5):
        return {self.name: h5["ExplainedVarianceLogger/values"][-1]}


class ConvergenceTolerance(BaseSingleRunEvaluator):
    _name = "Minimum relative loss change"

    def _evaluate(self, data_reader, h5):
        loss = h5["LossLogger/values"][...]
        rel_change = (loss[1:] - loss[:-1]) / loss[:-1]
        return {self.name: rel_change.min()}


class AllPValues(BaseSingleRunEvaluator):
    _name = "All P values"

    def __init__(self, summary, mode, class_name, **kwargs):
        super().__init__(summary)
        self.mode = mode
        self._name = f"All P values for mode {mode}"
        self.class_name = class_name

    def _calculate_p_values_from_factors(self, data_reader, h5):
        decomposition = self.load_final_checkpoint(h5)
        factors = decomposition.factor_matrices[self.mode]

        classes = data_reader.classes[self.mode][self.class_name].squeeze()

        assert len(set(classes)) == 2

        indices = [
            [i for i, c in enumerate(classes) if c == class_] for class_ in set(classes)
        ]
        p_values = tuple(
            ttest_ind(factors[indices[0]], factors[indices[1]], equal_var=False).pvalue
        )
        return p_values

    def _evaluate(self, data_reader, h5):
        p_values = self._calculate_p_values_from_factors(data_reader, h5)

        return {f"p_value_component{i}": p_value for i, p_value in enumerate(p_values)}


class MinPValue(AllPValues):
    _name = "Best P value"

    def __init__(self, summary, mode, class_name, **kwargs):
        super().__init__(summary, mode, class_name)
        self._name = f"Best P value for mode {mode}"

    def _evaluate(self, data_reader, h5):
        p_values = self._calculate_p_values_from_factors(data_reader, h5)
        return {self.name: min(p_values), "component": int(np.argmin(p_values))}


class ClassBalance(BaseSingleRunEvaluator):
    # Only works for two classes
    # TODO: consider supporting more classes
    _name = "Class balance"

    def __init__(self, summary, mode, class_name, **kwargs):
        super().__init__(summary)
        self.mode = mode
        self.class_name = class_name

    def _evaluate(self, data_reader, h5):

        classes = data_reader.classes[self.mode][self.class_name].squeeze()

        unique_classes = list(set(classes))
        assert len(unique_classes) == 2

        num_class1 = int(np.sum(classes == unique_classes[0]))
        num_class2 = int(np.sum(classes == unique_classes[1]))

        ratio = num_class1 / (num_class1 + num_class2)

        return {
            self.name: ratio,
            f"No. class {unique_classes[0]}": num_class1,
            f"No. class {unique_classes[1]}": num_class2,
        }


class WorstDegeneracy(BaseSingleRunEvaluator):
    _name = "Worst degeneracy"

    def __init__(self, summary, modes=None, return_permutation=False, **kwargs):
        super().__init__(summary)
        self.modes = modes
        self.return_permutation = return_permutation

    def _evaluate(self, data_reader, h5):
        decomposition = self.load_final_checkpoint(h5)
        factors = decomposition.factor_matrices

        modes = self.modes
        if modes is None:
            modes = range(len(decomposition.factor_matrices))
        R = decomposition.factor_matrices[0].shape[1]
        min_score = np.inf

        for (p1, p2) in itertools.permutations(range(R), r=2):

            factors_p1 = [
                fm[:, p1]
                for mode, fm in enumerate(decomposition.factor_matrices)
                if mode in modes
            ]
            factors_p2 = [
                fm[:, p2]
                for mode, fm in enumerate(decomposition.factor_matrices)
                if mode in modes
            ]

            score = metrics._factor_match_score(
                factors_p1, factors_p2, nonnegative=False, weight_penalty=False
            )[0]

            if score < min_score:
                min_score = score
                worst_p1 = p1
                worst_p2 = p2

        if self.return_permutation:
            return {self.name: min_score, f"permutation:": (worst_p1, worst_p2)}

        return {self.name: min_score}


class Parafac2WorstDegeneracy(BaseSingleRunEvaluator):
    _name = "Worst degeneracy parafac2"

    def __init__(self, summary, modes=None, return_permutation=False, **kwargs):
        super().__init__(summary)
        self.modes = modes
        self.return_permutation = return_permutation

    def _evaluate(self, data_reader, h5):
        decomposition = self.load_final_checkpoint(h5)
        factors = decomposition.factor_matrices

        if self.modes is None:
            modes = range(len(decomposition.factor_matrices))
        R = decomposition.factor_matrices[0].shape[1]
        min_score = np.inf

        for (p1, p2) in itertools.permutations(range(R), r=2):

            # factors_p2 = [
            #    fm[:, p2] for mode, fm in enumerate(decomposition.factor_matrices) if mode in [0, 2]
            # ]
            factors_p1 = [
                factors[0][:, np.newaxis, p1],
                [f[:, np.newaxis, p1] for f in factors[1]],
                factors[2][:, np.newaxis, p1],
            ]
            factors_p2 = [
                factors[0][:, np.newaxis, p2],
                [f[:, np.newaxis, p2] for f in factors[1]],
                factors[2][:, np.newaxis, p2],
            ]

            score = metrics._factor_match_score_parafac2(
                factors_p1, factors_p2, nonnegative=False, weight_penalty=False
            )[0]

            if score < min_score:
                min_score = score
                worst_p1 = p1
                worst_p2 = p2

        if self.return_permutation:
            return {self.name: min_score, f"permutation:": (worst_p1, worst_p2)}

        return {self.name: min_score}


class WorstDegeneracy2(BaseSingleRunEvaluator):
    _name = 'WorstDegeneracy2'
    def _evaluate(self, data_reader, h5):
        decomposition = self.load_final_checkpoint(h5)
        degeneracy_scores = decomposition.degeneracy()
    
        return {self.name: np.min(degeneracy_scores), 
                'components': np.array(np.unravel_index(np.argmin(degeneracy_scores), shape=degeneracy_scores.shape)).to_list(),
                'TC': degeneracy_scores}                

class CoreConsistency(BaseSingleRunEvaluator):
    # Only works with three modes

    _name = "Core Consistency"

    def _evaluate(self, data_reader, h5):
        decomposition = self.load_final_checkpoint(h5)
        factor_matrices = decomposition.factor_matrices

        # FIX:
        weights = decomposition.weights
        d = len(factor_matrices)
        factor_matrices = [fm * weights ** (1 / d) for fm in factor_matrices]
        # ENDFIX

        cc = tenkit.metrics.core_consistency(data_reader.tensor, *factor_matrices)
        return {self.name: np.asscalar(cc)}


class Parafac2CoreConsistency(BaseSingleRunEvaluator):
    # Only works with three modes

    _name = "Parafac2 Core Consistency"

    def _evaluate(self, data_reader, h5):
        decomposition = self.load_final_checkpoint(h5)

        P_k = decomposition.projection_matrices
        A = decomposition.A
        B = decomposition.blueprint_B
        C = decomposition.C

        cc = tenkit.metrics.core_consistency_parafac2(
            data_reader.tensor.transpose(2, 0, 1), P_k, A, B, C
        )

        return {self.name: np.asscalar(cc)}



class BaseMatlabEvaluator(BaseSingleRunEvaluator):
    def __init__(self, summary, matlab_scripts_path=None, **kwargs):
        super().__init__(summary)
        self.matlab = ["matlab"]
        self.options = ["-nosplash", "-nodesktop", "-r"]
        if matlab_scripts_path is None:
            matlab_scripts_path = Path(__file__).parent / "legacy_matlab_code"
        self.matlab_scripts_path = str(matlab_scripts_path)


class MaxKMeansAcc(BaseMatlabEvaluator):
    _name = "Max Kmeans clustering accuracy"

    def __init__(self, summary, mode, class_name, matlab_scripts_path=None, **kwargs):
        self.mode = mode
        self.class_name = class_name
        super().__init__(summary, matlab_scripts_path)

    def _evaluate(self, data_reader, h5):
        decomposition = self.load_final_checkpoint(h5)
        factor_matrix = decomposition.factor_matrices[self.mode]

        classes = data_reader.classes[self.mode][self.class_name].squeeze()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            tmp_matlab_factor_file = tmpdir / "tmp_matlab_factor.mat"
            tmp_matlab_classes_file = tmpdir / "tmp_matlab_classes.mat"

            tmp_outfile = tmpdir / "tmp_matlab_kmeans_acc.out"

            savemat(str(tmp_matlab_classes_file), {"classes": classes})
            savemat(str(tmp_matlab_factor_file), {"factormatrix": factor_matrix})

            num_classes_elements = classes.shape[0]
            num_samples = factor_matrix.shape[0]
            assert num_classes_elements == num_samples, (
                f"Length of classes vector ({num_classes_elements}) "
                f"differs from number of samples ({num_samples}). "
            )

            command = [
                f"load('{tmp_matlab_factor_file}');\
                  load('{tmp_matlab_classes_file}'); \
                  addpath(genpath('{self.matlab_scripts_path}'));\
                  [acc]=run_kmeans_acc_from_python(classes, factormatrix');\
                  save('{tmp_outfile}');\
                  exit"
            ]

            p = Popen(self.matlab + self.options + command)
            stdout, stderr = p.communicate()

            outdict = loadmat(tmp_outfile)
            acc = outdict["acc"].tolist()[0][0]

            return {self.name: acc}


class ComponentWisePercentExplained(BaseEvaluator):
    def _evaluate(self, data_reader, h5):
        decomposition = self.load_final_checkpoint(h5)

        componentwise_explained = {}
        for component in range(decomposition.rank):
            single_component_decomposition = decomposition.get_single_component_decomposition(component)
            single_component_tensor = single_component_decomposition.construct_tensor()
            percent_exp = metrics.percent_explained(data_reader.tensor, single_component_tensor)
            componentwise_explained[f'%_exp_component_{component}'] = percent_exp
        return  componentwise_explained


class Weights(BaseEvaluator):
    def _evaluate(self, data_reader, h5):
        decomposition = self.load_final_checkpoint(h5)
        return {
            f'weights_component{r}': w
                for r, w in enumerate(decomposition.weights)
        }


class FactorNorms(BaseEvaluator):
    def _evaluate(self, data_reader, h5):
        decomposition = self.load_final_checkpoint(h5)
        norms = np.ones(decomposition.rank)
        for fm in decomposition.factor_matrices:
            norms *= np.linalg.norm(fm, axis=0)
        return {
            f'norm_component{r}': norm
                for r, norm in enumerate(norms)
        }


class Parafac2FactorNorms(BaseEvaluator):
    def _evaluate(self, data_reader, h5):
        decomposition = self.load_final_checkpoint(h5)
        norms = np.linalg.norm(decomposition.A, axis=0)
        norms *= np.linalg.norm(decomposition.B[0], axis=0)
        norms *= np.linalg.norm(decomposition.C, axis=0)
        return {
            f'norm_component{r}': norm
                for r, norm in enumerate(norms)
        }