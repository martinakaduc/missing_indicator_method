import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency, mode
from scipy.stats.contingency import crosstab
from statsmodels.stats.multitest import multipletests
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.impute import MissingIndicator
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from gcimpute.gaussian_copula import GaussianCopula
from gcimpute.low_rank_gaussian_copula import LowRankGaussianCopula
from tqdm import tqdm


class GCImputer(BaseEstimator, TransformerMixin):
    def __init__(self, **gc_params):
        # kwargs depend on the model used, so assign them whatever they are
        for key, value in gc_params.items():
            setattr(self, key, value)

        self._param_names = list(gc_params.keys())

    def get_params(self, deep=True):
        return {param: getattr(self, param) for param in self._param_names}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def fit(self, X, y=None):
        gc_params = self.get_params()
        self.gc = GaussianCopula(**gc_params)
        self.gc.fit(X)
        return self.gc

    def transform(self, X, y=None):
        X_imputed = self.gc.transform(X)
        return X_imputed


class LRGCImputer(BaseEstimator, TransformerMixin):
    def __init__(self, **gc_params):
        # kwargs depend on the model used, so assign them whatever they are
        for key, value in gc_params.items():
            setattr(self, key, value)

        self._param_names = list(gc_params.keys())

    def get_params(self, deep=True):
        return {param: getattr(self, param) for param in self._param_names}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def fit(self, X, y=None):
        gc_params = self.get_params()
        print(gc_params)
        self.lrgc = LowRankGaussianCopula(**gc_params)
        self.lrgc.fit(X)
        return self.lrgc

    def transform(self, X, y=None):
        X_imputed = self.lrgc.transform(X)
        return X_imputed


# New transform to add indicator for each partially observed feature
# i.e. the Missing Indicator Method (MIM)


class MIM(MissingIndicator):
    def __init__(
        self, missing_values=np.nan, features="missing-only", alpha=0.05, **kwargs
    ):
        self.mim_features = features
        self.alpha = alpha
        self.kwargs = kwargs

        if features == "dynamic" or features == "dynamic_wga":
            features = "missing-only"

        super().__init__(
            missing_values=missing_values,
            features=features,
        )

    def fit(self, X, y=None, X_imputed=None):
        mask = super()._fit(X, y)
        self.mask = mask
        self.p_vals = []

        # print(f"Features with missing values: {self.features_}")

        if self.mim_features == "dynamic":
            if y is None:
                raise ValueError("features was set to dynamic, but y is None")

            # indicators_to_keep = []
            if len(self.features_) > 0:
                for i in self.features_:
                    # Test for relation between groups
                    # Use t-test if y is continuous
                    # Use Chi-Square test if y is categorical
                    if y.dtype.name in ["int64", "object", "category"]:
                        # Make contingency table and do Chi-Squared test
                        _, table = crosstab(mask[:, i], y)
                        _, p_val, _, _ = chi2_contingency(table)
                    else:
                        # Group y by value of indicator and do two-sample t-test
                        y_groups = [y[mask[:, i] == k] for k in [0, 1]]
                        p_val = ttest_ind(*y_groups, equal_var=False).pvalue

                    # if p_val < self.alpha:
                    #     indicators_to_keep.append(i)

                    self.p_vals.append(p_val)

                # self.features_ = np.array(indicators_to_keep)

                # Keep indicator only if p_val is significant
                # Use Benjamini-Hochbert correction on p-values
                test_results, _, _, _ = multipletests(
                    pvals=self.p_vals,
                    alpha=self.alpha,
                    method="fdr_bh",
                )
                self.features_ = np.flatnonzero(test_results)

        elif self.mim_features == "dynamic_wga":
            if y is None:
                raise ValueError("features was set to dynamic_wga, but y is None")

            # Use Genetic Algorithm to select features
            # Alforithm Sketch:

            # Input: R, Y, GA_params
            # Output: I (subset of indicator indices)

            # Initialize populatnion of binary vectors Z (each of length p)
            # for generation in 1 to max_generations:
            #     Evaluate fitness of each Z_i using:
            #         - Subset R[:, Z_i==1]
            #         - Classifier performance or mutual information with Y
            #     Select top-performing individuals
            #     Apply crossover and mutation to create next generation
            #     Keep best individuals (elitism)
            # Best_subset = individual with highest fitness
            # I = {j for j in range(p) if Best_subset[j] == 1}
            # return I

            population_size = self.kwargs.get("population_size", 1000)
            max_generations = self.kwargs.get("max_generations", 10)
            mutation_rate = self.kwargs.get("mutation_rate", 0.1)
            crossover_rate = self.kwargs.get("crossover_rate", 0.7)
            selection_rate = self.kwargs.get("selection_rate", 0.5)
            elitism_rate = self.kwargs.get("elitism_rate", 0.1)

            # Implementation
            if len(self.features_) > 0:
                initial_population = np.random.randint(
                    2, size=(population_size, len(self.features_))
                )
                best_individual = initial_population[0]
                best_fitness = 0

                for generation in range(max_generations + 1):
                    print(f"Generation {generation}/{max_generations}")
                    population_size = len(initial_population)

                    # Evaluate fitness of each individual
                    list_fitness = []
                    for individual in tqdm(
                        initial_population, desc="Evaluating individuals"
                    ):
                        # Classifier performance
                        _model_ = DecisionTreeRegressor(max_depth=5)
                        # _model_ = GaussianProcessRegressor(
                        #     kernel=C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0),
                        #     n_restarts_optimizer=10,
                        #     random_state=42,
                        # )
                        masked_X = np.hstack((X_imputed, mask[:, individual]))
                        _model_.fit(masked_X, y)
                        fitness = _model_.score(masked_X, y)
                        list_fitness.append(fitness)
                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_individual = individual

                    if generation == max_generations:
                        break

                    # Selection
                    selected_individuals = initial_population[np.argsort(list_fitness)]
                    selected_individuals = selected_individuals[
                        -int(population_size * selection_rate) :
                    ]

                    # Crossover
                    for i in range(0, len(selected_individuals), 2):
                        if np.random.rand() < crossover_rate:
                            crossover_point = np.random.randint(
                                1, len(selected_individuals[0]) - 1
                            )
                            (
                                selected_individuals[i][:crossover_point],
                                selected_individuals[i + 1][:crossover_point],
                            ) = (
                                selected_individuals[i + 1][:crossover_point],
                                selected_individuals[i][:crossover_point],
                            )
                    # Mutation
                    for individual in selected_individuals:
                        if np.random.rand() < mutation_rate:
                            mutation_point = np.random.randint(0, len(individual))
                            individual[mutation_point] = 1 - individual[mutation_point]
                    # Elitism
                    num_elites = int(population_size * elitism_rate)
                    elite_individuals = initial_population[np.argsort(list_fitness)][
                        -num_elites:
                    ]

                    # Combine selected individuals and elite individuals
                    initial_population = np.vstack(
                        (selected_individuals, elite_individuals)
                    )

                # Select features
                self.features_ = np.flatnonzero(best_individual)

        return self

    def transform(self, X, y=None):

        check_is_fitted(self)

        # Need not validate X again as it would have already been validated
        # in the Imputer calling MissingIndicator
        if not self._precomputed:
            X = self._validate_input(X, in_fit=False)
        else:
            if not (hasattr(X, "dtype") and X.dtype.kind == "b"):
                raise ValueError("precomputed is True but the input data is not a mask")

        imputer_mask, features = self._get_missing_features_info(X)

        if self.mim_features == "missing-only":
            features_diff_fit_trans = np.setdiff1d(features, self.features_)
            if self.error_on_new and features_diff_fit_trans.size > 0:
                raise ValueError(
                    "The features {} have missing values "
                    "in transform but have no missing values "
                    "in fit.".format(features_diff_fit_trans)
                )

        if self.mim_features in ["missing-only", "dynamic", "dynamic_wga"]:
            if len(self.features_) < self._n_features:
                imputer_mask = imputer_mask[:, self.features_]

        return imputer_mask

    def fit_transform(self, X, y=None, X_imputed=None):
        self.fit(X, y, X_imputed)
        imputer_mask = self.mask

        if len(self.features_) < self._n_features:
            imputer_mask = imputer_mask[:, self.features_]

        # print(f"Alpha: {self.alpha}\tKept features: {len(self.features_)}")
        # print(np.mean([
        #     pval for pval, j in zip(self.p_vals, range(X.shape[1])) if j not in self.features_
        # ]))

        # if len(immputer_mask.shape) == 3:
        #     imputer_mask = imputer_mask.squeeze()

        return imputer_mask


class Oracle_MIM(MissingIndicator):
    def __init__(
        self,
        *,
        indicators_to_keep,
        missing_values=np.nan,
    ):
        self.indicators_to_keep = indicators_to_keep
        super().__init__(
            missing_values=missing_values,
            features="missing-only",
        )

    def fit(self, X, y=None):
        mask = super()._fit(X, y)
        self.mask = mask
        self.features_ = self.indicators_to_keep
        return self

    def transform(self, X, y=None):

        check_is_fitted(self)

        # Need not validate X again as it would have already been validated
        # in the Imputer calling MissingIndicator
        if not self._precomputed:
            X = self._validate_input(X, in_fit=False)
        else:
            if not (hasattr(X, "dtype") and X.dtype.kind == "b"):
                raise ValueError("precomputed is True but the input data is not a mask")

        imputer_mask, _ = self._get_missing_features_info(X)
        imputer_mask = imputer_mask[:, self.features_]

        return imputer_mask

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        imputer_mask = self.mask

        imputer_mask = imputer_mask[:, self.features_]

        return imputer_mask


class Categorical_MIM(MIM):
    def __init__(self, *, missing_values=np.nan, features="missing-only", alpha=0.05):

        # self.mim_features = features
        # self.alpha=alpha

        # if features == "dynamic":
        #     features = "missing-only"

        super().__init__(
            alpha=alpha,
            missing_values=missing_values,
            features=features,
        )

    def fit(self, X, y=None):
        return super().fit(X, y)

    def transform(self, X, y=None):
        check_is_fitted(self)

        # Need not validate X again as it would have already been validated
        # in the Imputer calling MissingIndicator
        if not self._precomputed:
            X = self._validate_input(X, in_fit=False)
        else:
            if not (hasattr(X, "dtype") and X.dtype.kind == "b"):
                raise ValueError("precomputed is True but the input data is not a mask")

        imputer_mask, features = self._get_missing_features_info(X)

        if self.mim_features == "missing-only":
            features_diff_fit_trans = np.setdiff1d(features, self.features_)
            if self.error_on_new and features_diff_fit_trans.size > 0:
                raise ValueError(
                    "The features {} have missing values "
                    "in transform but have no missing values "
                    "in fit.".format(features_diff_fit_trans)
                )

        if self.mim_features == "dynamic":

            if type(X) != pd.DataFrame:
                X = pd.DataFrame(X)

            X.iloc[:, self.features_] = np.where(
                X.iloc[:, self.features_].isna(),
                "UNK",
                X.iloc[:, self.features_],
            )

            no_mim_features = list(set(range(X.shape[1])) - set(self.features_))
            X.iloc[:, no_mim_features] = np.where(
                X.iloc[:, no_mim_features].isna(),
                mode(X.iloc[:, no_mim_features]).mode,
                X.iloc[:, no_mim_features],
            )

        elif self.mim_features == "missing-only":

            if type(X) != pd.DataFrame:
                X = pd.DataFrame(X)

            X = np.where(
                X.isna(),
                "UNK",
                X,
            )

        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        X = X.copy()

        if self.mim_features == "dynamic":

            if type(X) != pd.DataFrame:
                X = pd.DataFrame(X)

            X.iloc[:, self.features_] = np.where(
                X.iloc[:, self.features_].isna(),
                "UNK",
                X.iloc[:, self.features_],
            )

            no_mim_features = list(set(range(X.shape[1])) - set(self.features_))
            X.iloc[:, no_mim_features] = np.where(
                X.iloc[:, no_mim_features].isna(),
                mode(X.iloc[:, no_mim_features]).mode,
                X.iloc[:, no_mim_features],
            )

            X = X.to_numpy()

        elif self.mim_features == "missing-only":

            if type(X) != pd.DataFrame:
                X = pd.DataFrame(X)

            X = np.where(
                X.isna(),
                "UNK",
                X,
            )

        return X
