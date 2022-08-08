from abc import ABC, abstractmethod


class Algo(ABC):
    """
    Base class for an RL algorithm.

    This class defines the interface for an learning algorithm which will define the
    way a model is trained by interacting with an environment.
    """

    def __init__(self, hypers: list, training_params: dict, env):
        self._hypers = hypers
        self._training_params = training_params
        self._env = env

        for key, param in hypers.items():
            if key not in self.defined_hyperparams() or not isinstance(
                param, self.defined_hyperparams()[key]
            ):
                raise ValueError(f"Bad hyperparameter value for {key}")

    @property
    def hyperparams(self):
        return self._hypers

    @property
    def training_params(self):
        return self._training_params

    @property
    def env(self):
        return self._env

    @classmethod
    @abstractmethod
    def defined_hyperparams(cls) -> dict:
        """
        Returns a dict of str -> type of hyperparameters that can be
        present in the hypers argument initializing this class.
        """

    @abstractmethod
    def train(self):
        """
        Perform one iteration of training.
        """

    @abstractmethod
    def test(self):
        """
        Test the performance of the model.
        """
