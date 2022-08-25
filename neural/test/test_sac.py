from neural.algos.sac_mp import SoftActorCritic

from gym import make

import pytest


def env():
    return make("Acrobot-v1")


def init_sac(hypers=list()):
    layers = {
        "actor": [
            ("l1", "tanh", "input", "input*4"),
            ("l2", "ReLU", "input*4", "output"),
        ],
        "q_1": [
            ("l1", "tanh", "input + output", "input*4"),
            ("l2", "ReLU", "input*4", 1),
        ],
        "q_2": [
            ("l1", "tanh", "input + output", "input*4"),
            ("l2", "ReLU", "input*4", 1),
        ],
    }
    return SoftActorCritic(hypers, layers, {}, env())


class TestSoftActorCritic:
    def test_algo_build_success(self):
        s = init_sac(hypers={"future_reward_discount": 1.0, "target_update_step": 0.1})

    def test_algo_build_non_existant_hyper(self):
        with pytest.raises(ValueError):
            s = init_sac(hypers={"fake": 0.1})

    def test_algo_build_bad_type_hyper(self):
        with pytest.raises(ValueError):
            s = init_sac(hypers={"target_update_step": "string"})
