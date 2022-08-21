from pickle import loads, dumps
from neural.algos.sac_context import SACContext
from neural.model.model import NormalModel
from neural.model.layer import ActivatedLayer
from torch.nn import Tanh, MSELoss
from torch.optim import Adam
from copy import deepcopy

import torch
import pytest


def test_can_pickle_context():
    context = SACContext()

    context.actor = NormalModel("actor", [ActivatedLayer("l1", 100, 200, Tanh(), True)])
    context.actor_optim = Adam(context.actor.parameters(), 0.001)
    context.actor_loss_fn = MSELoss()

    output = dumps(context)
    restored = loads(output)


def test_can_pickle_and_share():
    context = SACContext()

    context.actor = NormalModel("actor", [ActivatedLayer("l1", 100, 200, Tanh(), True)])
    context.actor_optim = Adam(context.actor.parameters(), 0.001)
    context.actor_loss_fn = MSELoss()

    context.update_shared()

    context.shared.actor.share_memory()

    pickled = dumps(context)
    restored = loads(pickled)

    for param in restored.actor.parameters():
        param = 2.0 * param

    restored.shared.actor = restored.actor

    for orig, new in zip(
        context.shared.actor.parameters(), restored.actor.parameters()
    ):
        assert torch.equal(orig, new)
