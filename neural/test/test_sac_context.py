from pickle import loads, dumps
from neural.algos.sac_context import SACContext
from neural.model.model import NormalModel
from neural.model.layer import ActivatedLayer
from torch.nn import Tanh, MSELoss
from torch.optim import Adam

import pytest


def test_can_pickle_context():
    context = SACContext()

    context.actor = NormalModel("actor", [ActivatedLayer("l1", 100, 200, Tanh(), True)])
    context.actor_optim = Adam(context.actor.parameters(), 0.001)
    context.actor_loss_fn = MSELoss()

    output = dumps(context)
    restored = loads(output)
