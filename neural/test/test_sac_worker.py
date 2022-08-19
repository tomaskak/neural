from neural.algos.sac_worker import SACWorker
from neural.util.process_orchestrator import (
    WorkerSpace,
    init_pool_wrap,
    ProcessOrchestrator,
    Work,
)
from neural.model.model import NormalModel

from copy import deepcopy
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import torch

import pytest


class TestSACWorker:
    def test_calculate_size(self):
        IN = 3
        OUT = 2
        BATCH = 10

        worker = SACWorker(IN, OUT, BATCH)

        FLOAT_32_SIZE = 4
        # -1 IN + OUT for simblified 2-stage process doesn't need new-state-actions stored.
        # -1 1 for target_vs which can be calculated in the same process as the value updater.
        assert (
            worker.shared_memory_size
            == ((BATCH * (7 + OUT)) + (BATCH * (2 + 3 * IN + 2 * OUT))) * FLOAT_32_SIZE
        )

    def test_init_worker_space(self):
        IN = 3
        OUT = 2
        BATCH = 10
        worker = SACWorker(IN, OUT, BATCH)

        init = worker.init_shared_worker_space(
            num_workers=2,
            actor={"model": "a", "optim": "oa"},
            q_1={"model": "1", "optim": "o1"},
            q_2={"model": "2", "optim": "o2"},
            value={"model": "v", "optim": "ov"},
            target={"model": "t"},
        )
        assert init.num_workers == 2
        assert init.space_needed == worker.shared_memory_size
        assert init.defs == [
            ("mb:states", np.float32, (BATCH, IN)),
            ("mb:next_states", np.float32, (BATCH, IN)),
            ("mb:state_actions", np.float32, (BATCH, (OUT + IN))),
            ("mb:actions", np.float32, (BATCH, OUT)),
            ("mb:rewards", np.float32, (BATCH, 1)),
            ("mb:dones", np.float32, (BATCH, 1)),
            ("actor_forward", np.float32, (BATCH, OUT)),
            ("q_1_forward", np.float32, (BATCH, 1)),
            ("q_2_forward", np.float32, (BATCH, 1)),
            ("value_forward", np.float32, (BATCH, 1)),
            ("target_value_forward", np.float32, (BATCH, 1)),
            ("log_probs", np.float32, (BATCH,)),
            ("new_qs", np.float32, (BATCH, 1)),
            ("target_qs", np.float32, (BATCH, 1)),
        ]
        for elem in [
            ("model:actor", "a"),
            ("model:q_1", "1"),
            ("model:q_2", "2"),
            ("model:value", "v"),
            ("model:target", "t"),
            ("optim:actor", "oa"),
            ("optim:q_1", "o1"),
            ("optim:q_2", "o2"),
            ("optim:value", "ov"),
        ]:
            assert elem in init.args

        for event_name in [
            "new_qs_done",
            "q_1_done",
            "q_2_done",
            "value_and_target_done",
        ]:
            assert event_name in dict(init.args)

    def test_sychronization(self):
        IN = 3
        OUT = 2
        BATCH = 10
        worker = SACWorker(IN, OUT, BATCH)

        # make models
        actor = NormalModel("actor", [torch.nn.Linear(IN, OUT * 2)])
        q_1 = torch.nn.Sequential(torch.nn.Linear(IN + OUT, 1))
        q_2 = torch.nn.Sequential(torch.nn.Linear(IN + OUT, 1))
        value = torch.nn.Sequential(torch.nn.Linear(IN, 1))
        target = deepcopy(value)

        copy_actor = deepcopy(actor)
        copy_q_1 = deepcopy(q_1)
        copy_q_2 = deepcopy(q_2)
        copy_value = deepcopy(value)
        copy_target = deepcopy(target)

        actor.share_memory()
        q_1.share_memory()
        q_2.share_memory()
        value.share_memory()
        target.share_memory()

        actor_optim = torch.optim.Adam(actor.parameters(), 0.001)
        q_1_optim = torch.optim.Adam(q_1.parameters(), 0.001)
        q_2_optim = torch.optim.Adam(q_2.parameters(), 0.001)
        value_optim = torch.optim.Adam(value.parameters(), 0.001)

        # make minibatch
        states = torch.randn(BATCH, IN)
        next_states = torch.randn(BATCH, IN)
        actions = torch.randn(BATCH, OUT)
        state_actions = torch.cat((states, actions), 1)
        rewards = torch.randn(BATCH, 1)
        dones = torch.randn(BATCH, 1)

        # init worker and process orchestrator (inits WorkerSpace)
        device = "cpu"
        worker = SACWorker(IN, OUT, BATCH)
        init = worker.init_shared_worker_space(
            num_workers=5,
            actor={"model": actor, "optim": actor_optim},
            q_1={"model": q_1, "optim": q_1_optim},
            q_2={"model": q_2, "optim": q_2_optim},
            value={"model": value, "optim": value_optim},
            target={"model": target},
        )
        orchestrator = ProcessOrchestrator(
            init,
            [
                Work(provides="actor", fn=SACWorker.actor, args=(device,)),
                Work(provides="q_1", fn=SACWorker.q, args=("1", 0.001, device)),
                Work(provides="q_2", fn=SACWorker.q, args=("2", 0.001, device)),
                Work(provides="value", fn=SACWorker.value, args=(0.001, device)),
            ],
        )

        # load minibatch
        data = WorkerSpace.data
        data["mb:states"][:] = states[:]
        data["mb:next_states"][:] = next_states[:]
        data["mb:state_actions"][:] = state_actions[:]
        data["mb:actions"][:] = actions[:]
        data["mb:rewards"][:] = rewards[:]
        data["mb:dones"][:] = dones[:]

        assert torch.equal(states, torch.from_numpy(data["mb:states"]))
        assert torch.equal(next_states, torch.from_numpy(data["mb:next_states"]))
        assert torch.equal(state_actions, torch.from_numpy(data["mb:state_actions"]))
        assert torch.equal(actions, torch.from_numpy(data["mb:actions"]))
        assert torch.equal(rewards, torch.from_numpy(data["mb:rewards"]))
        assert torch.equal(dones, torch.from_numpy(data["mb:dones"]))

        # execute
        success, output = orchestrator.execute()
        assert success

        # expect all models updated
        def any_difference(ones, twos):
            for one, two in zip(ones, twos):
                if torch.any(torch.ne(one, two)):
                    return True
            return False

        with torch.no_grad():
            assert any_difference(copy_actor.parameters(), actor.parameters())
            assert any_difference(copy_q_1.parameters(), q_1.parameters())
            assert any_difference(copy_q_2.parameters(), q_2.parameters())
            assert any_difference(copy_value.parameters(), value.parameters())
            assert any_difference(copy_target.parameters(), target.parameters())
