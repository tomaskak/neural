from neural.algos.sac_worker import SACWorker
from neural.util.process_orchestrator import WorkerSpace, init_pool_wrap

from multiprocessing.shared_memory import SharedMemory
import numpy as np

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
            num_workers=2, actor="a", q_1="1", q_2="2", value="v", target="t"
        )
        assert init.num_workers == 2
        assert init.space_needed == worker.shared_memory_size
        assert init.defs == [
            ("mb:states", np.float32, BATCH * IN),
            ("mb:next_states", np.float32, BATCH * IN),
            ("mb:state_actions", np.float32, BATCH * (OUT + IN)),
            ("mb:actions", np.float32, BATCH * OUT),
            ("mb:rewards", np.float32, BATCH),
            ("mb:dones", np.float32, BATCH),
            ("actor_forward", np.float32, BATCH * OUT),
            ("q_1_forward", np.float32, BATCH),
            ("q_2_forward", np.float32, BATCH),
            ("value_forward", np.float32, BATCH),
            ("target_value_forward", np.float32, BATCH),
            ("log_probs", np.float32, BATCH),
            ("new_qs", np.float32, BATCH),
            ("target_qs", np.float32, BATCH),
        ]
        assert init.args == [
            ("model:actor", "a"),
            ("model:q_1", "1"),
            ("model:q_2", "2"),
            ("model:value", "v"),
            ("model:target", "t"),
        ]
