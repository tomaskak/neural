from neural.algos.sac_replay import sac_replay
from unittest import TestCase
from unittest.mock import MagicMock, patch, call, ANY
from queue import Queue
from copy import deepcopy

import time
import torch
import numpy as np
import pytest


def make_nd_array(size: int):
    return np.ndarray(shape=(size,), dtype=np.float32)


IN_SIZE = 3
OUT_SIZE = 2
MAX_Q = 3


@pytest.mark.timeout(0.5)
class TestSACReplay(TestCase):
    def setUp(self):
        self._push_batch_q = Queue()
        self._hypers = {"experience_replay_size": 1000, "minibatch_size": 2}
        self._new_data_q = Queue()

        self._data = [
            make_nd_array(IN_SIZE),
            make_nd_array(OUT_SIZE),
            make_nd_array(1),
            make_nd_array(IN_SIZE),
            make_nd_array(1),
        ]

    def call(self):
        sac_replay(
            self._new_data_q,
            self._push_batch_q,
            self._hypers,
            in_size=IN_SIZE,
            out_size=OUT_SIZE,
            push_q_length=MAX_Q,
            device="cpu",
        )

    def _push_4_data(self):
        # Push enough data to reach minibatch sampling which is 2x minibatch size
        self._new_data_q.put(("EXP", deepcopy(self._data)))
        self._new_data_q.put(("EXP", deepcopy(self._data)))
        self._new_data_q.put(("EXP", deepcopy(self._data)))
        self._new_data_q.put(("EXP", deepcopy(self._data)))

    def test_sac_will_stop_loop(self):
        self._new_data_q.put("STOP")
        self.call()

    def test_deletes_done_data(self):
        self._push_4_data()

        self._new_data_q.put(("DONE", 0))
        self._new_data_q.put("STOP")

        self.call()

        assert self._push_batch_q.qsize() == 1
        batch = self._push_batch_q.get()

        assert len(batch) == 6

        for part in batch:
            assert part.shape[0] == 2
            assert part.dtype == torch.float32

        for i in range(6):
            if i == 5:
                state_actions = torch.cat(
                    (torch.tensor(self._data[0]), torch.tensor(self._data[1])), 0
                )
                assert torch.equal(batch[i][0], state_actions)
            else:
                assert torch.equal(batch[i][0], torch.tensor(self._data[i]))

    def test_queue_is_full_no_additional_pushes(self):
        # 3 elements to fill the q
        self._push_batch_q.put({})
        self._push_batch_q.put({})
        self._push_batch_q.put({})

        self._push_4_data()

        self._new_data_q.put("STOP")

        self.call()

        assert self._push_batch_q.qsize() == 3
