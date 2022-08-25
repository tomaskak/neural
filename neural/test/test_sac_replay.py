from neural.algos.sac_replay import sac_replay_store, sac_replay_sample
from neural.util.exp_replay import ExpReplayWriter, SharedBuffers
from unittest import TestCase
from unittest.mock import Mock, patch, call, ANY
from torch.multiprocessing import Queue
from copy import deepcopy

import time
import torch
import numpy as np
import pytest


def make_nd_array(size: int, value: int = np.random.default_rng().integers(0, 100)):
    out = np.ndarray(shape=(size,), dtype=np.float32)
    out[:] = value
    return out


IN_SIZE = 3
OUT_SIZE = 2
MAX_Q = 3


@pytest.mark.timeout(0.5)
@patch("neural.util.exp_replay.np.random.default_rng")
class TestSACReplayStore(TestCase):
    def test_pushes_data_to_buffers(self, rng_mock):
        BUF_ID = 4
        mock = Mock()
        mock.integers = Mock(return_value=BUF_ID)
        rng_mock.return_value = mock

        self._exp_q = Queue()
        self._data = [
            make_nd_array(IN_SIZE),
            make_nd_array(OUT_SIZE),
            make_nd_array(1),
            make_nd_array(IN_SIZE),
            make_nd_array(1),
        ]

        shared = SharedBuffers(
            100, 10, dtype="f", elem_parts=[IN_SIZE, OUT_SIZE, 1, IN_SIZE, 1]
        )

        self._exp_q.put(("EXP", self._data))
        self._exp_q.put("STOP")

        sac_replay_store(self._exp_q, shared)
        with shared[BUF_ID] as buf:
            assert np.concatenate(self._data).tolist() == buf[0 : shared.item_size]


@pytest.fixture
def mock_rng():
    with patch("neural.util.exp_replay.np.random.default_rng") as rng_mock:
        yield rng_mock


@pytest.mark.timeout(0.5)
class TestSACReplaySample(TestCase):
    @pytest.fixture(autouse=True)
    def set_up(self, mock_rng):
        self._BUF_ID = 4
        self._OTHER_ID = 0
        self._DATA_VAL = 6
        mock = Mock()
        mock.integers = Mock(return_value=self._BUF_ID)
        self._rng_mock = mock

        mock_rng.return_value = self._rng_mock

        self._batch_q = Queue()
        self._returns_q = Queue()
        self._data = [
            make_nd_array(IN_SIZE, self._DATA_VAL),
            make_nd_array(OUT_SIZE, self._DATA_VAL),
            make_nd_array(1, self._DATA_VAL),
            make_nd_array(IN_SIZE, self._DATA_VAL),
            make_nd_array(1, self._DATA_VAL),
        ]
        self._hypers = {}
        self._hypers["minibatch_size"] = 3
        self._start, self._stop = (Mock(), Mock())
        self._start.wait.return_value = True
        self._stop.is_set.side_effect = [False, False, True]

        self._shared = SharedBuffers(
            100, 10, dtype="f", elem_parts=[IN_SIZE, OUT_SIZE, 1, IN_SIZE, 1]
        )

        self._replay = ExpReplayWriter(self._shared)
        for _ in range(10):
            self._replay.push(self._data)

    def call(self):
        sac_replay_sample(
            self._batch_q,
            self._returns_q,
            self._start,
            self._stop,
            self._hypers,
            self._shared,
            "cpu",
        )

    def assert_batch_q_elems_match_expected(self, elems):
        for cmd, (batch_id, batch) in elems:
            assert cmd == "PROCESS"
            assert batch_id in (0, 1)
            for given, expected in zip(
                batch, self._shared.item_parts + [IN_SIZE + OUT_SIZE]
            ):
                assert given.shape == (self._hypers["minibatch_size"], expected)
                assert given.device == torch.device("cpu")
                assert torch.equal(given[:], torch.ones_like(given) * self._DATA_VAL)

    def test_starts_stops_on_events(self):
        self._stop.is_set.side_effect = [True]

        self.call()

        self._start.wait.assert_called_once()
        self._stop.is_set.call_count == 3

    def test_pushes_data(self):
        self._rng_mock.integers = Mock(
            side_effect=[self._BUF_ID, [3, 0, 9], self._BUF_ID, [2, 6, 4]]
        )

        self.call()

        assert self._batch_q.qsize() == 2
        self.assert_batch_q_elems_match_expected(
            [self._batch_q.get(), self._batch_q.get()]
        )

    def test_reuses_returned_buffers(self):
        # Create a message indicating it is safe to reuse the 0-id buffers
        self._returns_q.put(("DONE", 0))

        # Populate a second buffer
        self._rng_mock.integers = Mock(return_value=self._OTHER_ID)
        other_data = [
            make_nd_array(IN_SIZE, 4),
            make_nd_array(OUT_SIZE, 4),
            make_nd_array(1, 4),
            make_nd_array(IN_SIZE, 4),
            make_nd_array(1, 4),
        ]
        for _ in range(10):
            self._replay.push(self._data)

        # The initial sample creation will use OTHER_ID but this will be rewritten by the return
        # reuse action.
        self._rng_mock.integers = Mock(
            side_effect=[self._OTHER_ID, [3, 0, 9], self._BUF_ID, [2, 6, 4]]
        )

        self.call()

        assert self._batch_q.qsize() == 2
        item_one = self._batch_q.get()
        item_two = self._batch_q.get()
        _, (_, batch_one) = item_one
        _, (_, batch_two) = item_two
        for one, two in zip(batch_one, batch_two):
            assert one.data_ptr() == two.data_ptr()

        self.assert_batch_q_elems_match_expected([item_one, item_two])
