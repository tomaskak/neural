from neural.algos.sac_train import sac_train
from unittest import TestCase
from unittest.mock import MagicMock, patch, call, ANY
from queue import Queue

import pytest

# The timeout is because sac_train is a loop so any test has the potential of being stuck
@pytest.mark.timeout(0.5)
@patch("neural.algos.sac_train.process_batch")
class TestSACTrain(TestCase):
    def setUp(self):
        self._models_optims = MagicMock()
        self._next_batch_q = Queue()
        self._hypers = {}
        self._done_queue = Queue()
        self._report_queue = Queue()

    def call(self, steps_to_report=100):
        sac_train(
            self._models_optims,
            self._next_batch_q,
            self._hypers,
            steps_to_report,
            self._report_queue,
            self._done_queue,
        )

    def test_sac_will_stop_loop(self, process_mock):
        self._next_batch_q.put("STOP")
        self.call()

    def test_sac_will_report_after_N_steps(self, process_mock):
        N = 5
        for i in range(5):
            self._next_batch_q.put(("PROCESS", (i, {})))
        self._next_batch_q.put("STOP")

        self.call(steps_to_report=N)

        assert not self._report_queue.empty()
        assert {"completed": 5} == self._report_queue.get()

    def test_batch_ids_passed_to_done_q(self, process_mock):
        for i in range(5):
            self._next_batch_q.put(("PROCESS", (i, {})))
        self._next_batch_q.put("STOP")

        self.call()

        for i in range(5):
            assert not self._done_queue.empty(), i
            assert ("DONE", i) == self._done_queue.get(), i

    def test_batches_passed_to_process_fn(self, process_mock):
        for i in range(5):
            self._next_batch_q.put(("PROCESS", (i, {"key": str(i)})))
        self._next_batch_q.put("STOP")

        self.call()

        process_mock.assert_has_calls(
            [
                call(self._models_optims, self._hypers, {"key": "0"}, False),
                call(self._models_optims, self._hypers, {"key": "1"}, False),
                call(self._models_optims, self._hypers, {"key": "2"}, False),
                call(self._models_optims, self._hypers, {"key": "3"}, False),
                call(self._models_optims, self._hypers, {"key": "4"}, False),
            ]
        )
