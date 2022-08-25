from neural.util.exp_replay import ExpReplayShared, SharedBuffers
import numpy as np
from multiprocessing import Process
from unittest.mock import Mock, patch, call
import pytest


def write_over_buf(bufs, elem):
    i, max_elem, buf = bufs.buffers[0]
    buf[0] = elem


class TestSharedBuffers:
    def test_can_pickle(self):
        bufs = SharedBuffers(6, 2, "f", [2, 1, 3])

        for index, max_elem, b in bufs.buffers:
            assert len(b) == 18
            assert index.value == 0
            assert max_elem.value == 0

        i, max_elem, buf = bufs.buffers[0]
        i.value = 2
        max_elem.value = 1
        buf[0:3] = [0.345, 0.666, 0.123]

        p = Process(target=write_over_buf, args=(bufs, 0.555))
        p.start()
        p.join()

        # Without conversion to float32 on both sides it will compare beyond the true
        # data size.
        assert np.float32(buf[0]) == np.float32(0.555)


@pytest.fixture
def rng_mock():
    with patch("neural.util.exp_replay.np.random.default_rng") as m:
        yield m


class TestExpReplayShared:
    @pytest.fixture(autouse=True)
    def setup(self, rng_mock):
        SECOND_BUF = 1
        self._mock = Mock()
        self._mock.integers = Mock(return_value=SECOND_BUF)
        rng_mock.return_value = self._mock
        self.make_replay()

    def make_replay(self):
        self._replay = ExpReplayShared(SharedBuffers(6, 2, "f", [2, 1]))

    @pytest.fixture
    def second_buffer_loaded(self, setup):
        self.load_active_buffer_n_times(2)

    def load_active_buffer_n_times(self, n):
        for i in range(3 * 2):
            k = i * 3
            self._replay.push(
                [
                    np.array([k, k + 1], dtype=np.float32),
                    np.array([k + 2], dtype=np.float32),
                ]
            )

    def test_read_write(self, second_buffer_loaded):
        self._mock.integers = Mock(side_effect=([1, [2, 0, 1]]))
        sample = self._replay.sample(3)

        twos = sample[0]
        ones = sample[1]

        for i in range(3):
            assert twos[i, :].tolist() in ([9, 10], [12, 13], [15, 16])
            assert ones[i, 0] in (11, 14, 17)

    def test_read_write_past_max_elem(self, second_buffer_loaded):
        # Load buffer more times then elem size to check that max_elem does not
        # extend beyond buffer length and cause a sample to reach beyond mem length.
        self.load_active_buffer_n_times(10)

        self._mock.integers = Mock(side_effect=([1, [2, 0, 1]]))
        sample = self._replay.sample(3)

        self._mock.integers.assert_has_calls([call(0, 2, None), call(0, 3, size=(3,))])

    def test_empty_partitions_not_used_for_sampling(self, second_buffer_loaded):
        self._mock.integers = Mock(side_effect=([0, 1, [2, 0, 1]]))
        sample = self._replay.sample(3)

        self._mock.integers.assert_has_calls(
            [call(0, 2, None), call(0, 2, None), call(0, 3, size=(3,))]
        )

    def test_sampling_from_all_empty_buffers_raises(self):
        # No patches or mocks or data loading results in the class choosing buffers
        # at random and finding them all empty.
        self.make_replay()

        with pytest.raises(RuntimeError):
            sample = self._replay.sample(3)
