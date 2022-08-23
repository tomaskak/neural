from neural.util.exp_replay import ExpReplayShared, SharedBuffers
import numpy as np
from multiprocessing import Process
from unittest.mock import Mock, patch
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


@patch("neural.util.exp_replay.np.random.default_rng")
class TestExpReplayShared:
    def test_read_write(self, rng_mock):
        # All integers calls will select the second partition
        mock = Mock()
        mock.integers = Mock(return_value=1)
        rng_mock.return_value = mock

        bufs = ExpReplayShared(SharedBuffers(6, 2, "f", [2, 1]))
        # Write over the same buffer twice
        for i in range(3 * 2):
            k = i * 3
            bufs.push(
                [
                    np.array([k, k + 1], dtype=np.float32),
                    np.array([k + 2], dtype=np.float32),
                ]
            )

        mock.integers = Mock(side_effect=([1, [2, 0, 1]]))
        sample = bufs.sample(3)
        print(f"sample={sample}")

        twos = sample[0]
        ones = sample[1]

        for i in range(3):
            assert twos[i, :].tolist() in ([9, 10], [12, 13], [15, 16])
            assert ones[i, 0] in (11, 14, 17)
