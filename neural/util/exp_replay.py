import torch
import numpy as np
from multiprocessing import Array, Value
from array import typecodes

from ..tools.timer import timer


class ExpReplay:
    def __init__(self, N: int, dtypes: list):
        assert isinstance(N, int) and N > 0, "bad N"
        assert isinstance(dtypes, list), "bad dtypes"

        self._N = N
        self._buffers = []

        for dt in dtypes:
            self._buffers.append(np.zeros(N, dtype=dt))

        self._idx = 0
        self._max_elem = 0

    def push(self, values: list):
        idx = self._idx
        for buf, v in zip(self._buffers, values):
            buf[idx] = v
        self._idx += 1
        if self._idx > self._max_elem:
            self._max_elem = self._idx
        if self._idx >= self._N - 1:
            self._idx = 0

    def __len__(self):
        return self._max_elem

    def sample(self, sample_size):
        assert sample_size > 0
        sample = [np.zeros_like(buf[:sample_size]) for buf in self._buffers]

        indexes = np.random.choice(
            self._max_elem - 1, size=(sample_size,), replace=False
        )
        for count, i in enumerate(indexes):
            for s, buf in zip(sample, self._buffers):
                s[count] = buf[i]
        return sample


class SharedBuffers:
    def __init__(
        self, size: int, partitions: int, dtype="f", elem_parts: list = list([1])
    ):
        assert isinstance(size, int) and size > 0, "bad size"
        assert isinstance(partitions, int) and partitions > 0, "bad partitions"
        assert dtype in typecodes, "bad dtypes"

        self._elem_size = 0
        for part in elem_parts:
            self._elem_size += part
        self._size = (size // partitions) * partitions * self._elem_size
        self._part_size = size // partitions
        self._paritions = partitions
        self._elem_parts = elem_parts

        # contains (idx, max_elem, buffer) for each partition
        self._buffers = [
            (
                Value("Q", lock=True),
                Value("Q", -1.0, lock=True),
                Array(dtype, self._part_size * self._elem_size, lock=True),
            )
            for _ in range(partitions)
        ]

    @property
    def buffers(self):
        return self._buffers

    @property
    def size(self):
        return self._size

    @property
    def partitions(self):
        return self._paritions

    @property
    def num_items(self):
        return self._size / self._elem_size

    @property
    def item_size(self):
        return self._elem_size

    @property
    def item_parts(self):
        return self._elem_parts


class ExpReplayShared:
    def __init__(self, buffers: SharedBuffers):
        assert buffers is not None

        self._shared = buffers
        self._generator = np.random.default_rng()

    def _random_part(self, sample_shape=None) -> int:
        return self._generator.integers(0, self._shared.partitions, sample_shape)

    def push(self, values: list):
        idx, max_elem, buf = self._shared.buffers[self._random_part()]

        with idx.get_lock():
            with max_elem.get_lock():
                i = idx.value
                for val in values:
                    if hasattr(val, "__iter__"):
                        buf[i : i + len(val)] = val[:]
                        i += len(val)
                    else:
                        buf[i] = val
                        i += 1
                max_elem.value += 1
                idx.value = 0 if i >= len(buf) else i

    def __len__(self):
        return self._shared.size

    def _zeros(self, sample_size):
        zeros = []
        for part in self._shared.item_parts:
            zeros.append(np.zeros(shape=(sample_size, part), dtype=np.float32))
        return zeros

    def sample(self, sample_size):
        assert sample_size > 0
        idx, max_elem, buf = self._shared.buffers[self._random_part()]
        while max_elem.value < sample_size:
            idx, max_elem, buf = self._shared.buffers[self._random_part()]
        sample = self._zeros(sample_size)

        with max_elem.get_lock():
            indexes = self._generator.integers(
                0, min(max_elem.value - 1, len(buf) - 1), size=(sample_size,)
            )

        for count, i in enumerate(indexes):
            offset = i * self._shared.item_size
            for part, sz in zip(sample, self._shared.item_parts):
                part[count, :] = buf[offset : offset + sz]
                offset += sz
        return sample
