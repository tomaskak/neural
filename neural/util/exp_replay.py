import torch
import numpy as np
from json import dump

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


class ExpReplayTorch:
    """
    Takes a list of sizes and types which will be used to build tensors that hold
    N*size of type to represent an element.

    Slower than the non-torch one.
    """

    def __init__(self, N: int, defs: list, device="cpu"):
        assert isinstance(N, int) and N > 0, "bad N"
        assert isinstance(defs, list), "bad defs"

        self._buffers = []

        self._N = N
        for elem_def in defs:
            size, dtype = elem_def
            assert size > 0

            self._buffers.append(
                (elem_def, torch.zeros((N * size,), dtype=dtype, device=device))
            )

        self._idx = 0
        self._max_elem = 0

    def push(self, values: list):
        idx = self._idx

        for (buf_def, buf), val in zip(self._buffers, values):
            sz, dtype = buf_def
            if sz != 1:
                for i, elem in enumerate(val):
                    buf[idx * sz + i] = elem
            else:
                buf[idx] = val

        self._idx += 1

        if self._idx > self._max_elem:
            self._max_elem = self._idx
        if self._idx >= self._N - 1:
            self._idx = 0

    def __len__(self):
        return self._max_elem

    def sample(self, sample_size):
        assert sample_size > 0 and sample_size <= (self._max_elem + 1)
        sample = [
            torch.zeros_like(
                buf[: buf_size * sample_size].view(sample_size, -1)
            ).float()
            for (buf_size, dtype), buf in self._buffers
        ]

        indexes = np.random.choice(
            self._max_elem - 1, size=(sample_size,), replace=False
        )

        for count, i in enumerate(indexes):
            for s, (buf_def, buf) in zip(sample, self._buffers):
                buf_size, dtype = buf_def
                slice = buf[i * buf_size : (i + 1) * buf_size]
                if buf_size > 1:
                    for k, elem in enumerate(slice):
                        s[count][k] = elem
                else:
                    s[count] = slice[0]
        return sample
