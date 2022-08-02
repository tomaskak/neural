import numpy as np


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
