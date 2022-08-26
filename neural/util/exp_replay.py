import torch
import numpy as np
from multiprocessing import Array, Value
from array import typecodes, array
from collections.abc import Iterable
from abc import ABC, abstractmethod

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


class Buffers(ABC):
    def __init__(
        self, size: int, partitions: int, dtype: str = "f", elem_parts: list = list([1])
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

    def __getitem__(self, index: int):
        return self._buffers[index]


class OneSimpleBuffer:
    def __init__(self, dtype: str, default_value, size: int):
        self._max_elem = -1
        self._index = 0
        self._buffer = array(dtype, [default_value for _ in range(size)])

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, index: int | slice):
        return self._buffer[index]

    def __setitem__(self, index: int | slice, new_value):
        self._buffer[index] = new_value

    @property
    def max_elem(self):
        return self._max_elem

    @max_elem.setter
    def max_elem(self, new_value):
        self._max_elem = new_value

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, new_value):
        self._index = new_value


class OneSharedBuffer:
    def __init__(self, dtype: str, size: int):
        self._max_elem = Value("Q", lock=True)
        self._index = Value("Q", lock=True)
        self._buffer = Array(dtype, size, lock=True)
        self._context_held = False

    def __enter__(self):
        self._context_held = True
        self._max_elem.get_lock().acquire()
        self._index.get_lock().acquire()
        return self

    def __exit__(self, *args, **kwargs):
        self._context_held = False
        self._max_elem.get_lock().release()
        self._index.get_lock().release()

    def __len__(self):
        return len(self._buffer)

    def _assert_locks_held(self):
        assert (
            self._context_held
        ), "Attempting to access synchronized properties of a shared buffer without holding locks by entering object's context"

    def __getitem__(self, index: int | slice):
        return self._buffer[index]

    def __setitem__(self, index: int | slice, new_value):
        self._buffer[index] = new_value

    @property
    def max_elem(self):
        self._assert_locks_held()
        return self._max_elem.value

    @max_elem.setter
    def max_elem(self, new_value):
        self._assert_locks_held()
        self._max_elem.value = new_value

    @property
    def index(self):
        self._assert_locks_held()
        return self._index.value

    @index.setter
    def index(self, new_value):
        self._assert_locks_held()
        self._index.value = new_value


class SharedBuffers(Buffers):
    def __init__(
        self, size: int, partitions: int, dtype:str="f", elem_parts: list = list([1])
    ):
        super().__init__(size, partitions, dtype, elem_parts)
        # contains (idx, max_elem, buffer) for each partition
        self._buffers = [
            OneSharedBuffer(dtype, self._part_size * self._elem_size)
            for _ in range(partitions)
        ]


def read_data(path:str, dtype:str, columns:tuple[int,int]):
    data = np.loadtxt(path,dtype, usecols=list(range(columns[0],columns[1])), delimiter=',')
    print(f"data read = {data}")
    return data
        
class StaticBuffersFromFile(Buffers):
    def __init__(self, path:str,  partitions: int, dtype:str="f", elem_parts: list=list([1]), columns:tuple[int,int]=None):
        print(f"Reading data from {path} using columns {columns}")
        data = read_data(path, dtype, columns)
        print(f"{len(data)} rows read from {path}")
        super().__init__(len(data), partitions, dtype, elem_parts)

        print(f"elem_parts={elem_parts}, size={self.size}, elem_size={self.item_size}")

        self._buffers = [
            OneSimpleBuffer(dtype, 0.0, self._part_size * self._elem_size)
            for _ in range(partitions)
            ]


        i = 0
        for row in data:
            for k, elem in enumerate(row[:]):
                self._buffers[0][i*len(row)+k] = elem
            i += 1
        self._buffers[0].index = i*len(data[0])
        self._buffers[0].max_elem = i

class SplitExpReplayReader:
    def __init__(self, buffers_one: Buffers, buffers_two: Buffers, percent_of_one: float, decrement: float=None):
        self._buffers_one = ExpReplayReader(buffers_one)
        self._buffers_two = ExpReplayReader(buffers_two)
        self._pct_one = percent_of_one
        self._decrement = decrement

    def sample(self, sample_size: int):
        from_one = max((sample_size * self._pct_one) // 1, 1)
        from_two = sample_size - from_one

        sample_one = self._buffers_one.sample(int(from_one))
        sample_two = self._buffers_two.sample(int(from_two))

        self._pct_one -= self._decrement

        print(f"sample_one={sample_one}, sample_two={sample_two}")
        return [np.concatenate(one, two) for one, two in zip(sample_one, sample_two)]

class ExpReplayCore:
    def __init__(self, buffers: Buffers):
        assert buffers is not None

        self._buffers = buffers
        self._generator = np.random.default_rng()

    def _random_part(self, sample_shape=None) -> int:
        return self._generator.integers(0, self._buffers.partitions, sample_shape)

    def __len__(self):
        return self._buffers.size


class ExpReplayReader(ExpReplayCore):
    def __init__(self, buffers: Buffers):
        super().__init__(buffers)

    def _zeros(self, sample_size):
        zeros = []
        for part in self._buffers.item_parts:
            zeros.append(np.zeros(shape=(sample_size, part), dtype=np.float32))
        return zeros

    def sample(self, sample_size: int):
        assert sample_size > 0

        sample = self._zeros(sample_size)

        buf = None
        if self._buffers.partitions == 1:
            buf = 0
            with self._buffers[buf] as buffer:
                assert (
                    buffer.max_elem > sample_size
                ), f"Attempting to sample {sample_size} items from a buffer of only length {buffer.max_elem}"
        else:
            # Selected buffer must have enough elements for the sample size
            done = False
            attempts = 0
            while not done and attempts < self._buffers.partitions * 2:
                attempts += 1
                buf = self._random_part()
                with self._buffers[buf] as buffer:
                    if buffer.max_elem > sample_size // 4:
                        done = True
            if not done:
                raise RuntimeError(
                    f"{attempts} attempts made to get a buffer of adequate size for a sample of {sample_size} with no successful candidates."
                )

        with self._buffers[buf] as buffer:
            indexes = self._generator.integers(
                0,
                min(buffer.max_elem, (len(buffer) / self._buffers.item_size)),
                size=(sample_size,),
            )
            indexes = indexes if sample_size > 1 else [indexes]

            for count, i in enumerate(indexes):
                offset = i * self._buffers.item_size
                for part, sz in zip(sample, self._buffers.item_parts):
                    part[count, :] = buffer[offset : offset + sz]
                    offset += sz
            return sample


class ExpReplayWriter(ExpReplayCore):
    def __init__(self, buffers: Buffers):
        super().__init__(buffers)

    def push(self, values: list):
        with self._buffers[self._random_part()] as buffer:
            i = buffer.index
            for val in values:
                if hasattr(val, "__iter__"):
                    buffer[i : i + len(val)] = val[:]
                    i += len(val)
                else:
                    buffer[i] = val
                    i += 1
            buffer.max_elem += 1
            buffer.index = 0 if i >= len(buffer) else i
