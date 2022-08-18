from collections.abc import Callable
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.pool import Pool
from multiprocessing import Process, Event
from time import time
from copy import deepcopy
from dill import dumps, loads

import numpy as np


class Work:
    """
    Helper class that provides a place to hold all of the required information for
    one chunk of work used in the ProcessOrchestrator.
    """

    def __init__(
        self, fn: Callable, provides: str, args: tuple = None, depends_on: str = None
    ):
        self._depends_on = depends_on
        self._provides = provides
        self._fn = fn
        self._args = (*args, provides) if args is not None else None

        assert self._provides is not None

    @property
    def depends_on(self):
        return self._depends_on

    @property
    def provides(self):
        return self._provides

    @property
    def fn(self):
        return self._fn

    @property
    def args(self):
        return self._args


class Init:
    """
    Represents the initialization information of a set of work.
    The size indicates the total buffer space required and the init_fn will be
    called on that space.
    """

    def __init__(self, num_workers: int, space_needed: int, init_defs: list):
        self._space = space_needed
        self._init_defs = init_defs
        self._num_workers = num_workers

    @property
    def space_needed(self):
        return self._space

    @property
    def num_workers(self):
        return self._num_workers

    @property
    def defs(self):
        return self._init_defs


class WorkerSpace:
    data = None
    block = None


def check_col_is_clear(matrix, col):
    for i in range(len(matrix)):
        if matrix[i][col]:
            return False
    return True


def check_row_is_clear(matrix, row):
    for j in range(len(matrix[row])):
        if matrix[row][j]:
            return False
    return True


def init_pool_wrap(shmem_name, shmem_space, init_defs):
    print(f"initializing space...")

    data = {}
    WorkerSpace.block = SharedMemory(name=shmem_name, create=False, size=shmem_space)
    block = WorkerSpace.block
    print(f"block={block}")
    print(f"block.buf={block.buf}")

    keys_are_unique = set()

    curr_offset = 0
    for key, dtype, size in init_defs:
        if key in keys_are_unique:
            raise ValueError(f"duplicate key {key}")
        keys_are_unique.add(key)
        print(
            f"adding {key} with size {size}, dtype={dtype}, starting at offset {curr_offset}"
        )
        print(
            f"reading bytes={bytes(block.buf[curr_offset:curr_offset+size*np.dtype(dtype).itemsize])}"
        )
        data[key] = np.ndarray(
            shape=(size,), dtype=dtype, buffer=block.buf, offset=curr_offset
        )
        print(f"initialized {key} data={data[key]}")
        curr_offset += np.dtype(dtype).itemsize * size

    WorkerSpace.data = data
    print(f"WorkerSpace.data={WorkerSpace.data}")
    print(f"initializing space done...")


class ProcessOrchestrator:
    """
    Orchestrates multiple processes to divide and conquer work items with dependencies.

    Uses the Init argument to create a shared memory space of the right size, then the
    initialization function to turn this space into correctly typed arrays in an easy
    to access structure (like a dict, but it is up to the caller).

    Each work item will be passed this ordered structure representing the shared memory
    space where i/o from the function will be placed.
    """

    def __init__(self, init: Init, work_defs: list, out_fn: Callable | None = None):
        _validate_work(work_defs)

        assert init.space_needed > 0

        self._adjacency_matrix, self._matrix_keys = make_adjacency_matrix(work_defs)
        self._work_items = dict([(self._matrix_keys[w.provides], w) for w in work_defs])

        self._shmem_name = str(time())
        self._shmem = SharedMemory(
            create=True, size=init.space_needed, name=self._shmem_name
        )
        self._init = init
        self._out_fn = out_fn

        self._pool = Pool(
            init.num_workers,
            initializer=init_pool_wrap,
            initargs=(self._shmem_name, init.space_needed, init.defs),
        )

        init_pool_wrap(self._shmem_name, self._init.space_needed, self._init.defs)
        self._data = WorkerSpace.data

    def execute(self, X):
        adjacency_matrix = deepcopy(self._adjacency_matrix)
        matrix_keys = self._matrix_keys
        done_event, error_event = (Event(), Event())
        pool = self._pool
        work_items = self._work_items
        completed = 0
        total = len(self._work_items)

        print(f"adjacency_matrix = {adjacency_matrix}")

        self._data["input"][:] = X
        print(f"self._dat={self._data}")

        def error_callback(e):
            print(f"exception in worker encountered: {e}")
            error_event.set()
            done_event.set()

        def callback(key):
            nonlocal adjacency_matrix, matrix_keys, work_items, completed, total, done_event, pool
            print(f"callback for {key}...")
            ij = matrix_keys[key]
            for k, row in enumerate(adjacency_matrix):
                is_set = row[ij]
                row[ij] = False
                if is_set:
                    if check_row_is_clear(adjacency_matrix, k):
                        print(f"callback kick off for {k}")
                        pool.apply_async(
                            work_items[k].fn,
                            work_items[k].args,
                            callback=callback,
                            error_callback=error_callback,
                        )
            completed += 1
            if completed == total:
                print(f"setting done event...")
                done_event.set()

        print(f"Kicking off work..")
        for i in range(len(adjacency_matrix)):
            if check_row_is_clear(adjacency_matrix, i):
                print(f"initial kick off for {i}")
                pool.apply_async(
                    work_items[i].fn,
                    work_items[i].args,
                    callback=callback,
                    error_callback=error_callback,
                )

        print(f"waiting on done event...")
        done_event.wait(timeout=30.0)
        if error_event.is_set():
            return False, None

        return True, self._out_fn(self._data) if self._out_fn is not None else None

    def __del__(self):
        try:
            self._pool.close()
            self._pool.join()
            self._shmem.unlink()
        except Exception as e:
            print(f"Encountered exception {e} on __del__ of ProcessOrchestrator")


def make_adjacency_matrix(work_defs):
    N = len(work_defs)
    adjacency_matrix = [[False] * N for _ in range(N)]
    matrix_keys = {}
    current_ij = 0

    def _index_of(key_to_ij, key):
        nonlocal current_ij
        if key not in matrix_keys:
            matrix_keys[key] = current_ij
            current_ij += 1
        return matrix_keys[key]

    # Build adjacency matrix
    for work_item in work_defs:
        provides_ij = _index_of(matrix_keys, work_item.provides)
        if work_item.depends_on is None:
            continue
        for dep in work_item.depends_on.split(","):
            depends_on_ij = _index_of(matrix_keys, dep)
            adjacency_matrix[provides_ij][depends_on_ij] = True

    return adjacency_matrix, matrix_keys


def _validate_work(work_defs: list):
    all_provides_unique = set()
    all_dependencies_satisfied = {}

    for work_item in work_defs:
        provides = work_item.provides
        if provides in all_provides_unique:
            raise ValueError(f"duplicate item provided in work items: {provides}")
        all_provides_unique.add(work_item.provides)

    for work_item in work_defs:
        depends_on = work_item.depends_on
        if depends_on is None:
            continue
        for dep in depends_on.split(","):
            if dep not in all_provides_unique:
                raise ValueError(
                    f"dependency not resolved: {work_item.provides} depends on {dep}"
                )

    adjacency_matrix, key_to_ij = make_adjacency_matrix(work_defs)

    # Use the adjacency matrix to search a key and all of its downstream edges for a cycle
    def dfs(matrix, upstream_ij, visited, current_i, length):
        downstream = set()
        if current_i in upstream_ij:
            for j in range(length):
                if matrix[current_i][j] and j in upstream_ij:
                    return True, None
                downstream.add(j)
            return False, downstream

        upstream_ij.add(current_i)
        for j in range(length):
            if matrix[current_i][j]:
                found, new_downstream = dfs(matrix, upstream_ij, visited, j, length)
                if found:
                    return True, None
                for k in new_downstream:
                    downstream.add(k)
                downstream.add(j)
        for j in range(length):
            if j in downstream:
                matrix[current_i][j] = True
        upstream_ij.remove(current_i)
        visited.add(current_i)
        return False, downstream

    found, _ = dfs(adjacency_matrix, set(), set(), 0, len(adjacency_matrix))
    if found:
        raise ValueError(f"circular dependency detected in workdefs")
