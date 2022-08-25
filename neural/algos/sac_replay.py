from ..util.exp_replay import ExpReplayReader, ExpReplayWriter, SharedBuffers
from ..tools.timer import timer, init_timer_manager, PrintManager
import torch
import numpy as np
from queue import Queue, Empty
from threading import Event


def sac_replay_store(exp_q: Queue, buffers: SharedBuffers):
    init_timer_manager(PrintManager(100000))
    exp_replay = ExpReplayWriter(buffers)

    for item in iter(exp_q.get, "STOP"):
        cmd, args = item
        if cmd == "EXP":
            with timer("push-to-replay"):
                exp_replay.push(args)
        else:
            print(f"Bad command passed to sac_replay_store: {cmd}, {args}")


def sac_replay_sample(
    batch_q: Queue,
    returns_q: Queue,
    start: Event,
    stop: Event,
    hypers: dict,
    buffers: SharedBuffers,
    device: str,
):
    init_timer_manager(PrintManager(20000))
    exp_replay = ExpReplayReader(buffers)
    live_batches = {}
    free_tensors = []

    start.wait()

    batch_id = 0
    while not stop.is_set():
        with timer("sample-loop"):
            with timer("sample"):
                try:
                    sample = exp_replay.sample(hypers["minibatch_size"])
                except RuntimeError as e:
                    print(f"Encountered exception {e} when sampling. Retrying.")
                    continue
            batch = make_as_tensor(sample, free_tensors, device)
            with timer("pushing-sample-to-q"):
                batch_q.put(("PROCESS", (batch_id, batch)))
            live_batches[batch_id] = batch
            batch_id += 1

            with timer("reclaiming-return"):
                try:
                    cmd, return_id = returns_q.get_nowait()
                    if cmd == "DONE":
                        free_tensors.append(live_batches[return_id])
                        del live_batches[return_id]
                    else:
                        print(
                            f"Unexpected command in returns queue: {cmd}, {return_id}"
                        )
                except Empty as e:
                    pass


def to_new_tensor(arr: np.ndarray, device: str):
    return torch.tensor(arr, dtype=torch.float32, device=device, requires_grad=False)


def to_tensor(arr: np.ndarray, old_tensor: torch.Tensor):
    old_tensor[:] = torch.from_numpy(arr)[:]
    return old_tensor


def make_state_actions(initial_tensors: list):
    return torch.cat((initial_tensors[0], initial_tensors[1]), dim=1)


def make_as_tensor(sample: list, free_tensors: list, device: str):
    if free_tensors:
        out = []
        to_replace = free_tensors.pop()
        for tensor, arr in zip(to_replace, sample):
            out.append(to_tensor(arr, tensor))
        out.append(to_replace[-1].copy_(make_state_actions(out)))
        return out

    else:
        out = [to_new_tensor(s, device) for s in sample]
        out.append(make_state_actions(out))
        return out
