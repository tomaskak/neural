from ..util.exp_replay import ExpReplay

import time
import torch
from queue import Queue


def sac_replay(
    new_data_q: Queue,
    push_batch_q: Queue,
    hypers: dict,
    in_size: int,
    out_size: int,
    push_q_length: int,
    device: str,
):
    replay_buf = _init_replay_buffer(
        hypers["experience_replay_size"], in_size, out_size
    )
    next_id = 0
    live_samples = {}

    for item in iter(new_data_q.get, "STOP"):
        cmd, args = item
        if cmd == "EXP":
            new_data = args
            replay_buf.push(new_data)

            if (
                len(replay_buf) >= hypers["minibatch_size"] * 2
                and push_batch_q.qsize() < push_q_length
            ):
                start = time.time()
                sample = replay_buf.sample(hypers["minibatch_size"])

                new_sample = []
                for i, part in enumerate(sample):
                    if i == 2 or i == 4:
                        new_sample.append(
                            torch.tensor(
                                part, device=device, dtype=torch.float32
                            ).reshape(-1, 1)
                        )
                    else:
                        new_sample.append(
                            torch.tensor(part, device=device, dtype=torch.float32)
                        )
                new_sample.append(torch.cat((new_sample[0], new_sample[1]), dim=1))
                push_batch_q.put(("PROCESS", (next_id, new_sample)))
                live_samples[next_id] = new_sample
                next_id += 1

        elif cmd == "DONE":
            batch_id = args
            del live_samples[args]


def _init_replay_buffer(replay_size: int, in_size: int, out_size: int):
    state_spec = ("f8", (in_size,))
    action_spec = ("f8", (out_size,))
    return ExpReplay(replay_size, [state_spec, action_spec, float, state_spec, float])
