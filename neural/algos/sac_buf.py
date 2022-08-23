from ..util.exp_replay import ExpReplay
from ..tools.timer import timer, init_timer_manager, PrintManager
import torch
import numpy as np
from queue import Queue


"""
Accepts new data and stores it in a local experience replay buffer.
"""

# def sac_buf(
#     new_data_q: Queue,
#     push_batch_q: Queue,
#     hypers: dict,
#     in_size: int,
#     out_size: int,
#     push_q_length: int,
#     device: str,
# ):
#     init_timer_manager(PrintManager(5000))
#     replay_buf = _init_replay_buffer(
#         hypers["experience_replay_size"], in_size, out_size
#     )
#     next_id = 0
#     live_samples = {}
#     empty_batch_bufs = []

#     for item in iter(new_data_q.get, "STOP"):
#         cmd, args = item
#         if cmd == "EXP":
#             new_data = args
#             replay_buf.push(new_data)

#             if (
#                 (len(replay_buf) >= hypers["minibatch_size"] * 2
#                  and push_batch_q.qsize() < push_q_length)):
#                 sample = replay_buf.sample(hypers["minibatch_size"])
#                 new_sample = []
#                 if not empty_batch_bufs:
#                     for i, part in enumerate(sample):
#                         # rewards at index 2, and dones at index 4, need to be reshaped . TODO: update buffer to output right shape
#                         if i == 2 or i == 4:
#                             new_sample.append(
#                                 torch.tensor(
#                                     part.astype(np.float32),
#                                     device=device,
#                                     dtype=torch.float32,
#                                     requires_grad=False,
#                                 ).reshape(-1, 1)
#                             )
#                         else:
#                             new_sample.append(
#                                 torch.tensor(
#                                     part.astype(np.float32),
#                                     device=device,
#                                     dtype=torch.float32,
#                                     requires_grad=False,
#                                 )
#                             )
#                     new_sample.append(torch.cat((new_sample[0], new_sample[1]), dim=1))
#                 else:
#                     new_sample = empty_batch_bufs.pop()
#                     for i, part in enumerate(sample):
#                         if i == 2 or i == 4:
#                             new_sample[i].copy_(
#                                 torch.from_numpy(part.astype(np.float32)).reshape(-1, 1)
#                             )
#                         else:
#                             new_sample[i].copy_(
#                                 torch.from_numpy(part.astype(np.float32))
#                             )
#                     new_sample[-1].copy_(
#                         torch.cat((new_sample[0], new_sample[1]), dim=1)
#                     )

#                 push_batch_q.put(("PROCESS", (next_id, new_sample)))
#                 if next_id not in live_samples:
#                     live_samples[next_id] = new_sample
#                     next_id += 1

#         elif cmd == "DONE":
#             # Reuse the tensors that are done so the shmem doesn't need to be recreated.
#             batch_id = args
#             empty_batch_bufs.append(live_samples[batch_id])
#             del live_samples[batch_id]


# def _init_replay_buffer(replay_size: int, in_size: int, out_size: int):
#     state_spec = ("f8", (in_size,))
#     action_spec = ("f8", (out_size,))
#     return ExpReplay(replay_size, [state_spec, action_spec, float, state_spec, float])
