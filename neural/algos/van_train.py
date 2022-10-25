from queue import Queue

from ..tools.timer import timer, init_timer_manager, PrintManager

import torch


def van_train(shared, policy, optim, next_batch_q, done_queue, device):
    """
    To train a policy using vanilla stochastic gradient descent.
    """
    policy.to(device)

    init_timer_manager(PrintManager(10000))
    while True:
        with timer("vanilla-train-loop"):
            item = next_batch_q.get()
        if item == "STOP":
            break

        if isinstance(item, tuple):
            cmd, args = item

            if cmd == "PROCESS":
                batch = args
                process_batch(policy, optim, batch, device)

                with torch.no_grad():
                    for share, updated in zip(shared.parameters(), policy.parameters()):
                        share[:] = updated[:]

                # done_queue.put(("DONE", batch_id))
                for item in batch:
                    del item
                del batch
            else:
                print(f"Received unknown command in van_train: cmd={cmd}, args={args}")
        else:
            print(f"Received unknown command of type {type(item)} and value={item}")


def process_batch(policy, optim, batch, device):
    policy.zero_grad()

    state, action, reward = batch

    log_prob = policy.log_prob_of(state, action)

    loss = -log_prob * reward

    loss.backward()

    optim.step()
