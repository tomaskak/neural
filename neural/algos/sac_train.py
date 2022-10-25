from ..algos.sac_context import SACContext
from queue import Queue
from ..tools.timer import timer, init_timer_manager, PrintManager

import torch


def sac_train(
    context: SACContext,
    next_batch_q: Queue,
    hypers: dict,
    steps_to_report: int,
    report_queue: Queue,
    done_queue: Queue,
    device: str = "cpu",
    discrete=False,
):
    """
    Defines the training loop for the SAC algorithm.

    Receives instructions from the next_batch_q which has recognized instructions:
        ("PROCESS", (<batch_id>, batch)) - train models on the batch
        ("STOP") - stop processing and exit loop

    On completion of a PROCESS instruction with batch_id and batch, loop will cleanup by
    calling del on batch and pushing ("DONE", batch_id) into done_queue.

    On completing steps_to_report number of training steps a dict with info of training progress
    will be pushed into the report_queue.
    """
    init_timer_manager(PrintManager(10000))
    items_processed = 0

    context.to(device)

    while True:
        with timer("training-loop"):
            with timer("time waiting for queue"):
                item = next_batch_q.get()
            if item == "STOP":
                break

            if type(item) == tuple:
                cmd, args = item
                if cmd == "PROCESS":
                    batch_id, batch = args
                    items_processed += 1
                    if items_processed % steps_to_report == 0:
                        try:
                            report_queue.put_nowait({"completed": items_processed})
                        except Exception as e:
                            print(f"exception putting results: {e}")
                        # This updates the state of all models in the 'shared' structure.
                        context.update_shared()

                    process_batch(context, hypers, batch, discrete)

                    done_queue.put(("DONE", batch_id))
                    for item in batch:
                        del item
                    del batch
                else:
                    print(
                        f"Received unknown command in sac_train: cmd={cmd}, args={args}"
                    )
            else:
                print(f"Received unknown command of type {type(item)} and value={item}")


def process_batch(
    context: SACContext, hypers: dict, batch: tuple, discrete: bool = False
):
    with timer("process-one-batch"):
        states, actions, rewards, next_states, dones, state_actions = batch

        context.q_1.zero_grad()
        context.q_2.zero_grad()
        context.actor.zero_grad()
        context.entropy_weight.grad = None

        if not discrete:
            predicted_q_1s = context.q_1.forward(state_actions)
            predicted_q_2s = context.q_2.forward(state_actions)
            new_actions, log_probs = context.actor.forward(states)

            # Entropy weight update
            entropy_loss = -(
                context.entropy_weight.exp()
                * (log_probs + hypers["target_entropy_weight"]).detach()
            ).mean()
            e_weight = context.entropy_weight.exp().detach()
            entropy_loss.backward()

            # Q updates
            new_state_actions = torch.cat((states, new_actions), 1)
            predicted_new_qs = torch.minimum(
                context.q_1.forward(new_state_actions),
                context.q_2.forward(new_state_actions),
            )
            next_actions, next_log_probs = context.actor.forward(next_states)

            next_state_actions = torch.cat((next_states, next_actions), dim=1)
            target_qs = torch.minimum(
                context.q_1_target.forward(next_state_actions),
                context.q_2_target.forward(next_state_actions),
            ) - e_weight * next_log_probs.reshape(-1, 1)

            q_target = rewards + dones * hypers["future_reward_discount"] * target_qs

            q_1_loss = context.q_1_loss_fn(predicted_q_1s, q_target.detach())
            q_2_loss = context.q_2_loss_fn(predicted_q_2s, q_target.detach())
            q_1_loss.backward()
            q_2_loss.backward()

            loss = context.actor_loss_fn(
                e_weight * log_probs.reshape(-1, 1), predicted_new_qs
            )
            loss.backward()
        else:
            predicted_q_1s = context.q_1.forward(states)
            predicted_q_2s = context.q_2.forward(states)
            new_actions = context.actor.forward(states)
            log_probs = torch.log(new_actions + (new_actions == 0.0) * 1e-8)

            # print(f"predicted_q_1s={predicted_q_1s}, new_actions={new_actions}, log_probs={log_probs}")

            # Entropy weight update
            entropy_loss = (
                (
                    new_actions.detach()
                    * -1
                    * (
                        context.entropy_weight.exp()
                        * (
                            log_probs.detach() + hypers["target_entropy_weight"]
                        ).detach()
                    )
                )
                .sum(dim=1)
                .mean()
            )
            e_weight = context.entropy_weight.exp().detach()
            entropy_loss.backward()

            next_actions = context.actor.forward(next_states)
            next_log_probs = torch.log(next_actions + (next_actions == 0.0) * 1e-8)

            # Q updates
            predicted_new_qs = torch.minimum(
                predicted_q_1s.detach(), predicted_q_2s.detach()
            )

            min_qs = torch.minimum(
                context.q_1_target.forward(next_states),
                context.q_2_target.forward(next_states),
            )
            # print(f"next_actions={next_actions}, min_qs={min_qs}, times={next_actions*min_qs}, next_log_probs={next_log_probs}")

            pre_sum = next_actions * (min_qs - e_weight * next_log_probs)
            target_qs = pre_sum.sum(dim=1).unsqueeze(-1)

            q_target = rewards + dones * hypers["future_reward_discount"] * target_qs

            # print(f"predicted_qs={predicted_q_1s.sum(dim=1).reshape(-1, 1)}, target={q_target}")

            # print(f"actions = {actions}")
            q_1_loss = context.q_1_loss_fn(
                predicted_q_1s.gather(1, actions.long()), q_target.detach()
            ).mean()
            q_2_loss = context.q_2_loss_fn(
                predicted_q_2s.gather(1, actions.long()), q_target.detach()
            ).mean()
            # print(f"q_1_loss={q_1_loss}, q_2_loss={q_2_loss}")
            q_1_loss.backward()
            q_2_loss.backward()

            # print(f"new_actions={new_actions}, log_probs={log_probs}, new_qs={predicted_new_qs}, inner={(e_weight * log_probs - predicted_new_qs)}")
            loss = (
                (new_actions * (e_weight * log_probs - predicted_new_qs))
                .sum(dim=1)
                .mean()
            )
            # print(f"actor_loss={loss}")
            loss.backward()

        context.actor_optim.step()
        context.q_1_optim.step()
        context.q_2_optim.step()
        context.entropy_weight_optim.step()
        with torch.no_grad():
            for target, update in zip(
                context.q_1_target.parameters(), context.q_1.parameters()
            ):
                target.copy_(
                    hypers["target_update_step"] * update
                    + (1.0 - hypers["target_update_step"]) * target
                )
            for target, update in zip(
                context.q_2_target.parameters(), context.q_2.parameters()
            ):
                target.copy_(
                    hypers["target_update_step"] * update
                    + (1.0 - hypers["target_update_step"]) * target
                )

        # Actor is the only model that needs to be updated on each step as the new data entered into
        # the replay buffer benefits from the latest model.
        context.shared.actor = context.actor
