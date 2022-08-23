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
    init_timer_manager(PrintManager(5000))     
    items_processed = 0
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

                    process_batch(context, hypers, batch)

                    done_queue.put(("DONE", batch_id))
                    for item in batch:
                        del item
                    del batch
                else:
                    print(f"Received unknown command in sac_train: cmd={cmd}, args={args}")
            else:
                print(f"Received unknown command of type {type(item)} and value={item}")


def process_batch(context: SACContext, hypers: dict, batch: tuple):
    with timer("process-one-batch"):
        states, actions, rewards, next_states, dones, state_actions = batch

        context.q_1.zero_grad()
        context.q_2.zero_grad()
        context.actor.zero_grad()
        context.value.zero_grad()

        predicted_values = context.value.forward(states)
        new_actions, log_probs = context.actor.forward(states)
        predicted_q_1s = context.q_1.forward(state_actions)
        predicted_q_2s = context.q_2.forward(state_actions)
        target_next_state_values = context.target_value.forward(next_states)

        rng = hypers["max_action"]
        # new_actions = torch.clamp(new_actions * rng, min=-rng, max=rng)
        new_actions = new_actions * rng

        new_state_actions = torch.cat((states, new_actions), 1)

        predicted_new_qs = torch.minimum(
            context.q_1.forward(new_state_actions),
            context.q_2.forward(new_state_actions),
        )

        # Q updates
        target_qs = (
            rewards
            + dones * hypers["future_reward_discount"] * target_next_state_values
        )
        target_qs = target_qs.reshape(-1, 1)

        q_1_loss = context.q_1_loss_fn(predicted_q_1s, target_qs.float())
        q_2_loss = context.q_2_loss_fn(predicted_q_2s, target_qs.float())

        q_1_loss.backward()
        q_2_loss.backward()

        # Value update
        target_vs = predicted_new_qs.detach() - log_probs.detach().reshape(-1, 1)
        v_loss = context.value_loss_fn(predicted_values, target_vs.float())

        v_loss.backward()

        # Policy update
        loss = context.actor_loss_fn(log_probs.reshape(-1, 1), predicted_new_qs)
        loss.backward()

        context.actor_optim.step()
        context.q_1_optim.step()
        context.q_2_optim.step()
        context.value_optim.step()
        with torch.no_grad():
            for target, update in zip(
                context.target_value.parameters(), context.value.parameters()
            ):
                target.copy_(
                    hypers["target_update_step"] * update
                    + (1.0 - hypers["target_update_step"]) * target
                )

        # Actor is the only model that needs to be updated on each step as the new data entered into
        # the replay buffer benefits from the latest model.
        context.shared.actor = context.actor
