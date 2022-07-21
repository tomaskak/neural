import gym
import torch
import numpy as np
import copy
import time

from collections import OrderedDict
from multiprocessing import freeze_support

from util.exp_replay import ExpReplay
from model.model import Model
from model.modules import Activation, Layer, Loss
from tools.sink import Sink

actor = None

Q_1 = None
Q_2 = None

value = None
target_value = None

graph_sink = None

if __name__ == "__main__":
    freeze_support()
    graph_sink = Sink()

    def reset_models():
        global actor, Q_1, Q_2, value, target_value, graph_sink

        actor = Model(
            "actor",
            [
                Layer("l_1", torch.nn.Linear(11, 300)),
                Activation("tanh_1", torch.nn.Tanh()),
                Layer("l_2", torch.nn.Linear(300, 300)),
                Activation("tanh_2", torch.nn.Tanh()),
                Layer("l_out", torch.nn.Linear(300, 2)),
                Activation("tanh_3", torch.nn.Tanh()),
            ],
            graph_sink,
        )

        Q_1 = Model(
            "Q_1",
            [
                Layer("l_1", torch.nn.Linear(12, 300)),
                Activation("tanh_1", torch.nn.Tanh()),
                Layer("l_2", torch.nn.Linear(300, 300)),
                Activation("tanh_2", torch.nn.Tanh()),
                Layer("l_out", torch.nn.Linear(300, 1)),
            ],
            graph_sink,
        )

        Q_2 = Model(
            "Q_2",
            [
                Layer("l_1", torch.nn.Linear(12, 300)),
                Activation("tanh_1", torch.nn.Tanh()),
                Layer("l_2", torch.nn.Linear(300, 300)),
                Activation("tanh_2", torch.nn.Tanh()),
                Layer("l_out", torch.nn.Linear(300, 1)),
            ],
            graph_sink,
        )

        value = Model(
            "value",
            [
                Layer("l_1", torch.nn.Linear(11, 300)),
                Activation("tanh_1", torch.nn.Tanh()),
                Layer("l_2", torch.nn.Linear(300, 300)),
                Activation("tanh_2", torch.nn.Tanh()),
                Layer("l_out", torch.nn.Linear(300, 1)),
            ],
            graph_sink,
        )

        target_value = Model("target_value", [], graph_sink)

        # TODO: Fix the copying of models.
        target_value._layers = copy.deepcopy(value.layers())

        # Remove the need for set_sink_hooks
        target_value.set_sink_hooks()

        time.sleep(1.0)

    reset_models()

    q_1_loss_fn = Loss("Q_1", torch.nn.MSELoss(), graph_sink)
    q_2_loss_fn = Loss("Q_2", torch.nn.MSELoss(), graph_sink)
    value_loss_fn = Loss("Value", torch.nn.MSELoss(), graph_sink)
    actor_loss_fn = Loss("Actor", lambda x, y: (x - y).mean(), graph_sink)

    graph_sink.start()

    env = gym.make("InvertedDoublePendulum-v4")

    GAMMA = 0.995

    Q_LEARNING_RATE = 0.001
    ACTOR_LEARNING_RATE = 0.001
    V_LEARNING_RATE = 0.001
    TAU = 0.001

    POLICY_ITERATIONS = 100000
    EPISODES = 10000000  # Max number of trajectories in a batch
    STEPS = 2000  # Max number of steps in a trajectory

    ALL_UPDATE = 1

    REPLAY_SIZE = 1000 * 1000
    MINI_BATCH = 64

    HIGH = 3.0
    LOW = -3.0
    HALF_LENGTH = 3.0
    MID = 0.0

    def new_replay_buffer(N, state_size):
        # Needs (state, action, reward, next_state, is_terminal)
        state_spec = ("f8", (state_size,))
        return ExpReplay(N, [state_spec, float, float, state_spec, float])

    def get_action(mu, log_sigma):
        global HIGH, LOW, MID, HALF_LENGTH

        std_normal = torch.distributions.normal.Normal(0, 1)
        z = std_normal.sample(sample_shape=mu.shape)
        sigma = torch.exp(log_sigma)
        action = torch.clamp((mu + sigma * z), min=LOW, max=HIGH)

        log_prob = torch.distributions.normal.Normal(mu, sigma).log_prob(action)

        return action, log_prob

    def update_all(batch):
        global value, target_value, actor, Q_1, Q_2, ACTOR_LEARNING_RATE, V_LEARNING_RATE, Q_LEARNING_RATE, GAMMA, HIGH, LOW, MID, HALF_LENGTH, TAU, q_1_loss_fn, q_2_loss_fn, value_loss_fn, actor_loss_fn

        states, actions, rewards, next_states, dones = batch
        Q_1.zero_grad()
        Q_2.zero_grad()
        actor.zero_grad()
        value.zero_grad()
        target_value.zero_grad()

        Q_1.metrics_on()
        Q_2.metrics_on()
        actor.metrics_on()
        value.metrics_on()
        target_value.metrics_on()

        state_actions = torch.tensor(
            np.concatenate((states, np.vstack(actions)), axis=1), requires_grad=False
        ).float()

        predicted_values = value.forward(torch.tensor(states).float())
        predicted_q_1s = Q_1.forward(torch.tensor(state_actions).float())
        predicted_q_2s = Q_2.forward(torch.tensor(state_actions).float())

        norm_params = actor.forward(torch.tensor(states).float())
        new_actions, log_probs = get_action(norm_params[:, 0], norm_params[:, 1])

        new_state_actions = torch.cat(
            (torch.tensor(states), new_actions.reshape(-1, 1)), 1
        ).float()
        Q_1.metrics_off()
        Q_2.metrics_off()

        predicted_new_qs = torch.minimum(
            Q_1.forward(new_state_actions), Q_2.forward(new_state_actions)
        )
        Q_2.metrics_on()

        loss_fn = torch.nn.MSELoss()

        # Q updates
        target_next_state_values = target_value.forward(
            torch.tensor(next_states).float()
        ).reshape(1, -1)
        target_qs = (
            torch.tensor(rewards)
            + torch.tensor(dones) * GAMMA * target_next_state_values
        )
        target_qs = target_qs.detach().reshape(-1, 1)

        q_1_loss = q_1_loss_fn(predicted_q_1s, target_qs.float())
        q_2_loss = q_1_loss_fn(predicted_q_2s, target_qs.float())

        q_1_loss.backward()
        q_2_loss.backward()

        # Value update
        target_vs = predicted_new_qs.detach() - log_probs.detach().reshape(-1, 1)
        v_loss = value_loss_fn(predicted_values, target_vs.float())

        v_loss.backward()

        # Policy update
        loss = actor_loss_fn(log_probs.reshape(-1, 1), predicted_new_qs)
        loss.backward()

        with torch.no_grad():
            for params in Q_1.parameters():
                params -= Q_LEARNING_RATE * params.grad

            for params in Q_2.parameters():
                params -= Q_LEARNING_RATE * params.grad

            for params in value.parameters():
                params -= V_LEARNING_RATE * params.grad

            for params in actor.parameters():
                # Subtract because we wish to minimize divergence.
                params -= ACTOR_LEARNING_RATE * params.grad

            for target, update in zip(target_value.parameters(), value.parameters()):
                target.copy_(TAU * update + (1.0 - TAU) * target)

        actor.metrics_off()
        value.metrics_off()
        target_value.metrics_off()

    SHOW = 100

    observation = env.reset()

    replay = new_replay_buffer(REPLAY_SIZE, len(observation))

    curr_max = 0.0
    total_steps = 0

    for episode in range(EPISODES):

        episode_steps = 0

        HIGH = env.action_space.high.item()
        LOW = env.action_space.low.item()
        length = HIGH - LOW
        MID = length / 2.0 + LOW
        HALF_LENGTH = length / 2.0

        observation = env.reset()

        for step in range(STEPS):
            actor.eval()
            total_steps += 1

            action = None
            with torch.no_grad():
                norm_params = actor.forward(
                    torch.tensor(np.array(observation), dtype=float).float()
                )
                normal = torch.distributions.normal.Normal(
                    norm_params[0], torch.exp(norm_params[1])
                )
                action = torch.clamp(normal.sample(), min=LOW, max=HIGH)

            next_observation, reward, done, info = env.step([action])

            replay.push(
                [
                    observation,
                    action.item(),
                    reward,
                    next_observation,
                    0.0 if done else 1.0,
                ]
            )

            observation = next_observation

            if done or step == STEPS - 1:
                episode_steps = step + 1
                if curr_max < step + 1:
                    curr_max = step + 1
                break

            if len(replay) > MINI_BATCH * 5 and total_steps % ALL_UPDATE == 0:
                actor.train()
                Q_1.train()
                Q_2.train()
                value.train()
                target_value.train()

                batch = replay.sample(MINI_BATCH)
                update_all(batch)

        # print(f"episode finished in {episode_steps}, curr_max={curr_max}")

        if (episode + 1) % SHOW == 0:
            COUNT = 50
            avg = 0.0
            local_max = 0
            print(
                f"showing {COUNT} runs after episode={episode}, curr_max={curr_max}..."
            )
            for i in range(COUNT):
                observation = env.reset()

                for step in range(STEPS):
                    action = None
                    with torch.no_grad():
                        norm_params = actor.forward(
                            torch.tensor(np.array(observation), dtype=float).float()
                        )
                        normal = torch.distributions.normal.Normal(
                            norm_params[0], torch.exp(norm_params[1])
                        )
                        action = norm_params[0]

                    next_observation, reward, done, info = env.step([action])

                    env.render()
                    if (i % (COUNT / 3)) == 0:
                        print(
                            f"obs={next_observation}, reward={reward}, action={action}."
                        )

                    if done or step == STEPS - 1:
                        if (i % (COUNT / 10)) == 0:
                            print(f"episode lasted {step+1} steps!\n")
                        episode_steps = step + 1
                        avg += (step + 1) * 1.0
                        if curr_max < step + 1:
                            curr_max = step + 1
                        if local_max < step + 1:
                            local_max = step + 1
                        break
                    observation = next_observation
            print(f"test phase completed with avg={avg/(COUNT*1.0)} steps!")
