import gym
import torch
import numpy as np
import copy
import time

from collections import OrderedDict
from multiprocessing import freeze_support

from util.exp_replay import ExpReplay
from model.model import Model
from model.modules import Activation, Layer
from tools.sink import Sink

actor = None
critic_1 = None
critic_2 = None

target_actor = None
target_critic_1 = None
target_critic_2 = None

actor_observers = []
critic_observers = []

graph_sink = None

if __name__ == "__main__":
    freeze_support()
    # graph_sink = GraphSink()
    graph_sink = Sink()

    def reset_models():
        global actor, critic_1, critic_2, target_actor, target_critic_2, target_critic_1, actor_observers, critic_observers, graph_sink

        actor = Model(
            "actor",
            [
                Layer("l_1", torch.nn.Linear(11, 300)),
                Activation("tanh_1", torch.nn.Tanh()),
                Layer("l_2", torch.nn.Linear(300, 300)),
                Activation("tanh_2", torch.nn.Tanh()),
                Layer("l_out", torch.nn.Linear(300, 1)),
                Activation("tanh_3", torch.nn.Tanh()),
            ],
            graph_sink,
        )

        critic_1 = Model(
            "critic_1",
            [
                Layer("l_1", torch.nn.Linear(12, 300)),
                Activation("tanh_1", torch.nn.Tanh()),
                Layer("l_2", torch.nn.Linear(300, 300)),
                Activation("tanh_2", torch.nn.Tanh()),
                Layer("l_out", torch.nn.Linear(300, 1)),
            ],
            graph_sink,
        )

        critic_2 = Model(
            "critic_1",
            [
                Layer("l_1", torch.nn.Linear(12, 300)),
                Activation("tanh_1", torch.nn.Tanh()),
                Layer("l_2", torch.nn.Linear(300, 300)),
                Activation("tanh_2", torch.nn.Tanh()),
                Layer("l_out", torch.nn.Linear(300, 1)),
            ],
            graph_sink,
        )

        target_actor = Model("target_actor", [], graph_sink)
        target_critic_1 = Model("target_critic_1", [], graph_sink)
        target_critic_2 = Model("target_critic_2", [], graph_sink)

        # TODO: Fix the copying of models.
        target_actor._layers = copy.deepcopy(actor.layers())
        target_critic_1._layers = copy.deepcopy(critic_1.layers())
        target_critic_2._layers = copy.deepcopy(critic_2.layers())

        # Remove the need for set_sink_hooks
        target_actor.set_sink_hooks()
        target_critic_1.set_sink_hooks()
        target_critic_2.set_sink_hooks()

        time.sleep(1.0)

    reset_models()
    graph_sink.start()

    env = gym.make("InvertedDoublePendulum-v4")
    # env = gym.make("InvertedPendulum-v4")

    GAMMA = 0.995

    CRITIC_LEARNING_RATE = 0.001
    ACTOR_LEARNING_RATE = 0.001
    TAU = 0.001

    EXPLORE_SIGMA = 0.3
    ACTION_SMOOTHING_SIGMA = 0.1
    ACTION_SMOOTHING_CLIP = 0.2

    POLICY_ITERATIONS = 100000
    EPISODES = 10000000  # Max number of trajectories in a batch
    STEPS = 2000  # Max number of steps in a trajectory

    ALL_UPDATE = 4

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

    def update_critic(batch):
        global critic_1, critic_2, CRITIC_LEARNING_RATE, target_actor, target_critic_1, target_critic_2, GAMMA, HIGH, LOW, MID, HALF_LENGTH, actor_observers, critic_observers

        states, actions, rewards, next_states, dones = batch

        target_actor.metrics_on()
        critic_1.metrics_on()
        critic_2.metrics_on()
        target_critic_1.metrics_on()
        target_critic_2.metrics_on()

        # [a.on() for a in actor_observers]
        # [c.on() for c in critic_observers]
        targets = None
        with torch.no_grad():
            smoothing_noise = torch.clamp(
                torch.normal(0.0, ACTION_SMOOTHING_SIGMA, size=(1, len(next_states))),
                -ACTION_SMOOTHING_CLIP,
                ACTION_SMOOTHING_CLIP,
            )
            target_actions = torch.add(
                torch.clamp(
                    target_actor(torch.tensor(next_states).float()), min=LOW, max=HIGH
                )
                * HALF_LENGTH
                + MID,
                smoothing_noise.T,
            )
            # print(f"next_states={next_states}, target_actions={target_actions}")
            for_critic = torch.tensor(
                np.concatenate((next_states, np.vstack(target_actions.numpy())), axis=1)
            ).float()
            # print(f"shape={for_critic.shape}, for_critic={for_critic}")
            target_qs = torch.minimum(
                target_critic_1(for_critic), target_critic_2(for_critic)
            )
            targets = torch.tensor(rewards) + GAMMA * torch.tensor(dones) * target_qs

        # [a.off() for a in actor_observers]
        # [c.off() for c in critic_observers]
        target_actor.metrics_off()
        critic_1.metrics_off()
        critic_2.metrics_off()
        target_critic_1.metrics_off()
        target_critic_2.metrics_off()

        critic_1.zero_grad()
        critic_2.zero_grad()

        state_actions = torch.tensor(
            np.concatenate((states, np.vstack(actions)), axis=1), requires_grad=False
        )

        mse_1 = (targets - critic_1(state_actions.float())) ** 2

        loss_1 = -mse_1.mean()
        loss_1.backward()

        with torch.no_grad():
            for param in critic_1.parameters():
                param += CRITIC_LEARNING_RATE * param.grad

        mse_2 = (targets - critic_2(state_actions.float())) ** 2
        loss_2 = -mse_2.mean()
        loss_2.backward()

        with torch.no_grad():
            for param in critic_2.parameters():
                param += CRITIC_LEARNING_RATE * param.grad

    def update_actor(batch):
        global actor, critic_1, device, ACTOR_LEARNING_RATE, HIGH, LOW, MID, HALF_LENGTH, actor_observers, critic_observers

        states, actions, rewards, next_states, dones = batch

        actor.zero_grad()
        critic_1.zero_grad()

        actor.metrics_on()
        # [a.on() for a in actor_observers]
        new_actions = torch.clamp(
            actor(torch.tensor(states).float()) * HALF_LENGTH + MID, min=LOW, max=HIGH
        )
        # [a.off() for a in actor_observers]
        actor.metrics_off()

        state_actions = torch.cat((torch.tensor(states), new_actions), axis=1)

        critic_1.metrics_on()
        # [c.on() for c in critic_observers]

        qs = critic_1(state_actions.float())

        # [c.off() for c in critic_observers]
        critic_1.metrics_off()

        direction = qs.mean()
        direction.backward()

        with torch.no_grad():
            for param in actor.parameters():
                param += ACTOR_LEARNING_RATE * param.grad

    def update_targets():
        global actor, critic_1, critic_2, target_actor, target_critic_2, target_critic_1, TAU

        with torch.no_grad():
            for target, update in zip(target_actor.parameters(), actor.parameters()):
                target.copy_(TAU * update + (1.0 - TAU) * target)

            for target, update in zip(
                target_critic_1.parameters(), critic_1.parameters()
            ):
                target.copy_(TAU * update + (1.0 - TAU) * target)

            for target, update in zip(
                target_critic_2.parameters(), critic_2.parameters()
            ):
                target.copy_(TAU * update + (1.0 - TAU) * target)

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

        def exploration_factor():
            normal = torch.distributions.normal.Normal(0.0, EXPLORE_SIGMA)
            sample = normal.sample()
            out = sample.item()
            return out

        observation = env.reset()

        for step in range(STEPS):
            if (total_steps + 1) % 300 == 0:
                [a.flush() for a in actor_observers]
                [c.flush() for c in critic_observers]

            actor.eval()
            total_steps += 1

            action = None
            with torch.no_grad():
                action = (
                    actor.forward(
                        torch.tensor(np.array([observation]), dtype=float).float()
                    )
                    * HALF_LENGTH
                    + MID
                )

            action = torch.clamp(action + exploration_factor(), min=LOW, max=HIGH)
            next_observation, reward, done, info = env.step(action[0])

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

            if len(replay) > MINI_BATCH * 5:
                actor.train()
                critic_1.train()
                critic_2.train()
                target_actor.train()
                target_critic_1.train()
                target_critic_2.train()

                batch = replay.sample(MINI_BATCH)
                update_critic(batch)

                if total_steps % ALL_UPDATE == 0:
                    update_actor(batch)
                    update_targets()

        # print(f"episode finished in {episode_steps}, curr_max={curr_max}")

        if (episode + 1) % SHOW == 0:
            target_actor.eval()

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
                        action = target_actor(
                            torch.tensor([observation], dtype=float).float()
                        )
                    action = torch.clamp(action, min=LOW, max=HIGH)

                    next_observation, reward, done, info = env.step(action[0])

                    env.render()
                    if (i % (COUNT / 3)) == 0:
                        print(
                            f"obs={next_observation}, reward={reward}, action={action}."
                        )

                    if done or step == STEPS - 1:
                        if (i % (COUNT / 5)) == 0:
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
