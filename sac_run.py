from neural.algos.sac import SoftActorCritic
from neural.tools.timer import timer, init_timer_manager, PrintManager

import torch
import argparse
import time
import gym

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", dest="file")
    parser.add_argument("--iterations", "-i", dest="iterations")
    parser.add_argument("--test-only", dest="test_only", default=False)
    parser.add_argument("--render", "-r", dest="render", default=False)
    args = parser.parse_args()

    init_timer_manager(PrintManager(10 * 1000))

    hypers = {
        "future_reward_discount": 0.995,
        "q_lr": 0.0001,
        "v_lr": 0.0001,
        "actor_lr": 0.001,
        "target_update_step": 0.001,
        "experience_replay_size": 1000 * 1000,
        "minibatch_size": 64,
    }
    layers = {
        "actor": [
            ("l1", "tanh", "input", "input*2"),
            ("l2", "tanh", "input*2", "output*2"),
        ],
        "q_1": [
            ("l1", "tanh", "input + output", "input*3"),
            ("l2", "tanh", "input*3", "input*3"),
            ("l3", "linear", "input*3", 1),
        ],
        "q_2": [
            ("l1", "tanh", "input + output", "input*3"),
            ("l2", "tanh", "input*3", "input*3"),
            ("l3", "linear", "input*3", 1),
        ],
        "value": [
            ("l1", "tanh", "input", "input*3"),
            ("l2", "tanh", "input*3", "input*3"),
            ("l3", "linear", "input*3", 1),
        ],
    }
    training_params = {
        "episodes_per_training": 20,
        "max_steps": 2000,
        "steps_between_updates": 1,
        "episodes_per_test": 5,
        "training_iterations": 100 if args.iterations is None else args.iterations,
        "device": "cpu",
    }

    env = gym.make("InvertedPendulum-v4")
    sac = SoftActorCritic(hypers, layers, training_params, env)

    if args.file is not None:
        sac.load(torch.load(args.file))

    for i in range(training_params["training_iterations"]):
        if not args.test_only:
            with timer(f"training"):
                sac.train()
        with timer(f"test"):
            test_results = sac.test(render=args.render)
        print(f"iteration-{i} results={test_results}")

    torch.save(sac.save(), f"./sac-{int(time.time())}.ai")

    print("DONE!")
