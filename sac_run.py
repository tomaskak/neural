from neural.algos.sac import SoftActorCritic
from neural.tools.timer import timer, init_timer_manager, PrintManager

import torch
import argparse
import time
import json
import gym

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", dest="file")
    parser.add_argument("--iterations", "-i", dest="iterations", type=int)
    parser.add_argument(
        "--test-only", dest="test_only", action="store_true", default=False
    )
    parser.add_argument(
        "--render", "-r", dest="render", action="store_true", default=False
    )
    parser.add_argument(
        "--multiprocess", "-m", dest="multiprocess", action="store_true", default=False
    )
    args = parser.parse_args()

    init_timer_manager(PrintManager(5 * 1000))

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
            ("l1", "tanh", "input + output", "input*3+output*3"),
            ("l2", "tanh", "input*3+output*3", "input*3+output*3"),
            ("l3", "linear", "input*3+output*3", 1),
        ],
        "q_2": [
            ("l1", "tanh", "input + output", "input*3+output*3"),
            ("l2", "tanh", "input*3+output*3", "input*3+output*3"),
            ("l3", "linear", "input*3+output*3", 1),
        ],
        "value": [
            ("l1", "tanh", "input", "input*3+output*3"),
            ("l2", "tanh", "input*3+output*3", "input*3+output*3"),
            ("l3", "linear", "input*3+output*3", 1),
        ],
    }
    training_params = {
        "episodes_per_training": 20,
        "max_steps": 10000,
        "steps_between_updates": 1,
        "episodes_per_test": 5,
        "training_iterations": 100 if args.iterations is None else args.iterations,
        "device": "cpu",
        "save_on_iteration": 20,
        "multiprocess": args.multiprocess,
    }

    # env_key = "Ant-v4"
    env_key = "InvertedPendulum-v4"
    env = gym.make(env_key)
    sac = SoftActorCritic(hypers, layers, training_params, env)

    log_file = f"./sac-{env_key}-{int(time.time())}.result"

    if args.file is not None:
        sac.load(torch.load(args.file))

    for i in range(training_params["training_iterations"]):
        if not args.test_only:
            with timer(f"training"):
                sac.train()

        with timer(f"test"):
            test_results = sac.test(render=args.render)
            with open(log_file, "a") as f:
                f.write(json.dumps({f"{i}": test_results}, indent=4))

        if (i + 1) % training_params["save_on_iteration"] == 0:
            torch.save(sac.save(), f"./sac-{env_key}-{int(time.time())}.ai")

        print(f"iteration-{i} results={test_results}")

    if not args.test_only:
        torch.save(sac.save(), f"./sac-{env_key}-{int(time.time())}.ai")

    print("DONE!")
