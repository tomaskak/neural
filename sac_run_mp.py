from neural.algos.sac_mp import SoftActorCritic
from neural.tools.timer import timer, init_timer_manager, PrintManager

from torch.multiprocessing import set_start_method

import torch
import argparse
import time
import json
import gym

if __name__ == "__main__":
    set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", dest="file")
    parser.add_argument(
        "--path", "-p", dest="path", type=str, default="/vol/run_artifacts"
    )
    parser.add_argument("--iterations", "-i", dest="iterations", type=int)
    parser.add_argument(
        "--test-only", dest="test_only", action="store_true", default=False
    )
    parser.add_argument(
        "--render", "-r", dest="render", action="store_true", default=False
    )
    args = parser.parse_args()

    init_timer_manager(PrintManager(5000))

    hypers = {
        "future_reward_discount": 0.995,
        "q_lr": 0.0003,
        "v_lr": 0.0003,
        "actor_lr": 0.0003,
        "max_action": 1.0,
        "target_update_step": 0.001,
        "experience_replay_size": 1000 * 1000,
        "minibatch_size": 128,
    }
    layers = {
        "actor": [
            ("l1", "tanh", "input", "input*4"),
            ("l2", "tanh", "input*4", "output*2"),
        ],
        "q_1": [
            ("l1", "tanh", "input + output", "input*6+output*6"),
            ("l2", "tanh", "input*6+output*6", "input*6+output*6"),
            ("l3", "linear", "input*6+output*6", 1),
        ],
        "q_2": [
            ("l1", "tanh", "input + output", "input*6+output*6"),
            ("l2", "tanh", "input*6+output*6", "input*6+output*6"),
            ("l3", "linear", "input*6+output*6", 1),
        ],
        "value": [
            ("l1", "tanh", "input", "input*6+output*6"),
            ("l2", "tanh", "input*6+output*6", "input*6+output*6"),
            ("l3", "linear", "input*6+output*6", 1),
        ],
    }
    training_params = {
        "episodes_per_training": 30,
        "max_steps": 10000,
        "steps_between_updates": 1,
        "episodes_per_test": 5,
        "training_iterations": 100 if args.iterations is None else args.iterations,
        "device": "cpu",
        "save_on_iteration": 5,
    }

    # env_key = "Ant-v4"
    # env_key = "InvertedPendulum-v4"
    env_key = "InvertedDoublePendulum-v4"
    env = gym.make(env_key)
    sac = SoftActorCritic(hypers, layers, training_params, env)

    log_file = args.path + f"/sac-{env_key}-{int(time.time())}.result"

    def save(data):
        torch.save(sac.save(), args.path + f"/sac-{env_key}-{int(time.time())}.ai")

    count = 0

    def result(res):
        global count
        with open(log_file, "a") as f:
            f.write(json.dumps({f"{count}": res}, indent=4))
        count += 1

    if args.file is not None:
        sac.load(torch.load(args.file))

    if args.test_only:
        for i in range(training_params["training_iterations"]):
            with timer(f"test"):
                test_results = sac.test(render=args.render)
                print(f"iteration-{i} results={test_results}")
    else:
        sac.start(render=args.render, save_hook=save, result_hook=result)

    if not args.test_only:
        torch.save(sac.save(), args.path + f"/sac-{env_key}-{int(time.time())}.ai")

    print("DONE!")
