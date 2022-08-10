from neural.algos.sac import SoftActorCritic
from neural.tools.timer import timer, init_timer_manager, PrintManager

import gym

if __name__ == "__main__":
    init_timer_manager(PrintManager(10 * 1000))

    hypers = {
        "future_reward_discount": 0.995,
        "q_lr": 0.0001,
        "v_lr": 0.0001,
        "actor_lr": 0.001,
        "target_update_step": 0.001,
        "experience_replay_size": 1000 * 1000,
        "minibatch_size": 128,
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
        "episodes_per_training": 5,
        "max_steps": 200,
        "steps_between_updates": 1,
        "episodes_per_test": 2,
        "training_iterations": 400,
        "device": "cpu",
    }

    env = gym.make("MountainCarContinuous-v0")
    sac = SoftActorCritic(hypers, layers, training_params, env)

    for i in range(training_params["training_iterations"]):
        with timer(f"training"):
            sac.train()
        with timer(f"test"):
            test_results = sac.test(render=False)
        print(f"iteration-{i} results={test_results}")

    print("DONE!")
