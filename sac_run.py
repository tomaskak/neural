from neural.algos.sac import SoftActorCritic

import gym

if __name__ == "__main__":
    hypers = {
        "future_reward_discount": 0.995,
        "q_lr": 0.0001,
        "v_lr": 0.0001,
        "actor_lr": 0.001,
        "target_update_step": 0.001,
        "experience_replay_size": 1000 * 1000,
        "minibatch_size": 32,
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
    training_params = {}

    env = gym.make("InvertedPendulum-v4")
    sac = SoftActorCritic(hypers, layers, training_params, env)

    for i in range(400):
        print(f"train-{i}")
        sac.train()
        print(f"test-{i}")
        test_results = sac.test(render=i >= 20)
        print(f"results={test_results}")

    print("DONE!")
