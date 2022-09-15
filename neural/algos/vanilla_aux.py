from ..model.layer_factory import to_layers
from ..model.model import Model, NormalModel

from abc import ABC, abstractmethod

from numpy import ndarray
import torch

class AuxiliaryAlgo:
    @abstractmethod
    def get_action(self, id: int, observation: ndarray) -> ndarray:
        pass

    @abstractmethod
    def process_result(self, id: int, result: dict):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def train_mode(self):
        pass

    @abstractmethod
    def test_mode(self):
        pass
    
class VanillaGradient(AuxiliaryAlgo):
    def __init__(self, hypers: list, layer_defs:list, in_size: int, out_size: int, device:str):
        self._hypers = hypers
        self._device = device

        self.policy = NormalModel("aux", to_layers(in_size, out_size, layer_defs), None, True)
        self.policy.to(device)

        self.policy_optim = torch.optim.Adam(self.policy.parameters(), hypers["lr"])

        self.pending_actions = dict()

        self._last_obs_action = None

        self._num_updates = 0


    def get_action(self, observation: ndarray) -> ndarray:
        with torch.no_grad():
            observation = torch.tensor([observation], device=self._device)
            action, log_probs = self.policy.forward(observation, deterministic=self._test)

        if not self._test:
            self._last_obs_action = (observation,action)

        # print(f"aux_action={action}")

        return action.cpu().numpy()

    def tag_last_action(self, id:int):
        if self._test:
            return
        
        self.pending_actions[id] = self._last_obs_action
        self._last_obs_action=None

    def process_result(self, id:int, result:dict):
        if self._test:
            return
        
        self._num_updates += 1
        
        assert id in self.pending_actions
        self.policy.zero_grad()

        log_prob = self.policy.log_prob_of(*self.pending_actions[id])
        loss = -log_prob*result["reward"]

        if self._num_updates % 5000 == 0:
            print(f"loss at update {self._num_updates} is {loss}")

        loss.backward()

        self.policy_optim.step()

        del self.pending_actions[id]

    def reset(self):
        self.pending_actions = dict()

    def train_mode(self):
        self._test = False

    def test_mode(self):
        self._test = True


class AuxEnv:
    def __init__(self, aux_ai: AuxiliaryAlgo, env):
        self._aux_ai = aux_ai
        self._env = env

    def test_mode(self):
        self._aux_ai.test_mode()

    def train_mode(self):
        self._aux_ai.train_mode()

    def step(self, action: ndarray):
        final_action = action
        
        aux_obs = self._env.needs_aux(action)
        if aux_obs is not None:
            aux_action = self._aux_ai.get_action(aux_obs)
            final_action = (action, aux_action[0])

        obs, reward, done, info = self._env.step(final_action)

        if "aux" in info:
            if len(info["aux"]["new_trades"]) > 0:
                self._aux_ai.tag_last_action(info["aux"]["new_trades"].pop())
            if info["aux"]["observations"].items():
                for id, res in info["aux"]["observations"].items():
                    self._aux_ai.process_result(id, res)


        return (obs, reward, done, info)


    def reset(self):
        self._aux_ai.reset()
        return self._env.reset()

    def render(self):
        return self._env.render()

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space    

    
