import torch
from copy import deepcopy


class SACCore:
    def __init__(self):
        self._actor = None
        self._q_1 = None
        self._q_2 = None
        self._q_1_target = None
        self._q_2_target = None

        self._entropy_weight = None

        self._actor_optim = None
        self._q_1_optim = None
        self._q_2_optim = None
        self._entropy_weight_optim = None

    @property
    def actor(self):
        return self._actor

    @actor.setter
    def actor(self, new_actor):
        self._actor = new_actor

    @property
    def q_1(self):
        return self._q_1

    @q_1.setter
    def q_1(self, new_q_1):
        self._q_1 = new_q_1

    @property
    def q_2(self):
        return self._q_2

    @q_2.setter
    def q_2(self, new_q_2):
        self._q_2 = new_q_2

    @property
    def q_1_target(self):
        return self._q_1_target

    @q_1_target.setter
    def q_1_target(self, new_q_1_target):
        self._q_1_target = new_q_1_target

    @property
    def q_2_target(self):
        return self._q_2_target

    @q_2_target.setter
    def q_2_target(self, new_q_2_target):
        self._q_2_target = new_q_2_target

    @property
    def entropy_weight(self):
        return self._entropy_weight

    @entropy_weight.setter
    def entropy_weight(self, new_entropy_weight):
        self._entropy_weight = new_entropy_weight

    @property
    def actor_optim(self):
        return self._actor_optim

    @actor_optim.setter
    def actor_optim(self, new_actor_optim):
        self._actor_optim = new_actor_optim

    @property
    def q_1_optim(self):
        return self._q_1_optim

    @q_1_optim.setter
    def q_1_optim(self, new_q_1_optim):
        self._q_1_optim = new_q_1_optim

    @property
    def q_2_optim(self):
        return self._q_2_optim

    @q_2_optim.setter
    def q_2_optim(self, new_q_2_optim):
        self._q_2_optim = new_q_2_optim

    @property
    def entropy_weight_optim(self):
        return self._entropy_weight_optim

    @entropy_weight_optim.setter
    def entropy_weight_optim(self, new_entropy_weight_optim):
        self._entropy_weight_optim = new_entropy_weight_optim


class SACShared(SACCore):
    def __init__(self):
        super().__init__()

    def _assign_or_copy(self, attr, new):
        current = getattr(self, attr)
        if current is None:
            setattr(self, attr, deepcopy(new))
        else:
            with torch.no_grad():
                for old, n in zip(current.parameters(), new.parameters()):
                    old[:] = n[:]

    def _assign_or_copy_optim(self, attr, new):
        current = getattr(self, attr)
        if current is None:
            setattr(self, attr, deepcopy(new))
        else:
            current.load_state_dict(new.state_dict())

    @SACCore.actor.setter
    def actor(self, new_actor):
        self._assign_or_copy("_actor", new_actor)

    @SACCore.q_1.setter
    def q_1(self, new_q_1):
        self._assign_or_copy("_q_1", new_q_1)

    @SACCore.q_2.setter
    def q_2(self, new_q_2):
        self._assign_or_copy("_q_2", new_q_2)

    @SACCore.q_1_target.setter
    def q_1_target(self, new_q_1_target):
        self._assign_or_copy("_q_1_target", new_q_1_target)

    @SACCore.q_2_target.setter
    def q_2_target(self, new_q_2_target):
        self._assign_or_copy("_q_2_target", new_q_2_target)

    @SACCore.entropy_weight.setter
    def entropy_weight(self, new_entropy_weight):
        with torch.no_grad():
            if self._entropy_weight is None:
                self._entropy_weight = new_entropy_weight
            else:
                self._entropy_weight.copy_(new_entropy_weight)

    @SACCore.actor_optim.setter
    def actor_optim(self, new_actor_optim):
        self._assign_or_copy_optim("_actor_optim", new_actor_optim)

    @SACCore.q_1_optim.setter
    def q_1_optim(self, new_q_1_optim):
        self._assign_or_copy_optim("_q_1_optim", new_q_1_optim)

    @SACCore.q_2_optim.setter
    def q_2_optim(self, new_q_2_optim):
        self._assign_or_copy_optim("_q_2_optim", new_q_2_optim)

    @SACCore.entropy_weight_optim.setter
    def entropy_weight_optim(self, new_entropy_weight_optim):
        self._assign_or_copy_optim("_entropy_weight_optim", new_entropy_weight_optim)


class SACContext(SACCore):
    def __init__(self):
        self._actor_loss_fn = None
        self._q_1_loss_fn = None
        self._q_2_loss_fn = None

        self._shared = SACShared()

    @property
    def actor_loss_fn(self):
        return self._actor_loss_fn

    @actor_loss_fn.setter
    def actor_loss_fn(self, new_actor_loss_fn):
        self._actor_loss_fn = new_actor_loss_fn

    @property
    def q_1_loss_fn(self):
        return self._q_1_loss_fn

    @q_1_loss_fn.setter
    def q_1_loss_fn(self, new_q_1_loss_fn):
        self._q_1_loss_fn = new_q_1_loss_fn

    @property
    def q_2_loss_fn(self):
        return self._q_2_loss_fn

    @q_2_loss_fn.setter
    def q_2_loss_fn(self, new_q_2_loss_fn):
        self._q_2_loss_fn = new_q_2_loss_fn

    @property
    def shared(self):
        return self._shared

    @shared.setter
    def shared(self, shared):
        self._shared = shared

    def update_shared(self):
        self.shared.actor = self.actor
        self.shared.q_1 = self.q_1
        self.shared.q_2 = self.q_2
        self.shared.q_1_target = self.q_1_target
        self.shared.q_2_target = self.q_2_target
        self.shared.entropy_weight = self.entropy_weight

        self.shared.actor_optim = self.actor_optim
        self.shared.q_1_optim = self.q_1_optim
        self.shared.q_2_optim = self.q_2_optim
        self.shared.entropy_weight_optim = self.entropy_weight_optim

    def to(self, device: str):
        self.actor.to(device)
        self.q_1.to(device)
        self.q_2.to(device)
        self.q_1_target.to(device)
        self.q_2_target.to(device)
        self.entropy_weight = self.entropy_weight.to(device)
