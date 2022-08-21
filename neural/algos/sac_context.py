class SACModels:
    def __init__(self):
        self._actor = None
        self._q_1 = None
        self._q_2 = None
        self._value = None
        self._target_value = None

        self._actor_optim = None
        self._q_1_optim = None
        self._q_2_optim = None
        self._value_optim = None

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
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value

    @property
    def target_value(self):
        return self._target_value

    @target_value.setter
    def target_value(self, new_target_value):
        self._target_value = new_target_value

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
    def value_optim(self):
        return self._value_optim

    @value_optim.setter
    def value_optim(self, new_value_optim):
        self._value_optim = new_value_optim


class SACShared(SACModels):
    def __init__(self):
        super().__init__()

    def _assign_or_copy(self, attr, new):
        current = getattr(self, attr)
        if current is None:
            setattr(self, attr, new)
        else:
            current.load_state_dict(new.state_dict())

    @SACModels.actor.setter
    def actor(self, new_actor):
        self._assign_or_copy("_actor", new_actor)

    @SACModels.q_1.setter
    def q_1(self, new_q_1):
        self._assign_or_copy("_q_1", new_q_1)

    @SACModels.q_2.setter
    def q_2(self, new_q_2):
        self._assign_or_copy("_q_2", new_q_2)

    @SACModels.value.setter
    def value(self, new_value):
        self._assign_or_copy("_value", new_value)

    @SACModels.target_value.setter
    def target_value(self, new_target_value):
        self._assign_or_copy("_target_value", new_target_value)

    @SACModels.actor_optim.setter
    def actor_optim(self, new_actor_optim):
        self._assign_or_copy("_actor_optim", new_actor_optim)

    @SACModels.q_1_optim.setter
    def q_1_optim(self, new_q_1_optim):
        self._assign_or_copy("_q_1_optim", new_q_1_optim)

    @SACModels.q_2_optim.setter
    def q_2_optim(self, new_q_2_optim):
        self._assign_or_copy("_q_2_optim", new_q_2_optim)

    @SACModels.value_optim.setter
    def value_optim(self, new_value_optim):
        self._assign_or_copy("_value_optim", new_value_optim)


class SACContext(SACModels):
    def __init__(self):
        self._actor_loss_fn = None
        self._q_1_loss_fn = None
        self._q_2_loss_fn = None
        self._value_loss_fn = None
        self._target_value_loss_fn = None

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
    def value_loss_fn(self):
        return self._value_loss_fn

    @value_loss_fn.setter
    def value_loss_fn(self, new_value_loss_fn):
        self._value_loss_fn = new_value_loss_fn

    @property
    def target_value_loss_fn(self):
        return self._target_value_loss_fn

    @target_value_loss_fn.setter
    def target_value_loss_fn(self, new_target_value_loss_fn):
        self._target_value_loss_fn = new_target_value_loss_fn

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
        self.shared.value = self.value
        self.shared.target_value = self.target_value

        self.shared.actor_optim = self.actor_optim
        self.shared.q_1_optim = self.q_1_optim
        self.shared.q_2_optim = self.q_2_optim
        self.shared.value_optim = self.value_optim
