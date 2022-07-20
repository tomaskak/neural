from neural.model.model import Model

from neural.model.modules import Activation, Layer

import torch


class TestModel:
    def test_model_forward(self):
        mod = Model(
            "test",
            [
                Layer("l1", torch.nn.Linear(4, 4)),
                Activation("tanh", torch.nn.Tanh()),
                Layer("l1", torch.nn.Linear(4, 1)),
            ],
        )

        loss_fn = torch.nn.MSELoss()

        target = 5.5
        x = [1.0, 2.0, 0.0, -3.0]
        for _ in range(10):
            mod.zero_grad()
            out = mod.forward(torch.tensor(x))
            loss = loss_fn(out, torch.tensor([target]))
            loss.backward()

            with torch.no_grad():
                for param in mod.parameters():
                    param -= (1.0 / 20.0) * param.grad

        assert abs(mod.forward(torch.tensor(x)).item() - target) < 0.1
