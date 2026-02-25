import torch
from src.models.Gatr_pf_e_noise import ExampleWrapper


class GraphTransformerNetWrapper(torch.nn.Module):
    def __init__(self, args, dev, **kwargs) -> None:
        super().__init__()
        self.mod = ExampleWrapper(args, dev, **kwargs)

    def forward(self, g, y, step_count, **kwargs):
        return self.mod(g, y, step_count, **kwargs)


def get_model(data_config, args, dev, **kwargs):
    model = GraphTransformerNetWrapper(args, dev, **kwargs)
    model_info = {}
    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.MSELoss()
