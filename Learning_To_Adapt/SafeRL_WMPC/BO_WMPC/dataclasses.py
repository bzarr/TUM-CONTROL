import torch
from dataclasses import dataclass


@dataclass
class Dataset:
    """ Class to combine two tensors that represent a dataset, mapping from a
    tensor of parametrizations X to a tensor of observations Y. Used for both
    the objective and the constraint satisfaction data."""
    
    X: torch.tensor
    Y: torch.tensor

    def add_data(self, x, y):
        self.X = torch.vstack((self.X, x.to(self.X)))
        self.Y = torch.vstack((self.Y, y.to(self.Y)))


@dataclass(eq=False, order=False, frozen=False)
class Trial:
    """ Class to hold information on each experiment trial, i.e. each
    parametrization that has been evaluated. Stores the parametrization in
    natural and normalized format, as well as the resulting objective values
    and a flag indicating feasibility."""

    id: int
    params_nat: torch.tensor
    params_nor: torch.tensor
    objectives: torch.tensor
    feasible: bool

    """ Put trial information into readable format. """
    def __repr__(self):
        params = self.params_nat.squeeze().cpu().numpy()
        objectives_0 = self.objectives[0].cpu().numpy()
        objectives_1 = self.objectives[1].cpu().numpy()

        s = "[{}] -> [{} | {}] | {}".format(
            ', '.join([f'{e:5.1f}' for e in params]),
            ', '.join([f'{e:2.2f}' for e in objectives_0]),
            ', '.join([f'{e:2.2f}' for e in objectives_1]),
            'success' if self.feasible else 'failure'
        )     
        return s
    
    """ Put trial information into csv-compatible format. """
    def to_csv(self):
        params_nat = self.params_nat.squeeze().cpu().numpy()
        params_nor = self.params_nor.squeeze().cpu().numpy()
        objectives_0 = self.objectives[0].cpu().numpy()
        objectives_1 = self.objectives[1].cpu().numpy()

        s = "{};{};{};{};{}\n".format(
            ','.join([f'{e:1.1f}' for e in params_nat]),
            ','.join([f'{e:1.3f}' for e in params_nor]),
            ','.join([f'{e:1.3f}' for e in objectives_0]),
            ','.join([f'{e:1.3f}' for e in objectives_1]),
            1 if self.feasible else 0
        )     
        return s