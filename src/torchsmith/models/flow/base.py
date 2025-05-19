from abc import ABC
from abc import abstractmethod

import torch


class ODE(ABC):
    """Ordinary Differential Equation.
    Let :math:`u` be a time-dependent vector field.
    .. math::
        u: \\mathbb{R}^d \\times [0,1] \\to \mathbb{R}^d, \\quad (x,t) \\mapsto u_t(x)

    Then the ODE is given by:

    .. math::
        dX_t = u_t(X_t)dt, \\quad X_0 = x_0.

    where :math:`u_t(.)` is the drift coefficient.
    """

    @abstractmethod
    def drift_coefficient(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class SDE(ABC):
    """Stochastic Differential Equation.
    Let :math:`u` be a time-dependent vector field.
    .. math::
        u: \\mathbb{R}^d \\times [0,1] \\to \mathbb{R}^d, \\quad (x,t) \\mapsto u_t(x)

    and Brownian motion :math:`$(W_t)_{0 \le t \le 1}`

    Then the SDE is given by:

    .. math::
        dX_t = u_t(X_t)dt + \\sigma_t d W_t, \\quad X_0 = x_0.

    where :math:`u_t(.)` is the drift coefficient
    and :math:`\\sigma_t(.)` is the diffusion coefficient.
    """

    @abstractmethod
    def drift_coefficient(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def diffusion_coefficient(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
