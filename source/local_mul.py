"""Local controls-to-camera models. Synthetic coefficients are NOT calibration."""
import torch
from torch import nn


class LCDResponse(nn.Module):
    """Smooth intensity response of normalized continuous controls, not integer DAC codes.

    T(u) = black + (white-black)*u**gamma. Parameters may be per cell.
    This parametric curve is a starting model; measured LUTs can replace it.
    """
    def __init__(self, black=0., white=1., gamma=1.):
        super().__init__()
        for name, value in [('black', black), ('white', white), ('gamma', gamma)]:
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float64).detach().clone())
        if (not all(torch.isfinite(v).all() for v in [self.black, self.white, self.gamma])
                or (self.black < 0).any() or (self.white > 1).any()
                or (self.white < self.black).any() or (self.gamma <= 0).any()):
            raise ValueError('Invalid LCD response')

    def forward(self, controls):
        return self.black + (self.white-self.black)*controls.pow(self.gamma)


class LocalMul(nn.Module):
    """Independent cells, ideal imaging; common camera ROI normalization with OpticalMul.

    Gain is incident intensity times detector scale. Background is additive camera
    intensity in arbitrary units. No blur, phase, joint polarization or saturation.
    """
    def __init__(self, config, lcd1_response=None, lcd2_response=None,
                 gain=1., background=0.):
        super().__init__()
        self.config = config
        self.lcd1_response = lcd1_response if lcd1_response is not None else LCDResponse()
        self.lcd2_response = lcd2_response if lcd2_response is not None else LCDResponse()
        q = (torch.arange(config.samples, dtype=torch.float64)-(config.samples-1)/2)*config.dx
        masks = []
        for i in range(config.rows):
            row = []
            for j in range(config.columns):
                cy=(i-(config.rows-1)/2)*config.cell_pitch
                cx=(j-(config.columns-1)/2)*config.cell_pitch
                row.append(((q[:, None]-cy).abs() <= config.cell_width/2) &
                           ((q[None, :]-cx).abs() <= config.cell_width/2))
            masks.append(torch.stack(row))
        cells = torch.stack(masks).double()
        if (cells.sum((-2,-1)) == 0).any():
            raise ValueError('Grid does not resolve every cell')
        self.register_buffer('cells', cells)
        self.register_buffer('gain', torch.as_tensor(gain, dtype=torch.float64).detach().clone())
        self.register_buffer('background', torch.as_tensor(background, dtype=torch.float64).detach().clone())
        if not torch.isfinite(self.gain).all() or (self.gain < 0).any() or not torch.isfinite(self.background).all():
            raise ValueError('Invalid detector gain/background')
        self.magnification = config.magnifications[0]*config.magnifications[1]
        self.camera_dx = config.dx*abs(self.magnification)

    def camera_intensity(self, x, weights):
        if x.shape != (self.config.columns,) or weights.shape != (self.config.rows, self.config.columns):
            raise ValueError('Invalid control shapes')
        if (not torch.isfinite(x).all() or not torch.isfinite(weights).all()
                or (x < 0).any() or (x > 1).any() or (weights < 0).any() or (weights > 1).any()):
            raise ValueError('Normalized controls must be in [0,1]')
        values = self.gain*self.lcd1_response(x.expand(self.config.rows,-1))*self.lcd2_response(weights)
        image = torch.einsum('ij,ijhw->hw', values, self.cells)/self.magnification**2
        if self.magnification < 0:
            image = image.flip((-2,-1))
        return image + self.background

    def readout(self, intensity):
        cells = self.cells if self.magnification > 0 else self.cells.flip((-2,-1))
        powers = torch.einsum('hw,ijhw->ij', intensity, cells)*self.camera_dx**2
        area = self.cells.sum((-2,-1))*self.config.dx**2
        return (powers/area).sum(-1)

    def forward(self, x, weights):
        return self.readout(self.camera_intensity(x, weights))


class IdealMul(LocalMul):
    """Ideal local product and sum, rendered in the common camera coordinates."""
    def __init__(self, config):
        super().__init__(config)
