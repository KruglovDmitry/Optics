"""Nonnegative VMM: optical local products, digital row integration."""
import torch
from torch import nn
from .config import Config
import math
from collections import OrderedDict
from .config import ConfigOpticBase
from .propagator import (DoublePlane, PropagatorSinc,
                         PropagatorSphericalLens, PropagatorFieldNormalization, PropagatorPhaseScreen)


class OpticalMul(nn.Module):
    def __init__(self, config, illumination_fields=None, illumination_weights=None,
                 lcd1_response=None, lcd2_response=None, checkpoint_modes=False):
        super().__init__()
        self.config = config
        self.lcd1_response = lcd1_response if lcd1_response is not None else nn.Identity()
        self.lcd2_response = lcd2_response if lcd2_response is not None else nn.Identity()
        self.checkpoint_modes = checkpoint_modes
        q = (torch.arange(config.samples, dtype=torch.float64)-(config.samples-1)/2)*config.dx
        self.register_buffer('coordinate', q)
        masks = []
        for i in range(config.rows):
            cy = (i-(config.rows-1)/2)*config.cell_pitch
            for j in range(config.columns):
                cx = (j-(config.columns-1)/2)*config.cell_pitch
                masks.append(((q[:, None]-cy).abs() <= config.cell_width/2) &
                             ((q[None, :]-cx).abs() <= config.cell_width/2))
        masks = torch.stack(masks).double().reshape(config.rows, config.columns,
                                                   config.samples, config.samples)
        if (masks.sum((-2, -1)) == 0).any():
            raise ValueError('Grid does not resolve every cell')
        self.register_buffer('cells', masks)
        m1, m2 = config.magnifications
        width2 = config.field_width*abs(m1)
        self.camera_dx = config.dx*abs(m1*m2)
        report = config.sampling_report()
        if not report['lens_phase_sampling_ok']:
            raise ValueError(f'Lens phase undersampled: {report}. '
                             'Increase lens_samples; do not interpret a coarse run physically.')

        # Fixed coherent source -> D -> L0 -> LCD1, computed on CPU once.
        # Only the resulting field is retained, not the large source operators.
        if illumination_fields is None:
            if illumination_weights is not None:
                raise ValueError('illumination_weights require illumination_fields')
            self.register_buffer('illumination_field', self._prepare_illumination())
            self.register_buffer('illumination_fields', None)
            self.register_buffer('illumination_weights', None)
        else:
            fields = torch.as_tensor(illumination_fields, dtype=torch.complex128).detach().clone()
            if fields.ndim != 3 or fields.shape[1:] != (config.samples, config.samples) or len(fields) == 0:
                raise ValueError('Expected illumination_fields[K,samples,samples] at LCD1')
            weights = (torch.ones(len(fields), dtype=torch.float64, device=fields.device)/len(fields)
                       if illumination_weights is None else
                       torch.as_tensor(illumination_weights, dtype=torch.float64, device=fields.device).detach().clone())
            if (weights.shape != (len(fields),) or not torch.isfinite(weights).all()
                    or (weights < 0).any() or weights.sum() <= 0 or not torch.isfinite(fields).all()):
                raise ValueError('Invalid illumination fields/weights')
            # Weights are physical quadrature/power weights; do not renormalize them.
            self.register_buffer('illumination_field', None)
            self.register_buffer('illumination_fields', fields)
            self.register_buffer('illumination_weights', weights)

        input_plane = DoublePlane(config.samples, aperture=config.field_width)
        lens_plane = DoublePlane(config.lens_samples, aperture=config.clear_aperture)
        matrix_plane = DoublePlane(config.samples, aperture=width2)
        output_width = width2 * abs(m2)
        output_plane = DoublePlane(config.samples, aperture=output_width)

        # LCD1 -> L1 -> L2 -> LCD2. Same explicit construction style as
        # the original OpticalMul, with spherical lenses and circular pupils.
        prop_one = PropagatorSinc(input_plane, lens_plane,
                                 ConfigOpticBase(config.wavelength, config.focal_1))
        prop_two = PropagatorSphericalLens(lens_plane, config.wavelength,
                                          config.focal_1, config.clear_aperture)
        prop_three = PropagatorSinc(lens_plane, lens_plane,
                                   ConfigOpticBase(config.wavelength, config.focal_1+config.focal_2))
        prop_four = PropagatorSphericalLens(lens_plane, config.wavelength,
                                           config.focal_2, config.clear_aperture)
        prop_five = PropagatorSinc(lens_plane, matrix_plane,
                                  ConfigOpticBase(config.wavelength, config.focal_2))

        # LCD2 -> L3 -> L4 -> camera.
        prop_six = PropagatorSinc(matrix_plane, lens_plane,
                                 ConfigOpticBase(config.wavelength, config.focal_3))
        prop_seven = PropagatorSphericalLens(lens_plane, config.wavelength,
                                            config.focal_3, config.clear_aperture)
        prop_eight = PropagatorSinc(lens_plane, lens_plane,
                                   ConfigOpticBase(config.wavelength, config.focal_3+config.focal_4))
        prop_nine = PropagatorSphericalLens(lens_plane, config.wavelength,
                                           config.focal_4, config.clear_aperture)
        prop_ten = PropagatorSinc(lens_plane, output_plane,
                                 ConfigOpticBase(config.wavelength, config.focal_4))

        # Circular pupils cannot be collapsed using the separable '+'
        # operator of the original repository; evaluate all elements in order.
        self._propagator_one = nn.Sequential(OrderedDict([
            ('LCD1_to_L1', prop_one), ('L1', prop_two),
            ('L1_to_L2', prop_three), ('L2', prop_four),
            ('L2_to_LCD2', prop_five),
            ('field_normalization', PropagatorFieldNormalization(config.field_width/width2)),
        ]))
        self._propagator_two = nn.Sequential(OrderedDict([
            ('LCD2_to_L3', prop_six), ('L3', prop_seven),
            ('L3_to_L4', prop_eight), ('L4', prop_nine),
            ('L4_to_camera', prop_ten),
            ('field_normalization', PropagatorFieldNormalization(width2/output_width)),
        ]))

    def _validate(self, x, weights):
        c = self.config
        if x.shape != (c.columns,) or weights.shape != (c.rows, c.columns):
            raise ValueError(f'Expected x[{c.columns}], W[{c.rows},{c.columns}]')
        if not torch.isfinite(x).all() or not torch.isfinite(weights).all():
            raise ValueError('Inputs must be finite')
        if (x < 0).any() or (x > 1).any() or (weights < 0).any() or (weights > 1).any():
            raise ValueError('This prototype accepts intensity transmissions in [0,1]')

    def camera_intensity(self, x, weights, source_field=None):
        self._validate(x, weights)
        # sqrt is essential: x,W encode INTENSITY transmissions, not amplitude.
        t1 = self.lcd1_response(x.expand(self.config.rows, -1))
        t2 = self.lcd2_response(weights)
        if (not torch.isfinite(t1).all() or not torch.isfinite(t2).all()
                or (t1 < 0).any() or (t1 > 1).any() or (t2 < 0).any() or (t2 > 1).any()):
            raise ValueError('LCD responses must be finite intensity transmissions in [0,1]')
        mask1 = torch.einsum('ij,ijhw->hw', torch.sqrt(t1), self.cells)
        mask2 = torch.einsum('ij,ijhw->hw', torch.sqrt(t2), self.cells)
        # Coordinates of LCD2 are inverted by the first relay. This compensates
        # the displayed weight image, rather than accidentally reversing W.
        if self.config.magnifications[0] < 0:
            mask2 = mask2.flip((-2, -1))
        def propagate(first_mask, second_mask, illumination):
            field = self._propagator_one(illumination * first_mask)
            return self._propagator_two(field * second_mask).abs().square()

        def one(illumination):
            if self.checkpoint_modes and torch.is_grad_enabled() and (mask1.requires_grad or mask2.requires_grad):
                from torch.utils.checkpoint import checkpoint
                return checkpoint(propagate, mask1, mask2, illumination, use_reentrant=False)
            return propagate(mask1, mask2, illumination)

        if source_field is not None:
            if source_field.shape != mask1.shape:
                raise ValueError('source_field must match the simulation grid')
            return one(source_field)
        if self.illumination_fields is None:
            return one(self.illumination_field)
        intensity = torch.zeros_like(mask1)
        for field, power in zip(self.illumination_fields, self.illumination_weights):
            intensity = intensity + power * one(field)
        return intensity

    def readout(self, intensity):
        # Both relays invert: default overall magnification +0.5.
        cells = self.cells
        if self.config.magnifications[0]*self.config.magnifications[1] < 0:
            cells = cells.flip((-2, -1))
        powers = torch.einsum('hw,ijhw->ij', intensity, cells)*self.camera_dx**2
        # Fixed geometry-only gain (ideal equal incident irradiance = 1).
        # No per-example fitting against W@x. Raster area differs at cell edges.
        reference_area = self.cells.sum((-2, -1))*self.config.dx**2
        return (powers/reference_area).sum(-1)

    def forward(self, x, weights, source_field=None):
        return self.readout(self.camera_intensity(x, weights, source_field))

    @staticmethod
    def _diffuser_phase(plane, rms, correlation_length, seed, modes=32):
        """Fixed continuous random Fourier profile sampled on the supplied grid.

        Seed fixes spatial frequencies and phases, not a pixel-noise image. Thus
        changing the sampling grid does not change the underlying diffuser.
        rms and correlation length describe its statistical model, not exact
        empirical statistics of one finite aperture and 32 Fourier features.
        """
        gen = torch.Generator().manual_seed(seed)
        frequencies = torch.randn((modes, 2), generator=gen, dtype=torch.float64)/correlation_length
        offsets = 2*math.pi*torch.rand(modes, generator=gen, dtype=torch.float64)
        y = plane.linspace_by_y[:, None]
        x = plane.linspace_by_x[None, :]
        phase = torch.zeros((len(y), x.shape[1]), dtype=torch.float64)
        for (kx, ky), offset in zip(frequencies, offsets):
            phase += torch.cos(kx*x + ky*y + offset)
        return phase * (rms*math.sqrt(2/modes))

    @torch.no_grad()
    def _prepare_illumination(self):
        """Return the fixed field density at LCD1; release illumination operators.

        No normalization to W@x or flat-field correction is performed. Gaussian
        source amplitude uses the same arbitrary field units as the rest of the
        model. This cache is fixed for gradients with respect to LCD transmissions;
        illumination parameter gradients are not calculated by this function.
        """
        config = self.config
        report = config.illumination_sampling_report()
        if not report['lens_phase_sampling_ok']:
            raise ValueError(f'L0 phase undersampled: {report}')
        source_plane = DoublePlane(config.illumination_samples, aperture=config.illumination_width)
        diffuser_plane = DoublePlane(config.illumination_samples, aperture=config.illumination_width)
        lens_plane = DoublePlane(config.illumination_lens_samples, aperture=config.clear_aperture)
        lcd_plane = DoublePlane(config.samples, aperture=config.field_width)

        yy, xx = torch.meshgrid(source_plane.linspace_by_y, source_plane.linspace_by_x, indexing='ij')
        source = config.source_amplitude * torch.exp(-(xx.square()+yy.square())/config.source_waist**2)
        source = source.to(torch.complex128)
        phase = self._diffuser_phase(diffuser_plane, config.diffuser_phase_rms,
                               config.diffuser_correlation_length, config.diffuser_seed)

        prop_one = PropagatorSinc(source_plane, diffuser_plane,
                                 ConfigOpticBase(config.wavelength, config.source_to_diffuser))
        prop_two = PropagatorPhaseScreen(diffuser_plane, phase, config.diffuser_diameter)
        prop_three = PropagatorSinc(diffuser_plane, lens_plane,
                                   ConfigOpticBase(config.wavelength, config.diffuser_to_lens))
        prop_four = PropagatorSphericalLens(lens_plane, config.wavelength,
                                           config.focal_0, config.clear_aperture)
        prop_five = PropagatorSinc(lens_plane, lcd_plane,
                                  ConfigOpticBase(config.wavelength, config.lens_to_lcd))
        # Sinc uses sqrt(sample-area)-normalized coefficients, including on grids
        # with different sample COUNTS. For square grids convert by dx_in/dx_out.
        scale = source_plane.pixel_size_by_x/lcd_plane.pixel_size_by_x
        chain = nn.Sequential(OrderedDict([
            ('source_to_D', prop_one), ('D', prop_two), ('D_to_L0', prop_three),
            ('L0', prop_four), ('L0_to_LCD1', prop_five),
            ('field_normalization', PropagatorFieldNormalization(scale)),
        ]))
        field = chain(source)
        if not torch.isfinite(field).all():
            raise RuntimeError('Illumination field contains non-finite values')
        return field.detach().contiguous()

    @staticmethod
    def angular_disk(config, angular_radius, radial=2, azimuthal=8):
        """Uniform angular disk in paraxial direction-cosine coordinates, unit total weight.

        Gauss-Legendre quadrature in radius squared, uniform azimuth. One wavelength,
        uniform irradiance envelope. Does not model L0 clipping or a real backlight.
        """
        if not math.isfinite(angular_radius) or not 0 <= angular_radius < 1 or radial < 1 or azimuthal < 1:
            raise ValueError('Invalid angular source quadrature')
        import numpy as np
        q = (torch.arange(config.samples,dtype=torch.float64)-(config.samples-1)/2)*config.dx
        if angular_radius == 0:
            return torch.ones((1,config.samples,config.samples),dtype=torch.complex128), torch.ones(1,dtype=torch.float64)
        nodes, weights = np.polynomial.legendre.leggauss(radial)
        fields=[]; powers=[]
        for node, weight in zip(nodes, weights):
            radius = angular_radius*math.sqrt((node+1)/2)
            for k in range(azimuthal):
                theta=2*math.pi*k/azimuthal
                phase=2*math.pi/config.wavelength*radius*(math.cos(theta)*q[None,:]+math.sin(theta)*q[:,None])
                fields.append(torch.exp(1j*phase))
                powers.append(weight/(2*azimuthal))
        return torch.stack(fields), torch.tensor(powers,dtype=torch.float64)
