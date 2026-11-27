import torch as _torch
import numpy as _np
from abc import ABC as _ABC
from typing import Tuple as _Tuple
import collections as _collections
import pickle as _pickle

class ConfigOpticBase(_ABC):
    """
    Абстрактный класс базовой информации о работе опттической установки.

    Поля:
        wavelength: длина волны, используемая в оптической установке.
        K: волновое число.
        distance: дистанция распространения светового поля между плоскостями.
    """
    def __init__(self, wavelength: float, distance: float):
        """
        Конструктор класса базовой информации о работе опттической установки.

        Args:
            wavelength: длина волны, используемая в оптической установке.
            distance: дистанция распространения светового поля между плоскостями.
        """
        self._wavelength: float = wavelength
        self._K: float = 2 * _np.pi / wavelength
        self._distance = distance

    @property
    def wavelength(self) -> float:
        """
        Returns: длинна волны.
        """
        return self._wavelength

    @property
    def K(self) -> float:
        """
        Returns: волновое число.
        """
        return self._K
    
    @property
    def distance(self) -> float:
        """
        Returns: дистанция распространения.
        """
        return self._distance

class ConfigDesignPlane:
    """
    Класс данных о расчётной плоскости.
    """
    def __init__(self,
                 pixel_count: None | int | _Tuple[int, int] = None,
                 pixel_size: None | float | _Tuple[float, float] = None,
                 aperture: None | float | _Tuple[float, float] = None
                ):
        """
        Конструктор класса данных о расчётной плоскости.

        Args:
            pixel_count: данные о кол-ве пикселей в измерении.
            pixel_size: данные о размере пикселей в измерении.
            aperture: данные о апертуре измерения.

        Note:
            1. Достаточно указать два из трёх входных аргумента.
            2. Если было переданно одно число, будет считаться, что значение у каждого измерения совпадают,
            если было переданно два числа, то первое число будет ассоциированно с Y измерением, второе с X.
        """
        if aperture is None:
            aperture = ConfigDesignPlane.__get_excluded_third(pixel_count, pixel_size, False)
        elif pixel_count is None:
            pixel_count = ConfigDesignPlane.__get_excluded_third(aperture, pixel_size)
        elif pixel_size is None:
            pixel_size = ConfigDesignPlane.__get_excluded_third(aperture, pixel_count)            

        self._pixel_count: int | _Tuple[int, int] = pixel_count
        self._pixel_size: float | _Tuple[float, float] = pixel_size
        self._aperture: float | _Tuple[float, float] = aperture
    
    def transpose(self) -> 'ConfigDesignPlane':
        """
        Метод транспонирования данных о расчётной плоскости.
        Returns:
           Новые данне о расчётной плоскости, транспонированные относительно текущих.
        """
        data_pixel_count = self.pixel_count
        data_pixel_size = self.pixel_size
        data_aperture = self.aperture
        
        obj = ConfigDesignPlane([data_pixel_count[1], data_pixel_count[0]],
                                [data_pixel_size[1], data_pixel_size[0]],
                                [data_aperture[1], data_aperture[0]])
        return obj

    @staticmethod
    def __get_excluded_third(first_element, second_element, div: bool = True):
        ConfigDesignPlane.__check_for_none(first_element, second_element)
        first_element_y, first_element_x = ConfigDesignPlane.__return_tuple(first_element)
        second_element_y, second_element_x = ConfigDesignPlane.__return_tuple(second_element)
        if div:
            return first_element_y / second_element_y, first_element_x / second_element_x
        else:
            return first_element_y * second_element_y, first_element_x * second_element_x
    
    @staticmethod
    def __check_for_none(first_element, second_element):
        if (first_element is None) or (second_element is None):
            raise TypeError("One of the provided elements is None, it is not possible to obtain the full dimensions of the calculated plane.")

    @staticmethod
    def __return_element(element: int | _Tuple[int, int], dim: int = 0):
        if isinstance(element, _collections.abc.Sequence):
            return element[dim]
        else:
            return element

    @staticmethod
    def __return_tuple(element: int | _Tuple[int, int]):
        if isinstance(element, _collections.abc.Sequence):
            return element
        else:
            return element, element

    @staticmethod
    def __get_linspace_by_dim(aperture, pixel_count):
        linspace = _torch.linspace(-aperture / 2, aperture / 2, pixel_count + 1)[:pixel_count]
        linspace += aperture / (2 * pixel_count)
        return linspace

    @property
    def pixel_count_by_x(self) -> float:
        """
        Returns:
            Информация о кол-ве пикселей по оси X.
        """
        return ConfigDesignPlane.__return_element(self._pixel_count, 1)
    @property
    def pixel_count_by_y(self) -> float:
        """
        Returns:
            Информация о кол-ве пикселей по оси Y.
        """
        return ConfigDesignPlane.__return_element(self._pixel_count)
    @property
    def pixel_count(self) -> _Tuple[float, float]:
        """
        Returns:
            Информация о кол-ве пикселей по каждой оси [Y, x].
        """
        return ConfigDesignPlane.__return_tuple(self._pixel_count)
    @property
    def pixel_size_by_x(self) -> int:
        """
        Returns:
            Информация о размере пикселей по оси X.
        """
        return ConfigDesignPlane.__return_element(self._pixel_size, 1)
    @property
    def pixel_size_by_y(self) -> int:
        """
        Returns:
            Информация о размере пикселей по оси Y.
        """
        return ConfigDesignPlane.__return_element(self._pixel_size)
    @property
    def pixel_size(self) -> _Tuple[int, int]:
        """
        Returns:
            Информация о размере пикселей по каждой оси [Y, x].
        """
        return ConfigDesignPlane.__return_tuple(self._pixel_size)
    @property
    def aperture_width(self) -> float:
        """
        Returns:
            Информация о ширине расчётной плоскасти.
        """
        return ConfigDesignPlane.__return_element(self._aperture, 1)
    @property
    def aperture_height(self) -> float:
        """
        Returns:
            Информация о высоте расчётной плоскасти.
        """
        return ConfigDesignPlane.__return_element(self._aperture)
    @property
    def aperture(self) -> _Tuple[float, float]:
        """
        Returns:
            Информация о высоте и ширине расчётной плоскасти [H, W].
        """
        return ConfigDesignPlane.__return_tuple(self._aperture)
    @property
    def linspace_by_x(self) -> _torch.Tensor:
        """
        Returns:
            Расчётная сетка по оси X.
        """
        return ConfigDesignPlane.__get_linspace_by_dim(self.aperture_width, self.pixel_count_by_x)
    @property
    def linspace_by_y(self) -> _torch.Tensor:
        """
        Returns:
            Расчётная сетка по оси Y.
        """
        return ConfigDesignPlane.__get_linspace_by_dim(self.aperture_height, self.pixel_count_by_y)
    @property
    def meshgrid(self) -> _Tuple[_torch.Tensor, _torch.Tensor]:
        """
        Returns:
            Расчётная сетка по осям [Y, X].
        """
        linspace_by_x = self.linspace_by_x
        linspace_by_y = self.linspace_by_y
        return _torch.meshgrid((linspace_by_y, linspace_by_x))
    @property
    def T(self) -> 'ConfigDesignPlane':
        """
        Returns:
           Новые данне о расчётной плоскости, транспонированные относительно текущих.
        """
        return self.transpose()



# Configuration of the LCD-LCD-camera installation.
"""SI units throughout. Geometry from LCD_PAT_optical_design_v1.pdf."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Config:
    wavelength: float = 532e-9
    rows: int = 4
    columns: int = 8
    lcd_pixel_pitch: float = 135e-6
    cell_pixels: int = 4
    pitch_pixels: int = 8
    field_width: float = 10.368e-3
    samples: int = 512
    focal_1: float = 100e-3
    focal_2: float = 100e-3
    focal_3: float = 100e-3
    focal_4: float = 50e-3
    lens_diameter: float = 25.4e-3
    clear_aperture: float = 22e-3
    lens_samples: int = 18432

    # Illumination geometry from the stand drawing; source is coherent here.
    source_to_diffuser: float = 30e-3
    diffuser_to_lens: float = 100e-3
    lens_to_lcd: float = 100e-3
    focal_0: float = 100e-3
    diffuser_diameter: float = 4e-3
    illumination_width: float = 6e-3
    illumination_samples: int = 1024
    illumination_lens_samples: int = 12288
    # Synthetic Gaussian beam and fixed diffuser; not measured hardware data.
    source_waist: float = .5e-3  # field-amplitude 1/e radius
    source_amplitude: float = 1.0
    diffuser_phase_rms: float = math.pi
    diffuser_correlation_length: float = 20e-6
    diffuser_seed: int = 42

    def __post_init__(self):
        if min(self.rows, self.columns, self.samples, self.lens_samples,
               self.cell_pixels, self.pitch_pixels) <= 0:
            raise ValueError('Counts must be positive')
        if self.cell_pixels > self.pitch_pixels:
            raise ValueError('Cells overlap')
        if not 0 < self.clear_aperture <= self.lens_diameter:
            raise ValueError('Invalid clear aperture')
        if min(self.wavelength, self.field_width, self.lcd_pixel_pitch,
               self.focal_1, self.focal_2, self.focal_3, self.focal_4) <= 0:
            raise ValueError('Lengths must be positive')
        if self.extent(max(self.rows, self.columns)) >= self.field_width:
            raise ValueError('Encoded matrix does not fit in the simulation field')

        values = (self.source_to_diffuser, self.diffuser_to_lens,
                  self.lens_to_lcd, self.focal_0, self.diffuser_diameter,
                  self.illumination_width, self.source_waist,
                  self.diffuser_correlation_length)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('Illumination lengths must be finite and positive')
        if min(self.illumination_samples, self.illumination_lens_samples) < 2:
            raise ValueError('Illumination grids require at least two samples')
        if self.diffuser_diameter > self.illumination_width:
            raise ValueError('Diffuser aperture must fit in its calculation plane')
        if not math.isfinite(self.diffuser_phase_rms) or self.diffuser_phase_rms < 0:
            raise ValueError('Diffuser phase rms must be finite and nonnegative')
        if not math.isfinite(self.source_amplitude) or self.source_amplitude < 0:
            raise ValueError('Source amplitude must be finite and nonnegative')

    @property
    def dx(self):
        return self.field_width / self.samples

    @property
    def cell_width(self):
        return self.lcd_pixel_pitch * self.cell_pixels

    @property
    def cell_pitch(self):
        return self.lcd_pixel_pitch * self.pitch_pixels

    def extent(self, count):
        return (count - 1) * self.cell_pitch + self.cell_width

    @property
    def magnifications(self):
        return -self.focal_2 / self.focal_1, -self.focal_4 / self.focal_3

    def planes(self):
        a = self.focal_1
        b = a + self.focal_1 + self.focal_2
        c = b + self.focal_2
        d = c + self.focal_3
        e = d + self.focal_3 + self.focal_4
        return [('source', -(self.source_to_diffuser+self.diffuser_to_lens+self.lens_to_lcd)),
                ('D', -(self.diffuser_to_lens+self.lens_to_lcd)),
                ('L0', -self.lens_to_lcd),
                ('LCD1', 0.), ('L1', a), ('L2', b), ('LCD2', c),
                ('L3', d), ('L4', e), ('camera', e + self.focal_4)]

    def illumination_sampling_report(self):
        required = math.ceil(self.clear_aperture**2/(self.wavelength*self.focal_0))
        return dict(minimum_lens_samples=required,
                    lens_samples=self.illumination_lens_samples,
                    lens_phase_sampling_ok=self.illumination_lens_samples >= required,
                    diffuser_samples_per_correlation_length=(self.diffuser_correlation_length *
                        self.illumination_samples/self.illumination_width),
                    note='Necessary sampling indicators only; illumination convergence is unverified.')

    def sampling_report(self):
        # Max lens phase slope: |d phi/dr| = 2 pi r/(lambda f).
        # Nyquist requires dx <= lambda f/(2r), with r=clear_aperture/2.
        required = math.ceil(self.clear_aperture**2 /
                             (self.wavelength * min(self.focal_1, self.focal_2,
                                                    self.focal_3, self.focal_4)))
        n, k = self.lens_samples, self.samples
        return dict(lens_samples=self.lens_samples, minimum_lens_samples=required,
                    lens_phase_sampling_ok=n >= required,
                    lcd_samples_per_cell=self.cell_width / self.dx,
                    largest_dense_operator_complex128_GiB=max(n*n, n*k)*16/2**30,
                    note='Necessary phase-sampling check only; convergence still required.')


    def save(self, filename="config.pth"):
        """Save configuration as a plain dictionary, compatible with weights_only."""
        from dataclasses import asdict
        import torch
        torch.save(asdict(self), filename)

    @staticmethod
    def load(filename="config.pth"):
        import torch
        parameters = torch.load(filename, map_location="cpu", weights_only=True)
        # Permit old configuration files that did not use an effective pupil.
        if parameters.pop("effective_na", None) is not None:
            raise ValueError("effective_na is no longer supported; use physical aperture parameters")
        return Config(**parameters)
