import math
from pathlib import Path
from typing import Literal, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

__all__ = ["plot_ecg"]

_colors = ["#b140b1", "#538ec8", "#4e7010", "#0b0941", "#871838", "#9c0f93"]
_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

_Lines = Sequence[int | tuple[int, str]]
_PreparedLines = tuple[list[float], list[str]]


class _PlotEcgImpl:
    _MAX_SIGNAL_LENGTH = 60
    _CELLS_PER_SECOND = 25
    _CELLS_PER_MVOLT = 10

    def __init__(
        self,
        ecg: npt.NDArray[np.float32],
        sample_rate: float,
        amplitude: float | tuple[float, float],
        title: str,
        xlabel: str,
        ylabel: str,
        offset: int,
        linewidth: float,
        colors: Sequence[str],
        render_coef: float,
        save_path: Optional[Path],
        show: Optional[bool],
        distance: float,
        names: Optional[Sequence[str] | Literal["standart"]],
        ecg_lines: _Lines,
        ecg_lines_color: str,
        lead_lines: Optional[Sequence[_Lines]],
        lead_lines_color: str
    ) -> None:
        assert render_coef >= 0.1
        assert render_coef <= 10
        self._render_coef = render_coef

        self._ecg = _PlotEcgImpl._squeeze(ecg)
        number_leads, length = self._ecg.shape
        x = np.arange(offset, length + offset, dtype=np.float32) / sample_rate

        self._x_min = float(x[0])
        self._x_max = float(x[-1] + 1 / sample_rate)
        duration = self._x_max - self._x_min
        if duration > _PlotEcgImpl._MAX_SIGNAL_LENGTH:
            raise RuntimeError(f"Signal is too long ({duration} seconds)")

        self._y_min, self._y_max, y_init = _PlotEcgImpl._get_amplitudes(
            amplitude=amplitude,
            number_leads=number_leads,
            distance=distance
        )
        self._init_plot_size()
        self._init_major_grid()
        self._init_minor_grid()
        self._init_text(title, xlabel, ylabel)

        prepared_ecg_lines = _PlotEcgImpl._prepare_lines(ecg_lines, ecg_lines_color, sample_rate)

        if lead_lines is None:
            lead_lines = number_leads * [[]]
        assert len(lead_lines) == number_leads
        prepared_lead_lines = [_PlotEcgImpl._prepare_lines(lines, lead_lines_color, sample_rate)
                               for lines in lead_lines]

        self._plot(x=x, linewidth=linewidth, colors=colors, y_init=y_init,
                   distance=distance, ecg_lines=prepared_ecg_lines, lead_lines=prepared_lead_lines)

        if names is not None:
            self._plot_names(x=x[0] + 0.05, y_init=y_init + 0.1, distance=distance, names=names)

        _PlotEcgImpl._save_and_show(save_path=save_path, show=show)

    @staticmethod
    def _prepare_lines(lines: _Lines, default_color: str, sample_rate: float) -> _PreparedLines:
        positions: list[float] = []
        colors: list[str] = []
        for line in lines:
            positions.append(line if isinstance(line, int) else line[0])
            colors.append(default_color if isinstance(line, int) else line[1])
        positions = [p / sample_rate for p in positions]
        return positions, colors

    def _plot(
        self,
        x: npt.NDArray[np.float32],
        linewidth: float,
        colors: Sequence[str],
        y_init: float,
        distance: float,
        ecg_lines: _PreparedLines,
        lead_lines: list[_PreparedLines]
    ) -> None:
        linewidth *= self._render_coef
        lines_width = 1.5 * linewidth

        lines_pos, lines_colors = ecg_lines
        plt.vlines(lines_pos, ymin=self._y_min, ymax=self._y_max, color=lines_colors, lw=lines_width)

        y = y_init
        for i, signal in enumerate(self._ecg):
            color = colors[i % len(colors)]
            plt.plot(x, signal + y, linewidth=linewidth, color=color)

            lines_pos, lines_colors = lead_lines[i]
            y_min = y - .45 * distance
            y_max = y + .45 * distance
            plt.vlines(lines_pos, ymin=y_min, ymax=y_max, color=lines_colors, lw=lines_width)

            y -= distance

    def _plot_names(
        self,
        x: float,
        y_init: float,
        distance: float,
        names: Sequence[str] | Literal["standart"]
    ) -> None:
        if isinstance(names, str):
            names = _names
        assert len(names) >= len(self._ecg)
        names = names[:len(self._ecg)]

        fontsize = 20 * self._render_coef
        y = y_init
        for name in names:
            plt.text(x, y, name, fontsize=fontsize, backgroundcolor="#ffffffd0")
            y -= distance

    @staticmethod
    def _save_and_show(save_path: Optional[Path], show: Optional[bool]) -> None:
        if save_path is None:
            show = show is None or show
        else:
            plt.savefig(save_path)
            show = show is not None and show
        if not show:
            plt.close()

    def _init_text(self, title: str, xlabel: str, ylabel: str) -> None:
        if title:
            plt.title(title, fontsize=16 * self._render_coef)
        plt.xlabel(xlabel, fontsize=13 * self._render_coef)
        plt.ylabel(ylabel, fontsize=13 * self._render_coef)

    def _init_plot_size(self) -> None:
        num_horizontal_cells = _PlotEcgImpl._CELLS_PER_SECOND * (self._x_max - self._x_min)
        num_vertical_cells = _PlotEcgImpl._CELLS_PER_MVOLT * (self._y_max - self._y_min)

        width = num_horizontal_cells * self._render_coef / 5
        height = num_vertical_cells * self._render_coef / 5
        plt.figure(figsize=(width, height))
        plt.xlim(self._x_min, self._x_max)
        plt.ylim(self._y_min, self._y_max)

    def _init_major_grid(self) -> None:
        _cell_time = 0.2
        _cell_voltage = 0.5

        x_min = math.ceil(self._x_min / _cell_time) * _cell_time
        y_min = math.ceil(self._y_min / _cell_voltage) * _cell_voltage

        font_size = 12 * self._render_coef
        plt.xticks(np.arange(x_min, self._x_max + 1e-8, _cell_time), fontsize=font_size)
        plt.yticks(np.arange(y_min, self._y_max + 1e-8, _cell_voltage), fontsize=font_size)
        plt.grid(which="major", linewidth=0.75 * self._render_coef, color="black", alpha=0.4)

    def _init_minor_grid(self) -> None:
        _cell_time = 0.04
        _cell_voltage = 0.1

        x_min = math.ceil(self._x_min / _cell_time) * _cell_time
        y_min = math.ceil(self._y_min / _cell_voltage) * _cell_voltage

        plt.xticks(np.arange(x_min, self._x_max + 1e-8, _cell_time), minor=True)
        plt.yticks(np.arange(y_min, self._y_max + 1e-8, _cell_voltage), minor=True)
        plt.grid(which="minor", linewidth=0.25 * self._render_coef, color="black", alpha=0.5)

    @staticmethod
    def _get_amplitudes(
        amplitude: float | tuple[float, float],
        number_leads: int,
        distance: float
    ) -> tuple[float, float, float]:
        if not isinstance(amplitude, tuple):
            amplitude = (-amplitude, amplitude)

        height = amplitude[1] - amplitude[0]
        assert height > 0

        if number_leads == 1:
            return (amplitude[0], amplitude[1], 0)

        y_max = height + (number_leads - 1) * distance
        return (0, y_max, y_max - height / 2)

    @staticmethod
    def _squeeze(ecg: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        original_shape = ecg.shape
        ecg = np.squeeze(ecg)
        if ecg.ndim == 1:
            ecg = np.expand_dims(ecg, axis=0)
        if ecg.ndim != 2:
            raise RuntimeError(f"Bad shape: {original_shape}")
        if len(ecg) > 12:
            raise RuntimeError(f"Too many signals: {len(ecg)}")
        return ecg


def plot_ecg(
    ecg: npt.NDArray[np.float32],
    sample_rate: float = 500,
    amplitude: float | tuple[float, float] = 1.5,
    title: str = "",
    xlabel: str = "Секунды",
    ylabel: str = "мВ",
    offset: int = 0,
    linewidth: float = 2,
    colors: str | Sequence[str] = _colors,
    render_coef: float = 1,
    save_path: Optional[Path | str] = None,
    show: Optional[bool] = None,
    distance: float = 1.5,
    names: Optional[Sequence[str] | Literal["standart"] | str] = None,
    ecg_lines: Optional[_Lines] = None,
    ecg_lines_color: str = "#a0000080",
    lead_lines: Optional[Sequence[_Lines]] = None,
    lead_lines_color: str = "#0000a080",
) -> None:
    """
    :param ecg: ЭКГ (может быть от 1 до 12 отведений)
    :param sample_rate: частота дискретизации сигнала
    :param amplitude: отображаемый диапазон по оси y в мВ. Если подано одно число,
        то интерпритируется как пара (-x, x). При отображении нескольких оведений
        влияет на расстояние до первого отведения и на отступ после последнего
    :param title: заголовок
    :param xlabel: подпись оси x
    :param ylabel: подпись оси y
    :param offset: смещение (в точках) для пометок оси x
    :param linewidth: толщина линий сигнала
    :param colors: цвета линий ЭКГ
    :param render_coef: влияет на размер итовогого изображения (большее значение параметра
        приводит к большему качеству)
    :param save_path: сохранить изображение в файл вместо отображения на экране
    :param show: отображать или нет изображение на экране (поведение по умолчанию зависит
        от параметра ``save_path``)
    :param distance: расстояние между отведениями в мВ
    :param names: имена отведений. "standart" выставляет значения по умолчанию
    :param ecg_lines: линии по всей ЭКГ
    :param ecg_lines_color: цвет ЭКГ линий по умолчанию
    :param lead_lines: линии по каждому отведению в отдельности
    :param lead_lines_color: цвет линий на отведениях по умолчанию
    """
    if isinstance(save_path, str):
        save_path = Path(save_path)
    if isinstance(colors, str):
        colors = [colors]
    if isinstance(names, str) and names != "standart":
        names = [names]

    if ecg_lines is None:
        ecg_lines = []

    _PlotEcgImpl(
        ecg=ecg,
        sample_rate=sample_rate,
        amplitude=amplitude,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        offset=offset,
        linewidth=linewidth,
        colors=colors,
        render_coef=render_coef,
        save_path=save_path,
        show=show,
        distance=distance,
        names=names,
        ecg_lines=ecg_lines,
        ecg_lines_color=ecg_lines_color,
        lead_lines=lead_lines,
        lead_lines_color=lead_lines_color
    )
