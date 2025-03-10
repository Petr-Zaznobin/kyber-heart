from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import requests
from tabulate import tabulate


@dataclass
class Report:
    version: str
    start: datetime
    end: datetime
    dataset: str
    task_type: Optional[str]
    name: str
    description: str
    authors: list[str]
    url: str

    metrics: pd.DataFrame
    matrices: dict[str, npt.NDArray[np.int64]]

    def __post_init__(self) -> None:
        assert self.authors

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "dataset": self.dataset,
            "task_type": self.task_type,
            "name": self.name,
            "description": self.description,
            "authors": self.authors,
            "metrics": self.metrics.to_dict(),
            "matrices": {name: matrix.tolist() for name, matrix in self.matrices.items()},
            "url": self.url
        }

    def dump(self, path: Path | str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @staticmethod
    def load(path: Path | str) -> Report:
        with open(path, "r") as f:
            obj: dict[str, Any] = json.load(f)
        return Report(
            version=obj["version"],
            start=datetime.fromisoformat(obj["start"]),
            end=datetime.fromisoformat(obj["end"]),
            dataset=obj["dataset"],
            task_type=obj["task_type"],
            name=obj["name"],
            description=obj["description"],
            authors=obj["authors"],
            metrics=pd.DataFrame(obj["metrics"]),
            matrices={name: np.array(matrix) for name, matrix in obj["matrices"].items()},
            url=obj.get("url", "")
        )

    @staticmethod
    def _date_to_string(d: datetime) -> str:
        d_str = d.strftime("%c")
        if (tz := d.tzname()) is not None:
            d_str += f" {tz}"
        return d_str

    @staticmethod
    def _matrix_to_string(m: npt.NDArray[np.int64]) -> str:
        return pd.DataFrame(m).to_string(header=False, index=False)

    def _matrices_to_str(self) -> str:
        class_to_matrix = {c: [Report._matrix_to_string(m)] for c, m in self.matrices.items()}
        df = pd.DataFrame(class_to_matrix)
        num_columns = 5

        res = ""
        for begin in range(0, len(df.columns), num_columns):
            columns = df.columns[begin:begin + num_columns]
            matrices_str = tabulate(df[columns], tablefmt="grid", headers="keys", showindex=False)
            if begin:
                res += "\n"
            res += f"{matrices_str}\n"
        return res

    def print_matrices(self) -> None:
        print(self._matrices_to_str())

    def __str__(self) -> str:
        b: Callable[[str], str] = lambda x: "\033[1m" + x + "\033[0m"

        res = f"{b('Версия')}: {self.version}\n"
        res += f"{b('Начало')}: {Report._date_to_string(self.start)}\n"
        res += f"{b('Конец')}: {Report._date_to_string(self.end)}\n"

        res += f"{b('Название')}: {self.name}\n"
        res += f"{b('Описание')}: {self.description}\n"
        res += f"{b('Ссылка')}: {self.url}\n"
        if len(self.authors) == 1:
            res += f"{b('Автор')}: {self.authors[0]}\n"
        else:
            authors = ", ".join(self.authors)
            res += f"{b('Авторы')}: {authors}\n"

        res += f"\n{b('Метрики')}:\n"
        res += self.metrics.to_string(line_width=80, float_format=lambda x: str(round(x, 6)))

        res += f"\n\n{b('Матрицы рассогласования')}:\n\n"
        res += self._matrices_to_str()
        return res


def push_report(report: Report, url: str, token: str) -> str:
    response = requests.post(url=url, json=report.to_dict(), headers={"ApiKey": token})

    response.raise_for_status()

    return str(response.json()["result_url"])
