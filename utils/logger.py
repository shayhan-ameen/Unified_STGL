"""Console + file logging and a small CSV history writer."""

import csv
import logging
import os
import sys
from typing import Any, Dict, Optional


def get_logger(name: str = "unified_stgl", log_file: Optional[str] = None) -> logging.Logger:
    """Return a logger that writes to stdout and, optionally, to ``log_file``."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stdout for h in logger.handlers):
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        logger.addHandler(stream)

    if log_file is not None:
        log_file = os.path.abspath(log_file)
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file for h in logger.handlers):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger


class CSVHistory:
    """Append one row of scalars per call; the header is written on first use."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fieldnames = None

    def append(self, row: Dict[str, Any]) -> None:
        write_header = self._fieldnames is None or not os.path.exists(self.path)
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in self._fieldnames})
