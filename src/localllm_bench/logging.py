"""Project logging helpers."""

import logging
from pathlib import Path


def get_logger(source: str) -> logging.Logger:
    """Return a consistently named project logger.

    Parameters
    ----------
    source
        Source file or module name.

    Returns
    -------
    logging.Logger
        Logger scoped to the source module.
    """
    return logging.getLogger(f"localllm_bench.{Path(source).stem}")
