"""Runtime and host capability inspection."""

import hashlib
import importlib.util
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import psutil
from pydantic import BaseModel


class ExecutableInfo(BaseModel):
    """Availability and fingerprint of an external executable."""

    available: bool
    path: str | None = None
    sha256: str | None = None
    version: str | None = None


class CapabilityReport(BaseModel):
    """Host facts needed to interpret benchmark results."""

    os: str
    os_release: str
    architecture: str
    logical_cpus: int
    physical_cpus: int | None
    memory_bytes: int
    available_memory_bytes: int
    disk_free_bytes: int
    unified_memory: bool
    llama_bench: ExecutableInfo
    llama_server: ExecutableInfo
    mlx_installed: bool
    torch_installed: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_line(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = result.stdout.strip() or result.stderr.strip()
    return output.splitlines()[0] if output else None


def _executable(name: str, version_args: list[str] | None = None) -> ExecutableInfo:
    location = shutil.which(name)
    if location is None:
        return ExecutableInfo(available=False)
    resolved = Path(location).resolve()
    return ExecutableInfo(
        available=True,
        path=str(resolved),
        sha256=_sha256(resolved),
        version=(
            _first_line([str(resolved), *version_args])
            if version_args is not None
            else None
        ),
    )


def inspect_capabilities(root: Path | None = None) -> CapabilityReport:
    """Inspect the local host without modifying it.

    Parameters
    ----------
    root
        Filesystem location used for free-space reporting.

    Returns
    -------
    CapabilityReport
        Serializable capability and provenance information.
    """
    target = root or Path.cwd()
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(target)
    system = platform.system()
    logical_cpus = os.cpu_count() or 1
    return CapabilityReport(
        os=system,
        os_release=platform.release(),
        architecture=platform.machine(),
        logical_cpus=logical_cpus,
        physical_cpus=psutil.cpu_count(logical=False),
        memory_bytes=memory.total,
        available_memory_bytes=memory.available,
        disk_free_bytes=disk.free,
        unified_memory=system == "Darwin" and platform.machine() == "arm64",
        # llama-bench embeds its build commit in result rows but has no version flag.
        llama_bench=_executable("llama-bench"),
        llama_server=_executable("llama-server", ["--version"]),
        mlx_installed=importlib.util.find_spec("mlx") is not None,
        torch_installed=importlib.util.find_spec("torch") is not None,
    )


def report_as_dict(report: CapabilityReport) -> dict[str, Any]:
    """Convert a capability report to JSON-compatible values."""
    return report.model_dump(mode="json")
