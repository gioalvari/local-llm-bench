"""Process and host resource telemetry."""

import threading
import time

import psutil


def resource_sample(process: psutil.Process, started_ns: int) -> dict[str, int]:
    """Capture process-tree and host memory counters."""
    process_rss = 0
    process_tree_rss = 0
    try:
        process_rss = process.memory_info().rss
        process_tree_rss = process_rss + sum(
            child.memory_info().rss for child in process.children(recursive=True)
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    host = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "monotonic_offset_ns": time.monotonic_ns() - started_ns,
        "process_rss_bytes": process_rss,
        "process_tree_rss_bytes": process_tree_rss,
        "host_available_bytes": host.available,
        "swap_used_bytes": swap.used,
    }


class ResourceMonitor:
    """Sample one process tree in a background thread."""

    def __init__(self, process: psutil.Process, interval_seconds: float) -> None:
        """Initialize a monitor without starting it."""
        self.process = process
        self.interval_seconds = interval_seconds
        self.started_ns = time.monotonic_ns()
        self.samples: list[dict[str, int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._collect, daemon=True)

    def _collect(self) -> None:
        while not self._stop.is_set():
            self.samples.append(resource_sample(self.process, self.started_ns))
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        """Start collecting resource samples."""
        self._thread.start()

    def stop(self) -> list[dict[str, int]]:
        """Stop collection and return all captured samples."""
        self._stop.set()
        self._thread.join()
        return self.samples
