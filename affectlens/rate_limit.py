"""Small process-local sliding-window limiter for the single-worker demo."""

import math
import time
from collections import deque


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: int = 60):
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests: dict[str, deque[float]] = {}

    def check(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        timestamps = self.requests.setdefault(key, deque())
        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()
        if len(timestamps) >= self.limit:
            retry_after = max(1, math.ceil(self.window_seconds - (now - timestamps[0])))
            return False, retry_after
        timestamps.append(now)
        if len(self.requests) > 1024:
            self.requests = {item: values for item, values in self.requests.items() if values and values[-1] > cutoff}
        return True, 0
