import time
import threading
import json


class Timer:
    def __init__(self, name: str, out_fn):
        self._name = name
        self._out_fn = out_fn

        self._time_start = 0

    def __enter__(self):
        self._time_start = time.time()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        if exception_type is None:
            elapsed = time.time() - self._time_start
            self._out_fn(self._name, elapsed)


thread_local = threading.local()


class PrintManager:
    def __init__(self, reset_on_count=1000):
        self._timers = {}
        self._push_count = 0
        self._reset_on_count = reset_on_count

    def push(self, timer_name, elapsed):
        if timer_name not in self._timers:
            self._timers[timer_name] = {"count": 0, "total": 0}
        t = self._timers[timer_name]
        t["count"] += 1
        t["total"] += elapsed

        self._push_count += 1

        if self._push_count % self._reset_on_count == 0:
            msg = {}
            for name, values in self._timers.items():
                msg[name] = {
                    "count": values["count"],
                    "average(ms)": values["total"] / values["count"] * 1000,
                }
            print(json.dumps(msg, indent=4))


def timer_to_log(name: str) -> Timer:
    def out_fn(name: str, elapsed: float):
        print(f"{name} finished in {elapsed*1000}ms")

    return Timer(name, out_fn)


def timer(name: str) -> Timer:
    return Timer(name, getattr(thread_local, "timer_manager", None).push)


def init_timer_manager(mgr):
    thread_local.timer_manager = mgr
