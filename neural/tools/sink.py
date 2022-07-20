from multiprocessing import Process, Queue

from tools.app_manager import AppManager


def _proc_fn(work_q, done_q):
    app_manager = AppManager("RL-Training")

    for action in iter(work_q.get, "STOP"):
        if type(action) == str:
            if action == "RENDER":
                app_manager.render()
        else:
            action, args = action
            if action == "CREATE":
                namespace, graph_type = args
                app_manager.add(namespace, graph_type)
            elif action == "PUSH":
                namespace, graph_type, data = args
                app_manager.forward_to_handler(namespace, graph_type, data)
    done_q.put("DONE")


class Sink:
    def __init__(self, proc_fn=_proc_fn):
        self._work_q = Queue()
        self._done_q = Queue()

        self._process = Process(target=proc_fn, args=(self._work_q, self._done_q))
        self._process.start()

        self._on = False

    def __del__(self):
        self._work_q.put("STOP")
        self._process.join()

    def push(self, namespace, data_type, data):
        if not self._on:
            return
        self._work_q.put(("PUSH", (namespace, data_type, data)))

    def init_namespace(self, namespace, data_type):
        self._work_q.put(("CREATE", (namespace, data_type)))

    def start(self):
        self._work_q.put("RENDER")

    def enable(self):
        self._on = True

    def disable(self):
        self._on = False


class SinkClient:
    def __init__(self, base_namespace, sink):
        self._base = base_namespace
        self._sink = sink

    def push(self, namespace, data_type, data):
        self._sink.push(self._base + "-" + namespace, data_type, data)

    def init_namespace(self, namespace, data_type):
        self._sink.init_namespace(self._base + "-" + namespace, data_type)
