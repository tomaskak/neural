# from neural.util.process_orchestrator import *

# from multiprocessing.shared_memory import SharedMemory
# import numpy as np
# import time
# import pytest

# import sys


# def do_nothing(*args, **kwargs):
#     pass


# def write_item_fn(idx, start, length, key):
#     data = WorkerSpace.data
#     print(f"data={data}, key={key}")
#     sys.stdout.flush()
#     end = start + length
#     msg = data["input"][start:end]
#     data["msg"][start:end] = msg
#     if "first_letter" in data:
#         data["msg"][start] = data["first_letter"]
#     data["time"][idx] = time.time_ns() % 1000000000000
#     return key


# def init_msgtime(msg_length, num_times):
#     defs = [
#         ("input", "c", (msg_length,)),
#         ("msg", "c", (msg_length,)),
#         ("time", np.float32, (num_times,)),
#     ]
#     return defs


# def make_write_work_items(X, edges, init_args=None):
#     work_defs = []
#     for i, edge in enumerate(edges):
#         provides, depends_on = edge
#         work_defs.append(
#             Work(
#                 provides=provides,
#                 depends_on=depends_on,
#                 fn=write_item_fn,
#                 args=(i, i * 3, 3),
#             )
#         )
#     init = Init(
#         num_workers=2,
#         space_needed=len(X) + len(work_defs) * np.dtype(np.float32).itemsize * 2,
#         init_defs=init_msgtime(len(X), len(work_defs)),
#         args=init_args,
#     )
#     return work_defs, init


# def write_input(X):
#     WorkerSpace.data["input"][:] = X[:]


# class TestProcessOrchestrator:
#     @pytest.mark.parametrize(
#         "work_defs",
#         [
#             [
#                 Work(depends_on="B", provides="A", fn=do_nothing),
#                 Work(depends_on="A", provides="B", fn=do_nothing),
#             ],
#             [
#                 Work(depends_on="B", provides="A", fn=do_nothing),
#                 Work(depends_on="C", provides="B", fn=do_nothing),
#                 Work(depends_on="A", provides="C", fn=do_nothing),
#             ],
#             [Work(depends_on="A", provides="A", fn=do_nothing)],
#             [Work(depends_on="A", provides="B", fn=do_nothing)],
#             [
#                 Work(depends_on="B", provides="A", fn=do_nothing),
#                 Work(depends_on="C", provides="B", fn=do_nothing),
#                 Work(depends_on="A", provides="C", fn=do_nothing),
#                 Work(depends_on="D", provides="C", fn=do_nothing),
#             ],
#             [
#                 Work(depends_on="A", provides="B", fn=do_nothing),
#                 Work(depends_on=None, provides="B", fn=do_nothing),
#             ],
#         ],
#     )
#     def test_rejects_bad_input(self, work_defs):
#         with pytest.raises(Exception):
#             orchestrator = ProcessOrchestrator(work_defs)

#     def test_single_chain_of_items(self):
#         X = "hownowcow"
#         work_defs, init = make_write_work_items(
#             X, [("A", "B"), ("B", "C"), ("C", None)]
#         )

#         orchestrator = ProcessOrchestrator(init, work_defs, lambda x: x)

#         write_input(X)
#         success, output = orchestrator.execute()

#         assert success
#         assert (
#             "".join([c.decode("utf-8") for c in output["msg"]]) == "hownowcow"
#         ), f"{output['msg']}"
#         assert output["time"][1] < output["time"][0]
#         assert output["time"][2] < output["time"][1]

#     def test_single_item(self):
#         X = "how"
#         work_defs, init = make_write_work_items(X, [("C", None)])
#         orchestrator = ProcessOrchestrator(init, work_defs, lambda x: x)

#         write_input(X)
#         success, output = orchestrator.execute()

#         assert success
#         assert (
#             "".join([c.decode("utf-8") for c in output["msg"]]) == "how"
#         ), f"{output['msg']}"
#         assert output["time"][0] > 0

#     def test_disjoint_items(self):
#         X = "hownowcowwow"
#         work_defs, init = make_write_work_items(
#             X, [("A", None), ("B", "A"), ("C", None), ("D", "C")]
#         )
#         orchestrator = ProcessOrchestrator(init, work_defs, lambda x: x)

#         write_input(X)
#         success, output = orchestrator.execute()

#         assert success
#         assert (
#             "".join([c.decode("utf-8") for c in output["msg"]]) == "hownowcowwow"
#         ), f"{output['msg']}"
#         assert output["time"][0] < output["time"][1]
#         assert output["time"][2] < output["time"][3]

#     def test_dual_dependent_items(self):
#         X = "hownowcow"
#         work_defs, init = make_write_work_items(
#             X, [("A", None), ("B", None), ("C", "A,B")]
#         )
#         orchestrator = ProcessOrchestrator(init, work_defs, lambda x: x)

#         write_input(X)
#         success, output = orchestrator.execute()

#         assert success
#         assert (
#             "".join([c.decode("utf-8") for c in output["msg"]]) == "hownowcow"
#         ), f"{output['msg']}"
#         assert output["time"][0] < output["time"][2]
#         assert output["time"][1] < output["time"][2]

#     def test_dual_dependents(self):
#         X = "hownowcow"
#         work_defs, init = make_write_work_items(
#             X, [("A", None), ("B", "A"), ("C", "A")]
#         )
#         orchestrator = ProcessOrchestrator(init, work_defs, lambda x: x)

#         write_input(X)
#         success, output = orchestrator.execute()

#         assert success
#         assert (
#             "".join([c.decode("utf-8") for c in output["msg"]]) == "hownowcow"
#         ), f"{output['msg']}"
#         assert output["time"][0] < output["time"][2]
#         assert output["time"][0] < output["time"][1]

#     def test_dual_dependencies_dual_dependents(self):
#         X = "hownowcowwow"
#         work_defs, init = make_write_work_items(
#             X, [("A", None), ("B", None), ("C", "A,B"), ("D", "A")]
#         )
#         orchestrator = ProcessOrchestrator(init, work_defs, lambda x: x)

#         write_input(X)
#         success, output = orchestrator.execute()

#         assert success
#         assert (
#             "".join([c.decode("utf-8") for c in output["msg"]]) == "hownowcowwow"
#         ), f"{output['msg']}"
#         assert output["time"][0] < output["time"][2]
#         assert output["time"][1] < output["time"][2]
#         assert output["time"][0] < output["time"][3]

#     def test_init_args(self):
#         X = "hownowcow"
#         work_defs, init = make_write_work_items(
#             X, [("A", None), ("B", "A"), ("C", "B")], init_args=[("first_letter", "x")]
#         )
#         orchestrator = ProcessOrchestrator(init, work_defs, lambda x: x)

#         write_input(X)
#         success, output = orchestrator.execute()

#         assert success
#         assert (
#             "".join([c.decode("utf-8") for c in output["msg"]]) == "xowxowxow"
#         ), f"{output['msg']}"
