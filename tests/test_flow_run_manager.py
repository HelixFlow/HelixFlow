import threading
import time
import unittest
from types import SimpleNamespace

import core.frontend.graph as graph_module
from core.state import update_state_by_relation
from service.flow_run_manager import COMPLETED, PAUSED, FlowRunManager


def _field(name, value=None, reference=False):
    return {
        "name": name,
        "display_name": name,
        "value": value,
        "reference": reference,
        "field_type": "str",
    }


def _param(name, value):
    return {
        "name": name,
        "display_name": name,
        "value": value,
        "field_type": "str",
    }


def _graph_payload():
    return {
        "nodes": [
            {
                "id": "start",
                "data": {
                    "name": "start",
                    "display_name": "start",
                    "description": "",
                    "input": [_field("output")],
                    "output": None,
                    "params": None,
                },
            },
            {
                "id": "slow_1",
                "data": {
                    "name": "test_slow",
                    "display_name": "slow_1",
                    "description": "",
                    "input": [_field("question", "start/output", True)],
                    "output": [_field("answer")],
                    "params": [
                        _param("openai_api_key", "sk-real"),
                        _param("prompts", "say {question}"),
                    ],
                },
            },
            {
                "id": "end",
                "data": {
                    "name": "end",
                    "display_name": "end",
                    "description": "",
                    "input": None,
                    "output": [_field("input", "slow_1/answer", True)],
                    "params": None,
                },
            },
        ],
        "edges": [
            {
                "id": "start-slow_1",
                "source": "start",
                "target": "slow_1",
                "sourceHandle": None,
                "targetHandle": None,
            },
            {
                "id": "slow_1-end",
                "source": "slow_1",
                "target": "end",
                "sourceHandle": None,
                "targetHandle": None,
            },
        ],
    }


class FlowRunManagerPauseResumeTest(unittest.TestCase):
    def setUp(self):
        self._old_node_functions = dict(graph_module.NODE_FUNCTIONS)
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = []

        def slow_node(state):
            update_state_by_relation(state)
            question = state["fields"]["slow_1/question"].field_value
            self.calls.append(question)
            self.started.set()
            self.release.wait(2)
            state["fields"]["slow_1/answer"].field_value = f"echo:{question}"
            return state

        graph_module.NODE_FUNCTIONS["test_slow"] = SimpleNamespace(function=slow_node)
        self.manager = FlowRunManager()

    def tearDown(self):
        self.release.set()
        for run in list(self.manager._runs.values()):
            worker = run.worker
            if worker and worker.is_alive():
                worker.join(timeout=2)
        graph_module.NODE_FUNCTIONS.clear()
        graph_module.NODE_FUNCTIONS.update(self._old_node_functions)

    def _start_run_until_slow_node(self):
        run = self.manager.create_run("flow-id", _graph_payload(), {"output": "你好"})
        self.assertTrue(self.started.wait(3), "slow node did not start")
        return run["run_id"]

    def _wait_until_worker_idle(self, run_id):
        for _ in range(40):
            run = self.manager.get_run(run_id)
            if not run["worker_alive"]:
                return run
            time.sleep(0.05)
        self.fail("worker did not become idle")

    def _wait_until_status(self, run_id, status):
        for _ in range(40):
            run = self.manager.get_run(run_id)
            if run["status"] == status:
                return run
            time.sleep(0.05)
        self.fail(f"run did not reach status {status}")

    def _pause_and_wait_for_discard(self, run_id):
        paused = self.manager.pause_run(run_id)
        self.assertEqual(paused["status"], PAUSED)
        self.assertEqual(paused["next_nodes"], ["slow_1"])
        self.release.set()
        paused = self._wait_until_worker_idle(run_id)
        self.assertEqual(paused["status"], PAUSED)
        self.assertEqual(paused["next_nodes"], ["slow_1"])
        return paused

    def test_pause_then_resume_without_patch_completes_from_same_node(self):
        run_id = self._start_run_until_slow_node()
        self._pause_and_wait_for_discard(run_id)

        self.started.clear()
        resumed = self.manager.resume_run(run_id, {})
        self.assertEqual(resumed["status"], "running")
        self.assertEqual(resumed["next_nodes"], ["slow_1"])
        self.assertTrue(self.started.wait(3), "slow node did not restart")

        completed = self._wait_until_status(run_id, COMPLETED)
        self.assertIsNone(completed["error"])
        self.assertEqual(self.calls, ["你好", "你好"])
        self.assertFalse(
            any("did not advance" in (event.get("data", {}).get("error") or "") for event in completed["events"])
        )

    def test_polling_after_resume_does_not_rewind_checkpoint(self):
        run_id = self._start_run_until_slow_node()
        self._pause_and_wait_for_discard(run_id)

        self.release.clear()
        self.started.clear()
        self.manager.resume_run(run_id, {})
        self.assertTrue(self.started.wait(3), "slow node did not restart")

        for _ in range(5):
            polled = self.manager.get_run(run_id)
            self.assertEqual(polled["status"], "running")
            self.assertEqual(polled["next_nodes"], ["slow_1"])
            self.assertIsNone(polled["error"])
            self.assertTrue(polled["worker_alive"])

        self.release.set()
        completed = self._wait_until_status(run_id, COMPLETED)
        self.assertIsNone(completed["error"])
        self.assertEqual(completed["result"], {"input": "echo:你好"})

    def test_second_resume_click_is_rejected_without_corrupting_run(self):
        run_id = self._start_run_until_slow_node()
        self._pause_and_wait_for_discard(run_id)

        self.release.clear()
        self.started.clear()
        self.manager.resume_run(run_id, {})
        self.assertTrue(self.started.wait(3), "slow node did not restart")

        with self.assertRaisesRegex(ValueError, "Only paused runs can be resumed"):
            self.manager.resume_run(run_id, {})

        self.release.set()
        completed = self._wait_until_status(run_id, COMPLETED)
        self.assertIsNone(completed["error"])

    def test_resume_is_rejected_while_inflight_worker_is_still_returning(self):
        run_id = self._start_run_until_slow_node()
        self.manager.pause_run(run_id)

        with self.assertRaisesRegex(ValueError, "后台调用还在返回中"):
            self.manager.resume_run(run_id, {})

        self.release.set()
        self._wait_until_worker_idle(run_id)

    def test_editable_input_patch_is_applied_to_future_node(self):
        run_id = self._start_run_until_slow_node()
        self._pause_and_wait_for_discard(run_id)

        self.started.clear()
        self.manager.resume_run(run_id, {"inputs": {"output": "新的问题"}})
        self.assertTrue(self.started.wait(3), "slow node did not restart")
        completed = self._wait_until_status(run_id, COMPLETED)

        self.assertIsNone(completed["error"])
        self.assertEqual(self.calls, ["你好", "新的问题"])

    def test_locked_patch_is_rejected_before_resume(self):
        run_id = self._start_run_until_slow_node()
        self._pause_and_wait_for_discard(run_id)

        with self.assertRaisesRegex(ValueError, "当前暂停点不会生效"):
            self.manager.resume_run(run_id, {"fields": {"unknown/value": "bad"}})

    def test_masked_secret_patch_does_not_overwrite_real_secret(self):
        run_id = self._start_run_until_slow_node()
        paused = self._pause_and_wait_for_discard(run_id)
        self.assertEqual(paused["configurable"]["slow_1/openai_api_key"], "********")

        self.started.clear()
        self.manager.resume_run(run_id, {"configurable": {"slow_1/openai_api_key": "********"}})
        self.assertTrue(self.started.wait(3), "slow node did not restart")

        run = self.manager._get_run(run_id)
        self.assertEqual(run.config["configurable"]["slow_1/openai_api_key"], "sk-real")
        completed = self._wait_until_status(run_id, COMPLETED)
        self.assertIsNone(completed["error"])


if __name__ == "__main__":
    unittest.main()
