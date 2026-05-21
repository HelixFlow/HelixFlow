import unittest

from router.flow_manage import _mask_sensitive_payload, _serialize_flow_create_data
from utils.json_util import json_deserialization


class FlowManageTest(unittest.TestCase):
    def test_create_flow_data_is_serialized_for_database(self):
        graph_data = {
            "nodes": [{"id": "start", "data": {"name": "start"}}],
            "edges": [],
        }

        db_data, response_data = _serialize_flow_create_data({
            "name": "bizAgentTest",
            "description": "template",
            "status": 1,
            "data": graph_data,
        })

        self.assertIsInstance(db_data["data"], str)
        self.assertEqual(json_deserialization(db_data["data"]), graph_data)
        self.assertEqual(response_data, graph_data)

    def test_create_flow_error_log_payload_masks_keys(self):
        payload = {
            "data": {
                "nodes": [{
                    "data": {
                        "params": [
                            {"name": "openai_api_key", "value": "sk-real-secret"},
                            {"name": "openai_api_base", "value": "https://api.example.com/v1"},
                        ]
                    }
                }]
            },
            "token": "abc",
        }

        masked = _mask_sensitive_payload(payload)

        self.assertEqual(masked["token"], "********")
        self.assertEqual(masked["data"]["nodes"][0]["data"]["params"][0]["value"], "********")
        self.assertEqual(masked["data"]["nodes"][0]["data"]["params"][1]["value"], "https://api.example.com/v1")


if __name__ == "__main__":
    unittest.main()
