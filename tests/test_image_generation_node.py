import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.builtin.image_generation import draw_image
from core.initial import NODE_FUNCTIONS
from core.state import StateField
from service.flow_run_manager import flow_run_manager


def _state(prompt="画一张极简数据架构图"):
    return {
        "messages": [],
        "fields": {
            "draw_image_1/prompt": StateField(field_name="draw_image_1/prompt", field_value=prompt),
            "draw_image_1/answer": StateField(field_name="draw_image_1/answer", field_value=None),
        },
    }


def _config(api_key="sk-test"):
    return {
        "metadata": {"langgraph_node": "draw_image_1"},
        "configurable": {
            "draw_image_1/prompts": "请生成图片：{prompt}",
            "draw_image_1/model_name": "gpt-image-1",
            "draw_image_1/openai_api_key": api_key,
            "draw_image_1/openai_api_base": "https://api.openai.com/v1",
            "draw_image_1/size": "1024x1024",
            "draw_image_1/quality": "auto",
            "draw_image_1/output_format": "png",
            "draw_image_1/response_format": "b64_json",
        },
    }


class ImageGenerationNodeTest(unittest.TestCase):
    def test_missing_key_returns_node_error_without_raise(self):
        state = draw_image(_state(), _config(api_key=""))

        self.assertIn("缺少 openai_api_key", state["fields"]["draw_image_1/answer"].field_value)

    def test_generates_data_url_from_b64_response(self):
        fake_client = SimpleNamespace(
            images=SimpleNamespace(
                generate=lambda **kwargs: SimpleNamespace(data=[SimpleNamespace(b64_json="abc123")])
            )
        )

        with patch("core.builtin.image_generation.OpenAI", return_value=fake_client) as mocked:
            state = draw_image(_state(), _config())

        self.assertEqual(state["fields"]["draw_image_1/answer"].field_value, "data:image/png;base64,abc123")
        mocked.assert_called_once()

    def test_draw_image_node_runs_inside_agent_flow(self):
        fake_client = SimpleNamespace(
            images=SimpleNamespace(
                generate=lambda **kwargs: SimpleNamespace(data=[SimpleNamespace(b64_json="flow-image")])
            )
        )
        graph_data = {
            "nodes": [
                {
                    "id": "start",
                    "type": "genericNode",
                    "position": {"x": 80, "y": 180},
                    "data": {
                        "name": "start",
                        "display_name": "start",
                        "description": "开始节点",
                        "input": [{
                            "name": "output",
                            "display_name": "画图需求",
                            "field_type": "str",
                            "display_type": "text",
                            "required": True,
                            "show": True,
                            "value": "",
                            "reference": False,
                            "editable": True,
                            "description": "",
                        }],
                        "output": None,
                        "params": None,
                    },
                },
                {
                    "id": "draw_image_1",
                    "type": "genericNode",
                    "position": {"x": 360, "y": 180},
                    "data": {
                        "name": "draw_image",
                        "display_name": "draw_image_1",
                        "description": "画图节点",
                        "input": [{
                            "name": "prompt",
                            "display_name": "prompt",
                            "field_type": "str",
                            "display_type": "text",
                            "required": True,
                            "show": True,
                            "value": "start/output",
                            "reference": True,
                            "editable": True,
                            "description": "",
                        }],
                        "output": [{
                            "name": "answer",
                            "display_name": "image",
                            "field_type": "str",
                            "display_type": "text",
                            "required": True,
                            "show": True,
                            "value": "",
                            "reference": False,
                            "editable": False,
                            "description": "",
                        }],
                        "params": [
                            {"name": "prompts", "display_name": "prompts", "field_type": "str", "display_type": "textarea", "required": True, "show": True, "value": "画图：{prompt}", "editable": True, "description": ""},
                            {"name": "model_name", "display_name": "model", "field_type": "str", "display_type": "text", "required": True, "show": True, "value": "gpt-image-1", "editable": True, "description": ""},
                            {"name": "openai_api_key", "display_name": "api_key", "field_type": "str", "display_type": "text", "required": True, "show": True, "value": "sk-test", "editable": True, "description": ""},
                            {"name": "openai_api_base", "display_name": "api_base", "field_type": "str", "display_type": "text", "required": True, "show": True, "value": "https://api.openai.com/v1", "editable": True, "description": ""},
                            {"name": "size", "display_name": "size", "field_type": "str", "display_type": "text", "required": False, "show": True, "value": "1024x1024", "editable": True, "description": ""},
                            {"name": "quality", "display_name": "quality", "field_type": "str", "display_type": "text", "required": False, "show": True, "value": "auto", "editable": True, "description": ""},
                            {"name": "output_format", "display_name": "format", "field_type": "str", "display_type": "text", "required": False, "show": True, "value": "png", "editable": True, "description": ""},
                            {"name": "response_format", "display_name": "response", "field_type": "str", "display_type": "text", "required": False, "show": True, "value": "b64_json", "editable": True, "description": ""},
                        ],
                    },
                },
                {
                    "id": "end",
                    "type": "genericNode",
                    "position": {"x": 660, "y": 180},
                    "data": {
                        "name": "end",
                        "display_name": "end",
                        "description": "结束节点",
                        "input": None,
                        "output": [{
                            "name": "input",
                            "display_name": "图片结果",
                            "field_type": "str",
                            "display_type": "text",
                            "required": True,
                            "show": True,
                            "value": "draw_image_1/answer",
                            "reference": True,
                            "editable": False,
                            "description": "",
                        }],
                        "params": None,
                    },
                },
            ],
            "edges": [
                {"id": "start-draw_image_1", "source": "start", "target": "draw_image_1", "sourceHandle": "output", "targetHandle": "prompt"},
                {"id": "draw_image_1-end", "source": "draw_image_1", "target": "end", "sourceHandle": "answer", "targetHandle": "input"},
            ],
        }

        flow_draw_image = NODE_FUNCTIONS["draw_image"].function
        original_openai = flow_draw_image.__globals__["OpenAI"]
        flow_draw_image.__globals__["OpenAI"] = lambda **kwargs: fake_client
        try:
            output = flow_run_manager.run_process_compat("image-flow-test", graph_data, {"output": "画一张链路图"})
        finally:
            flow_draw_image.__globals__["OpenAI"] = original_openai

        self.assertEqual(output["input"], "data:image/png;base64,flow-image")


if __name__ == "__main__":
    unittest.main()
