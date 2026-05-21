from typing import Any, Dict

from openai import OpenAI

from core.frontend.annotation import node_config
from core.frontend.field import FrontendField, InputField, OutputField
from core.state import AppState, get_field_from_state, update_state_by_relation
from core.builtin.models import normalize_openai_base_url


prompt_input = InputField(
    name="prompt",
    display_name="prompt",
    field_type="str",
    required=True,
    show=True,
    description="图像生成提示词，可引用前置节点输出",
)

image_output = OutputField(
    name="answer",
    display_name="image",
    field_type="str",
    required=True,
    show=True,
    editable=False,
)

prompts = FrontendField(
    name="prompts",
    display_name="prompts",
    field_type="str",
    display_type="textarea",
    value="{prompt}",
    required=True,
    show=True,
    description="提示词模板，支持 {prompt}",
)

model = FrontendField(
    name="model_name",
    display_name="model",
    field_type="str",
    value="gpt-image-1",
    required=True,
    show=True,
    description="图像模型名，例如 gpt-image-1；兼容网关按其文档填写",
)

api_key = FrontendField(
    name="openai_api_key",
    display_name="api_key",
    field_type="str",
    value="",
    required=True,
    show=True,
    description="运行时填写，不要硬编码密钥",
)

api_base = FrontendField(
    name="openai_api_base",
    display_name="api_base",
    field_type="str",
    value="https://api.openai.com/v1",
    required=True,
    show=True,
    description="OpenAI 或兼容服务的 base_url",
)

size = FrontendField(
    name="size",
    display_name="size",
    field_type="str",
    value="1024x1024",
    required=False,
    show=True,
    description="图像尺寸，例如 1024x1024、1536x1024、1024x1536",
)

quality = FrontendField(
    name="quality",
    display_name="quality",
    field_type="str",
    value="auto",
    required=False,
    show=True,
    description="图像质量，例如 auto、low、medium、high、standard、hd",
)

output_format = FrontendField(
    name="output_format",
    display_name="format",
    field_type="str",
    value="png",
    required=False,
    show=True,
    description="输出格式：png、jpeg、webp",
)

response_format = FrontendField(
    name="response_format",
    display_name="response",
    field_type="str",
    value="b64_json",
    required=False,
    show=True,
    description="返回方式：b64_json 或 url",
)


def _format_prompt(template: str, fields: Dict[str, Any]) -> str:
    values = {key: "" if value is None else value for key, value in fields.items()}
    try:
        return str(template or "{prompt}").format(**values).strip()
    except Exception:
        return f"{template}\n\n{values.get('prompt', '')}".strip()


def _set_answer(state: AppState, current_node: str, value: str):
    state["fields"][f"{current_node}/answer"].field_value = value
    update_state_by_relation(state)
    return state


def _response_to_text(image_item: Any, fmt: str) -> str:
    b64_json = getattr(image_item, "b64_json", None)
    if b64_json:
        return f"data:image/{fmt or 'png'};base64,{b64_json}"
    url = getattr(image_item, "url", None)
    if url:
        return url
    return str(image_item)


@node_config(
    name="draw_image",
    description="生成图片 / Image Generation",
    inputs=[prompt_input],
    outputs=[image_output],
    parameters=[prompts, model, api_key, api_base, size, quality, output_format, response_format],
)
def draw_image(state: AppState, config):
    current_node = config["metadata"]["langgraph_node"]
    fields = get_field_from_state(state, current_node)
    configurable = config.get("configurable") or {}

    model_name = str(configurable.get(f"{current_node}/model_name") or "gpt-image-1").strip()
    key = str(configurable.get(f"{current_node}/openai_api_key") or "").strip()
    if not key:
        return _set_answer(state, current_node, "图像生成失败：缺少 openai_api_key")

    prompt = _format_prompt(str(configurable.get(f"{current_node}/prompts") or "{prompt}"), fields)
    if not prompt:
        return _set_answer(state, current_node, "图像生成失败：prompt 为空")

    try:
        base_url = normalize_openai_base_url(configurable.get(f"{current_node}/openai_api_base"))
    except Exception as exc:
        return _set_answer(state, current_node, f"图像生成失败：{exc}")

    fmt = str(configurable.get(f"{current_node}/output_format") or "png").strip()
    params = {
        "model": model_name,
        "prompt": prompt,
        "size": str(configurable.get(f"{current_node}/size") or "1024x1024").strip(),
        "quality": str(configurable.get(f"{current_node}/quality") or "auto").strip(),
        "output_format": fmt,
        "response_format": str(configurable.get(f"{current_node}/response_format") or "b64_json").strip(),
    }
    params = {key_: value for key_, value in params.items() if value}

    try:
        client = OpenAI(api_key=key, base_url=base_url)
        response = client.images.generate(**params)
    except Exception as exc:
        message = str(exc)
        if "response_format" in message and "response_format" in params:
            try:
                params.pop("response_format", None)
                response = OpenAI(api_key=key, base_url=base_url).images.generate(**params)
            except Exception as retry_exc:
                return _set_answer(state, current_node, f"图像生成失败：{retry_exc}")
        else:
            return _set_answer(state, current_node, f"图像生成失败：{exc}")

    data = getattr(response, "data", None) or []
    if not data:
        return _set_answer(state, current_node, "图像生成失败：模型没有返回图片")
    return _set_answer(state, current_node, _response_to_text(data[0], fmt))
