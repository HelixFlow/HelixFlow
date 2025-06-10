import json
import base64
def json_serialization(json_obj):
    try:
        return base64.b64encode(json.dumps(json_obj).encode('utf-8')).decode('utf-8')
    except Exception as e:
        return None

def json_deserialization(json_str):
    try:
        return json.loads(base64.b64decode(json_str).decode('utf-8'))
    except Exception as e:
        return None
def str_serialization(str_obj):
    try:
        return base64.b64encode(str_obj.encode('utf-8')).decode('utf-8')
    except Exception as e:
        return None
def str_deserialization(str_str):
    try:
        return base64.b64decode(str_str).decode('utf-8')
    except Exception as e:
        return None