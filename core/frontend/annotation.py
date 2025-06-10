

# 自定义装饰器以添加配置
def node_config(name: str, inputs: list, outputs: list, parameters: list, description: str = "",display_name:str =""):
    if display_name == "":
        display_name = name
    def decorator(func):

        func._node_config = {
            "name": name,
            "display_name": display_name,
            "description": description,
            "input": inputs,
            "output": outputs,
            "parameters": parameters
        }
        return func
    return decorator