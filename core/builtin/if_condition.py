from typing import Literal
from core.state import AppState
from utils.logger import logger

def get_current_if_condition(config):
    source_name = config['metadata']['langgraph_node']
    current_condition = config['configurable']['_edges'][source_name]
    return config['configurable'][current_condition]

def if_condition(appstate:AppState, config) -> Literal:
    print("if_condition=-=-=-=")

    conditions = get_current_if_condition(config)
    fallback_target = None

    for condition in conditions:

        param = condition['param']
        if param.name == 'else':
            fallback_target = condition['target']
            continue
        variable = appstate['fields'][param.value['reference']].field_value
        if param.value['compare_reference']:
            compare = appstate['fields'][param.value['compare_value']].field_value
        else:
            compare = param.value['compare_value']
        compare_eval = param.value['compare']
        if compare_eval == 'equal':
            if variable == compare:
                logger.info(f"{param.value['reference']} : {variable} == {compare}")
                return condition['target']
        elif compare_eval == 'not equal':
            if variable != compare:
                logger.info(f"{param.value['reference']} : {variable} != {compare}")
                return condition['target']
        elif compare_eval == 'longer than':
            if variable > compare:
                logger.info(f"{param.value['reference']} : {variable} > {compare}")
                return condition['target']
        elif compare_eval == 'shorter than':
            if variable < compare:
                logger.info(f"{param.value['reference']} : {variable} < {compare}")
                return condition['target']
        elif compare_eval == 'longer than or equal':
            if variable >= compare:
                logger.info(f"{param.value['reference']} : {variable} >= {compare}")
                return condition['target']
        elif compare_eval == 'shorter than or equal':
            if variable <= compare:
                logger.info(f"{param.value['reference']} : {variable} <= {compare}")
                return condition['target']
        elif compare_eval == 'contains':
            if compare in variable:
                logger.info(f"{param.value['reference']} : {variable} in {compare}")
                return condition['target']
        elif compare_eval == 'not contains':
            if compare not in variable:
                logger.info(f"{param.value['reference']} : {variable} not in {compare}")
                return condition['target']
        elif compare_eval == 'is empty':
            if variable == '' or variable == None:
                logger.info(f"{param.value['reference']} : {variable} is empty")
                return condition['target']
        elif compare_eval == 'is not empty':
            if variable != '' and variable is not None:
                logger.info(f"{param.value['reference']} : {variable} is not empty")
                return condition['target']

    if fallback_target:
        return fallback_target
    raise ValueError("No condition matched and no else branch was configured")
