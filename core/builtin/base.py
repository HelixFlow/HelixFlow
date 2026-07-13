from core.state import AppState, update_state_by_relation
def start_node(appstate :AppState):
    update_state_by_relation(appstate)
    return appstate

def end_node(appstate:AppState):
    update_state_by_relation(appstate)
    return appstate
