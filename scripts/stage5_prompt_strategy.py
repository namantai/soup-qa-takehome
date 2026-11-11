"""Public synthetic transformation used to audit config application, not Soup code."""
from copy import deepcopy

MARKER='QA_REQUIRED_PREFIX'


def add_marker(row):
    result=deepcopy(row)
    result['messages'][0]['content']=MARKER+'\n'+result['messages'][0]['content']
    return result
