from typing import TypedDict, List, Annotated, Union, Any, Dict
import pandas as pd
import operator

def update_logs(old_logs: List[str], new_logs: Union[str, List[str]]) -> List[str]:
    """custom reducer to append logs"""
    if isinstance(new_logs, str):
        return old_logs + [new_logs]
    return old_logs + new_logs

class AgentState(TypedDict):
    # original path and context
    file_path: str
    user_context: str
    
    # current data state
    current_df: pd.DataFrame
    df_history: Annotated[List[pd.DataFrame], operator.add]
    
    # agent outputs
    analysis_report: str
    cleaning_plan: str
    audit_report: Dict[str, Any] # results of the data audit
    
    # flow control
    is_clean: bool
    logs: Annotated[List[str], update_logs]
    messages: Annotated[List, operator.add]
