# src/itsm_analysis/graphs/main_state.py
from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict
import pandas as pd


# Define the state schema for your graph
class AgentGraphState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')


    features: Optional[pd.DataFrame] = None
    dashboard_output: Optional[Dict] = None
    # Add other state variables as needed