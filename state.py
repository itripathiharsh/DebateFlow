from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class DebateState(BaseModel):
    """
    Shared state for the LangGraph debate workflow.
    Compatible with Pydantic v2 and LangGraph 0.2+.
    """
    topic: str = ""
    current_round: int = 0
    history: List[Dict[str, Any]] = []
    memory_summary: str = ""
    winner: Optional[str] = None
    final_summary: Optional[str] = None
 
    class Config:
        extra = "forbid"  