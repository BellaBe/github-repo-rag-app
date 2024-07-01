from enum import Enum
from langchain_core.pydantic_v1 import BaseModel, Field

class AccuracyEnum(str, Enum):
    accurate = "accurate"
    inaccurate = "inaccurate"
        
class LLMEvalResult(BaseModel):
    accuracy: AccuracyEnum = Field(description="Label indicating if the answer is accurate or inaccurate.")
    feedback: str = Field(description="Explanation of why the specific label was assigned. Must be concise and not more than 2 sentences.")
 