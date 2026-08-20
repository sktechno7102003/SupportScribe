import json
from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal

class SupportTicketSchema(BaseModel):
    urgency_level : Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(
        description="Operational urgency based on busness impact"
    )
    primary_category : Literal["BILLING", "TECHNICAL_BUG", "ACCOUNT_ACCESS", "FEATURE_REQUEST"] = Field(
        description="Departmental routing category"
    )
    sentiment : Literal["POSITIVE", "NEUTRAL", "NEGATIVE","FRUSTATED"] = Field(
        description="Detected user emotional state"
    )
    extracted_entities : List[str] = Field(
        default_factory=list, 
        description="Specific error codes, order IDs, or dates found in the text"
    )
    recommended_action : str = Field(
        description="one-Sentence action item for the support representative"
    )

def validate_target(payload:dict):
    target_json_string = ''
    flag = True
    try:
        validated_obj = SupportTicketSchema(**payload)
        target_json_string = validated_obj.model_dump_json()
    except ValidationError as e:
        print("Data not validated using pydantic JSON schema")
        flag = False
    return target_json_string, flag

