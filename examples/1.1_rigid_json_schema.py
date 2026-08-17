# PS -> Create a JSON schema for customer query category, urgency level , recommended action and verify it

import json
from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal

class TicketTriageSchema(BaseModel):
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

# Pratical test: validating raw dictionary before turning it into training target
raw_target_data = {
    "urgency_level" : "CRITICAL",
    "primary_category" : "BILLING",
    "sentiment" : "FRUSTATED",
    "extracted_entities" : ["INV-98324", "$450.00"],
    "recommended_action" : "Issue immediate refund review and freeze recurring charge."
}


# Validate and convert into serialized json string
validated_obj = TicketTriageSchema(**raw_target_data)
target_json_string = validated_obj.model_dump_json()

print("Serialized Validated Target :")
print(target_json_string)

# Output
# Serialized Validated Target :
# {"urgency_level":"CRITICAL",
#  "primary_category":"BILLING",
#  "sentiment":"FRUSTATED",
#  "extracted_entities":["INV-98324","$450.00"],
#  "recommended_action":"Issue immediate refund review and freeze recurring charge."}