import random

def generate_mock_tickets(num_tickets: int = 250) -> list:
    """Generates synthetic noisy tickets. Intentionally injects 5% bad data to test validation."""
    categories = ["BILLING", "TECHNICAL_BUG", "ACCOUNT_ACCESS", "FEATURE_REQUEST"]
    urgencies = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    sentiments = ["POSITIVE", "NEUTRAL", "NEGATIVE", "FRUSTRATED"]
    
    bad_urgency_injections = ["URGENT", "ASAP", "IDK"] # These should fail Pydantic validation
    
    mock_texts = [
        "My screen goes black when I click checkout! Fix it! --\nSent from my Android",
        "Can I get a refund for invoice INV-112? Thanks.",
        "Need dark mode. My eyes hurt.",
        "Forgot my password and the reset link is expired. Help. \n\n Best regards, \n John"
    ]
    
    data = []
    for i in range(num_tickets):
        # 5% chance to inject bad target data
        if random.random() < 0.05:
            urgency = random.choice(bad_urgency_injections)
        else:
            urgency = random.choice(urgencies)
            
        data.append({
            "ticket_id": f"TCK-{1000+i}",
            "raw_text": random.choice(mock_texts) + f" (Ref: {random.randint(100, 999)})",
            "urgency_level": urgency,
            "primary_category": random.choice(categories),
            "sentiment": random.choice(sentiments),
            "extracted_entities": [f"ID-{random.randint(10,99)}"],
            "recommended_action": "Review immediately."
        })
    return data