# PS -> Raw customer ticket contains a lot of anomalies, Clean the customer ticket using the re library of python

import re
def sanitize_ticket_text(text:str) -> str:
    """
    Cleans raw customer ticket text:
    1. Removes excess newlines and redundant spaces.
    2. Strips common email disclaimer footers.
    3. Normalizes smart quotes to standard ASCII quotes.
    """

    # Replace smart quotes to normal quotes
    text = text.replace("“", '"').replace("”", '"').replace("`", "'").replace("’", "'")

    #strip email signature boilerplate
    text = re.sub(r'(--\s*\n.*$)|(Sent from my iPhone.*$)|(Best regards.*$)','',text, flags = re.DOTALL | re.IGNORECASE)

    # Collapse multiple whitespaces/newlines to single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Testing practically
dirty_sample = """
Hey Support! "My app crashed" with error code ERR_502.
I was charged $20 twice on order #88412!
Please fix this NOW!

--
Sent from my iPhone
"""

clean_sample = sanitize_ticket_text(dirty_sample)
print("Sanitized Ticket Text : ")
print(clean_sample)