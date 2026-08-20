import re


def clean_text(text:str) -> str:
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