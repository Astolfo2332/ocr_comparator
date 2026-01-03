import re

def get_alpha_numeric_string(s: str) -> str:
    data = re.sub(r'[^a-zA-Z0-9]', '', s)
    return data
