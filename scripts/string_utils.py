import re

def replace_br_with_newlines(text):
    br_pattern = re.compile(r'<br\s*/?>', re.IGNORECASE)
    return br_pattern.sub('\n', text)

def replace_newlines_with_br(text_content):
    lines = text_content.splitlines()
    return '<br>'.join(lines)
