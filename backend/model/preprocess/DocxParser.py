import docx
from pprint import pprint

def docxparser(filepath):
    doc = docx.Document(filepath)
    data = []
    for para in doc.paragraphs:
        para = para.text
        para.strip()
        if para.strip() == '':
            continue
        data.append(para)
    return data