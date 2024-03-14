# -*- coding: utf-8 -*-


from pdfparser import Converter
from pprint import pprint
import fitz

if __name__ == '__main__':
    converter = Converter('./5.pdf')
    textBlockList = converter.convert()
    text = []
    page_content = ""
    f = open("./5.txt", "w", encoding='UTF-8')
    for textBlock in textBlockList:
        pid, bbox, content = textBlock # (页码，文本框，文本内容)
        for i in range(pid - len(text) + 1):
            text.append("")
        text[pid] += content
    for i in text:
        f.write(i)
        print(i)
    f.close()
    converter.close()
