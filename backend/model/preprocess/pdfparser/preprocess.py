import re
# m model.preprocess.DocxParser import docxparser
from pprint import pprint
import sqlite3
import docx
import json
from pdfparser import Converter


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


def split(data, min_len, flag):
    '''
    对抽取出的文本内容进行分句和初步筛选
    :param data: 段落形式的文本内容，列表
    :return:
    split_data
    '''
    split_data = []
    for seq in data:
        # 统一符号格式，去除特殊符号
        seq = seq.replace("?", "？")
        seq = seq.replace("!", "！")
        seq = seq.replace(",", "，")
        seq = seq.replace(";", "；")
        seq = seq.replace("\"", "”")

        # 去除句中空格
        seq = seq.replace(" ", "")
        seq = seq.replace("　", "")
        # 仅去除中文间的空格
        # temp = re.search("([\u4e00-\u9fa5]{1})[ ]{1,}([\u4e00-\u9fa5]{1})", line)
        # while temp:
        #     a = temp.group(1)
        #     b = temp.group(2)
        #     line = re.sub("([\u4e00-\u9fa5]{1})[ 	]{1,}([\u4e00-\u9fa5]{1})", a + b, line, count=1)
        #     temp = re.search("([\u4e00-\u9fa5]{1})[ 	]{1,}([\u4e00-\u9fa5]{1})", line)

        seq = seq.replace("­", "")
        seq = seq.strip()

        # 分句
        seq_list1 = []
        seq_list2 = []
        temp = seq.split('。')
        # print(111, temp[-1])
        for i, l in enumerate(temp[:-1]):
            temp[i] = l + '。'
        seq_list1 += temp[:-1]
        if len(temp[-1]) > 5:
            seq_list1.append(temp[-1])

        for seq in seq_list1:
            temp = seq.split('！')
            for i, l in enumerate(temp[:-1]):
                temp[i] = l + '！'
            seq_list2 += temp[:-1]
            if len(temp[-1]) > 5:
                seq_list2.append(temp[-1])

        seq_list1 = []
        for seq in seq_list2:
            temp = seq.split('？')
            for i, l in enumerate(temp[:-1]):
                temp[i] = l + '？'
            seq_list1 += temp[:-1]
            if len(temp[-1]) > 5:
                seq_list1.append(temp[-1])

        seq_list2 = []
        for seq in seq_list1:
            temp = seq.split('；')
            for i, l in enumerate(temp[:-1]):
                temp[i] = l + '；'
            seq_list2 += temp[:-1]
            if len(temp[-1]) > 5:
                seq_list2.append(temp[-1])
        if flag:
            seq_list2.pop(0)    # pdf去除页眉导致的噪音文本
        for i in seq_list2:
            # todo:希望去掉 (1) (2) 这种序号，但直接使用lstrip会导致去掉了句首数字，如 1998年...
            # i = i.lstrip("()（）123456789 ")
            if re.match(r'[(（]\d+[)）]', i):
                i = i[re.match(r'[(（]\d+[)）]', i).span()[1]:]

            # 排除过短的句子
            if len(i) > min_len:
                split_data.append(i)
    return split_data


def data_reader(file_id):
    data = []
    # conn = sqlite3.connect('./database/mydata.db')
    # cur = conn.cursor()
    file_path = './6.pdf'
    if file_path[-3:] == 'pdf':
        converter = Converter(file_path)
        textBlockList = converter.convert()
        for textBlock in textBlockList:
            pid, bbox, content = textBlock  # (页码，文本框，文本内容)
            for i in range(pid - len(data) + 1):
                data.append("")
            data[pid] += content
        converter.close()
    else:
        data = docxparser(file_path)
    return data


def preprocess(file_id, json_path, min_len=10):
    '''
    根据fileid找到数据库中的文件，提取文本内容并预处理
    docx和pdf的文本提取方式不同：
        docx：python-docx库
        pdf：pdfparser
    预处理：
        1.对识别出的文本内容进行分句，通过句尾符号划分，如： 。？！（）“
            此外引号需要额外写规则处理，如
            句号后跟引号则引号也需要划入当前句子。
                他说：”今天天气真好。“
            具体情况在数据处理时边观察边优化
        2.删去不完整句子，英文占比过半的句子，过短的句子
        3.将最后保留的有效句子保存进数据库，或以json文件形式保存
            包括句子所属file_id,file_name,page
    :param file_id:
    :return:
    '''
    # 文本识别
    data = data_reader(file_id)
    # 分句
    split_data = split(data, 10)
    pprint(split_data)
    # 预处理
    processed_data = []
    for line in split_data:
        # 句长太短，或是英文超过一半的句子，将会被筛去
        if len(line) <= min_len:
            continue
        if len(re.findall("[a-zA-z]{1,}", line)) > (len(re.findall("[\u4e00-\u9fa5]{1}", line)) / 2):
            continue

        # 排除非完整句子
        if line.strip()[-1] not in ["。", "！", "？", "；", "“", "?", "!", ";"]:
            continue

        # 排除公式
        # 若句尾以 (1) 等序号的形式结尾，则认为句中有公式，筛去
        if re.search('[(]{1}[0-9]*[)]{1}$', line.strip()):
            continue

        # ===这里不是很懂===
        # 若句中出现() {} =，或多个符号连续（引号不算）的情况，判断为高级符号未识别，筛去。
        if re.search('[({\[]{1,}[,.;:，。；：！、…]{0,}[)}\]]{1,}', line):
            continue
        # if re.search('[=]+', line):
        #     print(line)
        #     continue
        if re.search('[,.;:，。；：！、]{2,}', line):
            continue
        # 若句中出现单个字成段的情况，筛去（todo
        if re.search('^[\u4e00-\u9fa5]{1}[,.;:，。；：！…=]{1}', line):
            continue
        if re.search('[,.;:，。；：！…=]{1}[\u4e00-\u9fa5]{1}[,.;:，。；：！…=]{1}', line):
            continue
        # 排查句首是否还有不合法情况
        line = line.lstrip(".、 ")
        line = line.replace(" ", "")
        line = line.replace("　", "")

        # 识别特殊字符，若句子包含特殊字符则筛去
        sp = False
        ch_punc = '，；。：！？ '
        for char in list(line[:-1]):
            if u'\u4e00' <= char <= u'\u9fff' or 32 <= ord(char) <= 126 or char in ch_punc:
                continue
            else:
                sp = True
        if sp:
            continue
        processed_data.append(line)

    # todo：将processed_data转为json格式文件保存到json_path
    # conn = sqlite3.connect('./database/mydata.db')
    # cur = conn.cursor()
    # for line in processed_data:
    #     # id = cur.execute("SELECT max(id) FROM articles").fetchall()[0][0]
    #     # if id is not None:
    #     #     id += 1
    #     # else:
    #     #     id = 0
    #     cur.execute("insert into articles values (?, ?, ?, ?)",
    #                 (None, file_id, 0, line))
    #     conn.commit()
    # f = open(json_path, 'w')
    # f.write(json.dumps(processed_data))
    # f.close()
    print("=====================================================================================================")
    pprint(processed_data)
    return processed_data

preprocess(1, "")