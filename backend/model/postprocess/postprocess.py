import os
from model.postprocess.ChERRANT.modules.annotator import Annotator
from model.postprocess.ChERRANT.modules.tokenizer import Tokenizer
import argparse
from collections import Counter
from tqdm import tqdm
import torch
from collections import defaultdict
from multiprocessing import Pool
from opencc import OpenCC
from model.postprocess.segment.segment_bert import segment
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# annotator, sentence_to_tokenized = None, None
cc = OpenCC("t2s")


import re


class M2Processor():
    def __init__(self, src_sent, edit_lines):
        self.src_sent = src_sent
        self.edit_lines = edit_lines
        self.edits = {}
        self.trg_sents = []

    def conv_edit(self, line):
        line = line.strip().split("|||")
        edit_span = line[0].split(" ")
        edit_span = (int(edit_span[0]), int(edit_span[1]))
        edit_res = line[2]
        editor = line[-1]
        if edit_span[0] == -1:
            return None
        if edit_span[0] == edit_span[1]:
            edit_tag = "ADD"
        elif edit_res == "-NONE-" or edit_res == "":
            edit_tag = "DEL"
        else:
            edit_tag = "REP"
        return editor, edit_tag, edit_span, edit_res

    def get_edits(self):
        for line in self.edit_lines:
            if line:
                edit_item = self.conv_edit(line)
                if not edit_item:
                    continue
                editor, edit_tag, edit_span, edit_res = edit_item
                if editor not in self.edits:
                    self.edits[editor] = []
                self.edits[editor].append({"span": edit_span, "op": edit_tag, "res": edit_res})

    def get_para(self):
        self.get_edits()
        if self.edits:
            for editor in self.edits:
                sent = self.src_sent.split(" ")
                for edit_item in self.edits[editor]:
                    edit_span, edit_tag, trg_tokens = edit_item["span"], edit_item["op"], edit_item["res"]
                    if edit_tag == "DEL":
                        sent[edit_span[0]:edit_span[1]] = [" " for _ in range(edit_span[1] - edit_span[0])]
                    else:
                        if edit_tag == "ADD":
                            if edit_span[0] != 0:
                                sent[edit_span[0] - 1] += " " + trg_tokens
                            else:
                                sent[edit_span[0]] = trg_tokens + " " + sent[edit_span[0]]
                        elif edit_tag == "REP":
                            src_tokens_len = len(sent[edit_span[0]:edit_span[1]])
                            sent[edit_span[0]:edit_span[1]] = [trg_tokens] + [" " for _ in range(src_tokens_len - 1)]
                sent = " ".join(sent).strip()
                res_sent = re.sub(" +", " ", sent)
                self.trg_sents.append(res_sent)
            return self.trg_sents
        else:
            return [self.src_sent]


def read_m2(m2_data):
    src_sent = None
    edit_lines = []
    fr = m2_data.split("\n")
    result = []
    for line in fr:
        line = line.strip()
        if line.startswith("S "):
            src_sent = line.replace("S ", "", 1)
        elif line.startswith("A "):
            edit_lines.append(line.replace("A ", "", 1))
        elif line == "":
            if edit_lines:
                result.append([src_sent, edit_lines[:]])
                edit_lines.clear()
    return result


def clean_edit(edit_lines, src_sent):
    """
    筛掉英文相关的编辑，<unk>则保持原内容不变
    :param edit_lines:
    :return:
    """
    clean_edit_lines = []
    src = src_sent.split(" ")
    for edit in edit_lines:
        tgt_content = edit.split("|||")[2]
        src_span = edit.split("|||")[0].split(" ")
        src_content = "".join(src[int(src_span[0]):int(src_span[1])])
        content = src_content + tgt_content

        if re.findall("[a-zA-Z]{1,}", src_content) or '#' in src_content:
            if (not re.findall("[a-zA-Z]{1,}", src_content)) and tgt_content == "-NONE-":
                clean_edit_lines.append(edit)
            continue
        # 关于引用标记 [1] 等不做处理
        if re.findall("[\[\]]{1,}", src_content):
            continue
        # 不加冒号
        if re.findall("[:：]{1,}", content) and error_type == "M":
            continue
        if src_content in ";；" and tgt_content == "。":
            continue
        clean_edit_lines.append(edit)
    return clean_edit_lines


def annotate(line, logger, annotator, sentence_to_tokenized, segmented=False, no_simplified=False):
    """
    :param no_simplified:
    :param segmented:
    :param line:
    :return:
    """
    sent_list = line.split("\t")[1:]
    source = sent_list[0]
    if segmented:
        source = source.strip()
    else:
        source = "".join(source.strip().split())
        # source = segment(source.strip())
    output_str = ""
    for idx, target in enumerate(sent_list[1:]):
        try:
            if segmented:
                target = target.strip()
            else:
                target = "".join(target.strip().split())
                # target = segment(target.strip())
            if not no_simplified:
                target = cc.convert(target)
            source_tokenized, target_tokenized = sentence_to_tokenized[source], sentence_to_tokenized[target]
            out, cors = annotator(source_tokenized, target_tokenized, idx)
            # logger.info(out)
            if idx == 0:
                output_str += "".join(out[:-1])
            else:
                output_str += "".join(out[1:-1])
        except Exception as e:
            logger.info("解析异常：")
            logger.info(e)
            raise Exception
    return output_str

def para2m2(srcs, tgts, batch_size, logger, Device=0, segmented=False, no_simplified=False):
    device = Device

    tokenizer = Tokenizer("char", device, segmented)
    # global annotator, sentence_to_tokenized
    annotator = Annotator.create_default("char", "first")
    sentence_to_tokenized = {}
    lines = []
    for idx, [src, tgt] in enumerate(zip(srcs, tgts)):
        lines.append(str(idx) + '\t' + src + '\t' + tgt)

    count = 0
    sentence_set = set()
    sentence_to_tokenized = {}
    for line in lines:
        sent_list = line.split("\t")[1:]
        for idx, sent in enumerate(sent_list):
            if segmented:
                sent = sent.strip()
            else:
                sent = "".join(sent.split()).strip()
                # sent = segment(sent.strip())
            if idx >= 1:
                if not no_simplified:
                    sentence_set.add(cc.convert(sent))
                else:
                    sentence_set.add(sent)
            else:
                sentence_set.add(sent)
    batch = []
    for sent in sentence_set:
        count += 1
        if sent:
            batch.append(sent)
        if count % batch_size == 0:
            results = tokenizer(batch)
            for s, r in zip(batch, results):
                sentence_to_tokenized[s] = r  # Get tokenization map.
            batch = []
    if batch:
        results = tokenizer(batch)
        for s, r in zip(batch, results):
            sentence_to_tokenized[s] = r  # Get tokenization map.

    # 单进程模式
    result = ""
    for line in lines:
        # logger.info("解析数据:"+line)
        ret = annotate(line, logger, annotator, sentence_to_tokenized, segmented, no_simplified)
        result += ret + "\n"
    # print(result)
    return result

    # 多进程模式：仅在Linux环境下测试，建议在linux服务器上使用
    # with Pool(args.worker_num) as pool:
    #     for ret in pool.imap(annotate, tqdm(lines), chunksize=8):
    #         if ret:
    #             f.write(ret)
    #             f.write("\n")

def m22para(result_m2_data, logger):
    srcs, tgts = [], []
    for src_sent, edit_lines in result_m2_data:
        edit_lines = clean_edit(edit_lines, src_sent)
        m2_item = M2Processor(src_sent, edit_lines)
        trg_sents = m2_item.get_para()
        srcs.append(src_sent)
        tgts.append(trg_sents[0])
    return srcs, tgts

def dedup(edit_lines):
    """
    编辑范围重复则随机选一个采用
    :param edit_lines:
    :return:
    """
    dup_span = []
    dedup_edit_lines = []
    for line in edit_lines:
        dup = False
        start, end = tuple(line.split("|||")[0].split(" "))
        if start == end:
            end = str(int(end) + 1)

        # 判断编辑范围是否重复
        for span in dup_span:
            if int(span[0]) <= int(start) < int(span[1]) \
                    or int(span[0]) <= int(end) - 1 < int(span[1]) \
                    or (int(start) < int(span[0]) and int(end) >= int(span[1])):  # 已有同范围编辑被采用
                dup = True
                break

        if not dup:
            dedup_edit_lines.append(line)
            dup_span.append(tuple(line.split("|||")[0].split(" ")))

    return dedup_edit_lines


def vote(m2_data_models, logger, min=2):
    """
    :param m2_data_models:
    :param min:
    :return:
    edit_vote_list:{
        src_sent:{
            edit: count
        }
    }
    """
    data = read_m2(m2_data_models[0][0])
    edit_vote_list = [{} for i in range(len(data))]
    src_list = [src for src, editlines in data]
    print(data)
    for m2_data, weight, model_type in m2_data_models:
        count = 0
        for idx, (src_sent, edit_lines) in enumerate(read_m2(m2_data)):
            edit_lines = clean_edit(edit_lines, src_sent)
            for edit in edit_lines:
                print(edit)
                if model_type == "seq2edit":

                    if edit.split("|||")[1] == "W":
                        edit_vote_list[idx][edit] = edit_vote_list[idx].setdefault(edit, 0.0) + float(float(weight)/2)
                        continue
                edit_vote_list[idx][edit] = edit_vote_list[idx].setdefault(edit, 0) + 1
            count += 1
        assert count == len(edit_vote_list), logger.info("len(read_m2(m2_data)) == len(edit_vote_list)"+str(count)+str(len(edit_vote_list)))
    result_m2_data = []
    for idx, edit_vote in enumerate(edit_vote_list):
        edit_lines = []
        for edit, count in edit_vote.items():
            if count >= min:
                edit_lines.append(edit)
        edit_lines = dedup(edit_lines)

        src = src_list[idx].replace("##", "")
        result_m2_data.append([src, edit_lines[:]])
        edit_lines.clear()
    return result_m2_data


def postprocess(srcs_models, tgts_models, batch_size, logger, device, min_vote=2, segmented=False):
    m2_data_models = []
    length = 0
    # logger.info("   转化为m2格式...")
    for srcs, tgts_tup in zip(srcs_models, tgts_models):
        tgts, weight, model_type = tgts_tup
        length = len(srcs)
        m2_data = para2m2(srcs, tgts, batch_size, Device=device, segmented=False, logger=logger)
        m2_data_models.append([m2_data, weight, model_type])
    # logger.info("   投票...")
    result_m2_data = vote(m2_data_models, min=min_vote, logger=logger)
    assert length == len(result_m2_data), logger.info("length == len(result_m2_data)"+str(length)+" "+str(len(result_m2_data)))
    # assert length == len(result_m2_data), logger.info(result_m2_data)
    # logger.info(result_m2_data)
    srcs, tgts = m22para(result_m2_data, logger)
    # for src, tgt in zip(srcs, tgts):
    #     print(src)
    #     print(tgt)
    #     print("=======")
    return srcs, tgts


if __name__ == "__main__":
    pass