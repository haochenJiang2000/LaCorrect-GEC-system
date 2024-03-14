import os
from collections import defaultdict
from pathlib import Path
import random
import string

FILTER = ["\x7f", " ", "\uf0e0", "\uf0a7", "\u200e", "\x8b", "\uf0b7", "\ue415", "\u2060", "\ue528", "\ue529", "ᩘ", "\ue074", "\x8b", "\u200c", "\ue529", "\ufeff", "\u200b", "\ue817", "\xad", '\u200f', '️', '่', '︎']
VOCAB_DIR = Path(__file__).resolve().parent.parent / "data"
PAD = "@@PADDING@@"
UNK = "@@UNKNOWN@@"
START_TOKEN = "$START"
SEQ_DELIMETERS = {"tokens": " ",
                  "labels": "SEPL|||SEPR",
                  "operations": "SEPL__SEPR",
                  "pos_tags": "SEPL---SEPR"}  # 分隔符，其中，如果一个source token被多次编辑，那么这些编辑label之间用"SEPL__SEPR"相分割


def split_char(line):
    """
    将中文按照字分开，英文按照词分开
    :param line: 输入句子
    :return: 分词后的句子
    """
    english = "abcdefghijklmnopqrstuvwxyz0123456789"
    output = []
    buffer = ""
    chinese_punct = "！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏."
    for s in line:
        if s in english or s in english.upper() or s in string.punctuation or s in chinese_punct:  # 英文或数字或标点不分
            buffer += s
        else:  # 中文或空格分
            if buffer and buffer != " ":
                output.append(buffer)
            buffer = ""
            if s != " ":
                output.append(s)
    if buffer:
        output.append(buffer)
    return output


def get_target_sent_by_edits(source_tokens, edits):
    """
    对源句子token列表应用编辑操作（Span-level），得到目标句子token列表
    :param source_tokens: 源句子token列表
    :param edits: 编辑序列
    :return:目标句子token列表
    """
    target_tokens = source_tokens[:]
    shift_idx = 0
    for edit in edits:
        start, end, label, _ = edit
        target_pos = start + shift_idx
        source_token = target_tokens[target_pos] \
            if len(target_tokens) > target_pos >= 0 else ''
        if label == "":
            if target_tokens:
                del target_tokens[target_pos]
                shift_idx -= 1
        elif start == end:
            word = label.replace("$APPEND_", "")  # 添加操作
            target_tokens[target_pos: target_pos] = [word]
            shift_idx += 1
        elif label.startswith("$TRANSFORM_"):  # 变形操作
            word = apply_reverse_transformation(source_token, label)
            if word is None:
                word = source_token
            target_tokens[target_pos] = word
        elif start == end - 1:  # 替换操作
            word = label.replace("$REPLACE_", "")
            target_tokens[target_pos] = word
        elif label.startswith("$MERGE_"):  # 合并操作
            target_tokens[target_pos + 1: target_pos + 1] = [label]
            shift_idx += 1

    return replace_merge_transforms(target_tokens)  # 将Merge操作应用到目标句子token列表（当前只是用$Merge标签标记了需要合并的地方）


def replace_merge_transforms(tokens):
    """
    对序列应用Merge变形编辑（将当前token与下一个token合并）
    :param tokens: 词序列列表
    :return: Merge完成后的序列列表
    """
    if all(not x.startswith("$MERGE_") for x in tokens):
        return tokens
    target_tokens = tokens[:]
    allowed_range = range(1, len(tokens) - 1)
    for i in range(len(tokens)):
        target_token = tokens[i]
        if target_token.startswith("$MERGE"):
            if target_token.startswith("$MERGE_SWAP") and i in allowed_range:
                target_tokens[i - 1] = tokens[i + 1]
                target_tokens[i + 1] = tokens[i - 1]
    target_line = " ".join(target_tokens)
    target_line = target_line.replace(" $MERGE_HYPHEN ", "-")
    target_line = target_line.replace(" $MERGE_SPACE ", "")
    target_line = target_line.replace(" $MERGE_SWAP ", " ")
    return target_line.split()


def convert_using_case(token, smart_action):
    """
    对当前token进行大小写变换
    :param token: 当前token
    :param smart_action: 编辑操作标签
    :return: 编辑完成后的token
    """
    if not smart_action.startswith("$TRANSFORM_CASE_"):
        return token
    if smart_action.endswith("LOWER"):
        return token.lower()
    elif smart_action.endswith("UPPER"):
        return token.upper()
    elif smart_action.endswith("CAPITAL"):
        return token.capitalize()
    elif smart_action.endswith("CAPITAL_1"):
        return token[0] + token[1:].capitalize()
    elif smart_action.endswith("UPPER_-1"):
        return token[:-1].upper() + token[-1]
    else:
        return token


def convert_using_verb(token, smart_action):
    """
    对当前token进行动词时形式变换（人称、时态等）
    :param token: 当前token
    :param smart_action: 编辑操作标签
    :return: 编辑完成后的token
    """
    key_word = "$TRANSFORM_VERB_"
    if not smart_action.startswith(key_word):
        raise Exception(f"Unknown action type {smart_action}")
    encoding_part = f"{token}_{smart_action[len(key_word):]}"
    decoded_target_word = decode_verb_form(encoding_part)
    return decoded_target_word


def convert_using_split(token, smart_action):
    """
    对当前token进行切分（去掉连字符-）
    :param token: 当前token
    :param smart_action: 编辑操作标签
    :return: 编辑完成后的token
    """
    key_word = "$TRANSFORM_SPLIT"
    if not smart_action.startswith(key_word):
        raise Exception(f"Unknown action type {smart_action}")
    target_words = token.split("-")
    return " ".join(target_words)


# TODO 单复数变换不止有加s，还有加es的情况？
def convert_using_plural(token, smart_action):
    """
    对当前token进行单复数变换
    :param token: 当前token
    :param smart_action: 编辑操作标签
    :return: 编辑完成后的token
    """
    if smart_action.endswith("PLURAL"):
        return token + "s"
    elif smart_action.endswith("SINGULAR"):
        return token[:-1]
    else:
        raise Exception(f"Unknown action type {smart_action}")


def apply_reverse_transformation(source_token, transform):
    """
    对token进行转换操作
    :param source_token:
    :param transform:
    :return:
    """
    if transform.startswith("$TRANSFORM"):
        # deal with equal
        if transform == "$KEEP":  # 冗余？
            return source_token
        # deal with case
        elif transform.startswith("$TRANSFORM_CASE"):
            return convert_using_case(source_token, transform)
        # deal with verb
        elif transform.startswith("$TRANSFORM_VERB"):
            return convert_using_verb(source_token, transform)
        # deal with split
        elif transform.startswith("$TRANSFORM_SPLIT"):
            return convert_using_split(source_token, transform)
        # deal with single/plural
        elif transform.startswith("$TRANSFORM_AGREEMENT"):
            return convert_using_plural(source_token, transform)
        # raise exception if not find correct type
        raise Exception(f"Unknown action type {transform}")
    else:
        return source_token


def read_parallel_lines(fn1, fn2):
    """
    读取平行语料文件
    :param fn1: 源句子文件（纠错前）
    :param fn2: 目标句子文件（纠错后）
    :return: 分别包含源句子和目标句子的两个列表
    """
    lines1 = read_lines(fn1, skip_strip=True)
    lines2 = read_lines(fn2, skip_strip=True)
    assert len(lines1) == len(lines2)
    out_lines1, out_lines2 = [], []
    for line1, line2 in zip(lines1, lines2):
        if not line1.strip() or not line2.strip():
            continue
        else:
            out_lines1.append(line1)
            out_lines2.append(line2)
    return out_lines1, out_lines2


def read_lines(fn, skip_strip=False):
    """
    从文件中读取每一行
    :param fn: 文件路径
    :param skip_strip: 是否跳过空行
    :return: 包含文件中每一行的列表
    """
    if not os.path.exists(fn):
        return []
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [s.strip() for s in lines if s.strip() or skip_strip]


def write_lines(fn, lines, mode='w'):
    """
    将数据写入到文件中
    :param fn: 输出文件路径
    :param lines: 需要写入的数据
    :param mode: 写入的模式（w、a等）
    :return:
    """
    if mode == 'w' and os.path.exists(fn):
        os.remove(fn)
    with open(fn, encoding='utf-8', mode=mode) as f:
        f.writelines(['%s\n' % s for s in lines])


def decode_verb_form(original):
    return DECODE_VERB_DICT.get(original)


def encode_verb_form(original_word, corrected_word):
    decoding_request = original_word + "_" + corrected_word
    decoding_response = ENCODE_VERB_DICT.get(decoding_request, "").strip()
    if original_word and decoding_response:
        answer = decoding_response
    else:
        answer = None
    return answer


def get_weights_name(transformer_name, lowercase):
    """
    获取预训练MLM名称（可以直接从Transformer的仓库里获取)
    :param transformer_name: MLM名称
    :param lowercase: MLM的选项，训练时是否区分英文大小写
    :return: 预训练语言模型的真实名称
    """
    if transformer_name == 'bert' and lowercase:
        return 'bert-base-uncased'
    if transformer_name == 'bert' and not lowercase:
        return 'bert-base-cased'
    if transformer_name == 'distilbert':
        if not lowercase:
            print('Warning! This model was trained only on uncased sentences.')
        return 'distilbert-base-uncased'
    if transformer_name == 'albert':
        if not lowercase:
            print('Warning! This model was trained only on uncased sentences.')
        return 'albert-base-v1'
    if lowercase:
        print('Warning! This model was trained only on cased sentences.')
    if transformer_name == 'roberta':
        return 'roberta-base'
    if transformer_name == 'gpt2':
        return 'gpt2'
    if transformer_name == 'transformerxl':
        return 'transfo-xl-wt103'
    if transformer_name == 'xlnet':
        return 'xlnet-base-cased'
    if transformer_name == "bert-base-chinese":
        return "bert-base-chinese"


def get_micro_data(filename1, filename2, num):
    """
    获取小数据集
    """
    with open(filename1, 'r', encoding="utf-8") as f:
        with open(filename2, 'w', encoding='utf-8') as out:
            lines = f.readlines()
            sample_lines = random.sample(lines, num)
            out.writelines(sample_lines)


def get_dev_set(total_filename, train_filename, dev_filename, num):
    """
    从训练集随机采样获取开发集
    """
    with open(total_filename, 'r', encoding='utf-8') as f:
        total = f.readlines()
        random.shuffle(total)
        dev = total[:num]
        train = total[num:]
    with open(train_filename, 'w', encoding='utf-8') as f:
        f.writelines(train)
    with open(dev_filename, 'w', encoding='utf-8') as f:
        f.writelines(dev)

def shuffle_file(total_filename):
    with open(total_filename, 'r', encoding='utf-8') as f:
        total = f.readlines()
        random.shuffle(total)
    with open(total_filename, 'w', encoding='utf-8') as f:
        f.writelines(total)

def adjust_labels(label_file_name):
    """
    调整label词典文件，将所有标签按照类别排列，并且获取每个类别的范围
    :param label_file_name:
    :return:
    """
    with open(label_file_name, 'r', encoding='utf-8') as f:
        labels = f.readlines()
        dic = defaultdict(list)
        for label in labels:
            dic[label.lstrip("$").split('_')[0]].append(label)
        start = 0
        for key, value in dic.items():
            end = start + len(value)
            print("{:s}类标签共有{:d}个，范围是：{:d}->{:d}".format(key, len(value), start, end))
            start = end
    with open(label_file_name, 'w', encoding='utf-8') as f:
        for value in dic.values():
            f.writelines(value)


def concat_file(file_list, output_file):
    with open(output_file, 'w', encoding='utf-8') as out:
        for file in file_list:
            with open(file, 'r', encoding='utf-8') as f:
                li = f.readlines()
                print('file:{:s} has {:d} sentences'.format(file, len(li)))
                out.writelines(li)


def convert_preprocessed_data(filename, seg=True):
    with open(filename, 'r', encoding='utf-8') as f:
        with open(filename+'.origin', 'w', encoding='utf-8') as o:
            li = f.readlines()
            for line in li:
                tokens = [pair.rsplit(SEQ_DELIMETERS['labels'], 1)[0]
                          for pair in line.split(SEQ_DELIMETERS['tokens'])]
                if seg:
                    new_line = " ".join(tokens[1:])
                else:
                    new_line = "".join(tokens[1:])
                o.write(new_line+'\n')

def calculate_preplexity(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        li = f.readlines()
        model = kenlm.Model('/home/zhenghuali/zhangyue/GECToR_Chinese/data/zh_giga.no_cna_cmn.prune01244.klm')
        total = 0.0
        for line in li:
            line = line.rstrip('\n')
            p = model.perplexity(line)
            total += p
    return total

def decode_unicode_file(filename):
    res = []
    with open(filename, 'rb') as f:
        for line in f.readlines():
            res.append(line.decode('unicode_escape').replace("u'", "'"))
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(res)

def calculate_vocab_coverage(labels_vocab_file, labels_in_test_file):
    """
    计算当前编辑空间内的编辑标签对测试集的覆盖率
    :param labels_vocab_file:
    :param labels_in_test_file:
    :return:
    """
    with open(labels_vocab_file, 'r', encoding='utf-8') as f1:
        with open(labels_in_test_file, 'r', encoding='utf-8') as f2:
            labels_vocab_set = set([label.split()[0] for label in f1.readlines()])
            labels_test_set = set([label.split()[0] for label in f2.readlines()])
            print("Edit space has {:d} labels.".format(len(labels_vocab_set)))
            print("Test set has {:d} labels.".format(len(labels_test_set)))
            inter = labels_vocab_set.intersection(labels_test_set)
            print("{:d} labels in both space.".format(len(inter)))
            print("Coverage: {:f}".format(len(inter) / len(labels_test_set)))


def select_edit_labels(labels_vocab_count_file, min_count):
    out_file = labels_vocab_count_file + '_min_count_{:d}'.format(min_count)
    with open(labels_vocab_count_file, 'r', encoding='utf-8') as f:
        with open(out_file, 'w', encoding='utf-8') as o:
            for line in f.readlines():
                label, count = line.rstrip('\n').split(" ")
                if int(count) > min_count:
                    o.write(label + '\n')
            o.writelines(["@@UNKNOWN@@\n", "@@PADDING@@\n"])
    calculate_vocab_coverage(out_file,
                             "/data2/jcli/GECToR_Chinese/data/labels_count_NLPCC2018_test_set.txt")

def get_parallel_dev_set(file1, file2, num):
    with open(file1, 'r', encoding='utf-8') as f1:
        with open(file2, 'r', encoding='utf-8') as f2:
            src = f1.readlines()
            trg = f2.readlines()
            para = list(zip(src, trg))
            random.shuffle(para)
    count = 0
    with open(file1 +'_train', 'w', encoding='utf-8') as f1:
        with open(file2 + '_train', 'w', encoding='utf-8') as f2:
            with open(file1 +'_dev', 'w', encoding='utf-8') as f3:
                with open(file2 + '_dev', 'w', encoding='utf-8') as f4:
                    for src, trg in para:
                        if count < num:
                            if src != trg:
                                f3.write(src)
                                f4.write(trg)
                                count += 1
                            else:
                                f1.write(src)
                                f2.write(trg)
                        else:
                            f1.write(src)
                            f2.write(trg)

def extract_parallel_from_XML(filename, output_dir):
    import zhconv
    with open(filename, 'r', )  as f:
        pre_line = ""
        sources = []
        targets = []
        for line in f:
            if "</TEXT>" in line:
                sources.append(zhconv.convert(pre_line, 'zh-cn'))
            elif "</CORRECTION>" in line:
                targets.append(zhconv.convert(pre_line, 'zh-cn'))
            pre_line = line
        print("{:s} has {:d} sentence pairs.".format(filename, len(sources)))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        with open(os.path.join(output_dir, "source"), "a", encoding="utf-8") as source:
            for line in sources:
                source.write(line)
        with open(os.path.join(output_dir, "target"), "a", encoding="utf-8") as target:
            for line in targets:
                target.write(line)

if __name__ == "__main__":
    # decode_unicode_file("/data2/jcli/results.txt")
    # get_micro_data("/data2/jcli/GECToR_Chinese/data/train_NLPCC2018/micro_train_char", "/data2/jcli/GECToR_Chinese/data/train_NLPCC2018/micro_micro_train_char", 10000)
    # get_dev_set("/home/zhenghuali/zhangyue/GECToR_Chinese/data/train_CGED_2016~2020/CGED_2016~2020_train_total",
    #             "/home/zhenghuali/zhangyue/GECToR_Chinese/data/train_CGED_2016~2020/CGED_2016~2020_train_char",
    #             "/home/zhenghuali/zhangyue/GECToR_Chinese/data/dev_CGED_2016~2020/CGED_2016~2020_dev_char", 500)
    # adjust_labels("/data2/jcli/GECToR_Chinese/data/output_vocabulary_chinese_char2/labels.txt")
    # concat_file(['/home/zhenghuali/zhangyue/GECToR_Chinese/NLPCC2018_GEC/Data/training/train_char.trg.self_refine_38.86', '/home/zhenghuali/zhangyue/GECToR_Chinese/zh-hsk/hsk_char.trg.self_refine_38.86'], '/home/zhenghuali/zhangyue/GECToR_Chinese/self_refinement_parallel_data_38.86/hsk_lang8.trg')
    # get_dev_set("/data2/jcli/GECToR_Chinese/data/train_hsk+lang8_refine/train_char_total",
    #             "/data2/jcli/GECToR_Chinese/data/train_hsk+lang8_refine/train_char",
    #             "/data2/jcli/GECToR_Chinese/data/dev_hsk+lang8_refine/dev_char", 5000)
    # convert_preprocessed_data('/data2/jcli/GECToR_Chinese/data/dev_hsk/dev_char')
    # calculate_preplexity('/data2/jcli/GECToR_Chinese/results/seg_word_baseline_total_accuracy.out')
    # extract_parallel_from_XML("/home/zhenghuali/zhangyue/GECToR_Chinese/CGED_train/train_CGED2020.xml")
    # shuffle_file("/home/zhenghuali/zhangyue/GECToR_Chinese/data/train_hsk/train_char_total")
    decode_unicode_file("/home/zhenghuali/zhangyue/GECToR_Chinese/36.76.txt")
    # get_parallel_dev_set("/home/zhenghuali/zhangyue/GECToR_Chinese/self_refinement_parallel_data_38.86/hsk_lang8.src", "/home/zhenghuali/zhangyue/GECToR_Chinese/self_refinement_parallel_data_38.86/hsk_lang8.trg", 5000)