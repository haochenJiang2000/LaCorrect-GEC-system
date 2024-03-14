# -*- coding: utf-8
import os

from model.utils.helpers import split_char
from model.config import *
from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline, DataCollatorForSeq2Seq
import difflib
import re
from opencc import OpenCC
from model.postprocess.postprocess import postprocess
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from model.gector.seq2seq.data_loader import MyDataset

cc = OpenCC("t2s")


def split_sentence(document: str, flag: str = "all", limit: int = 510):
    """
    Args:
        document:
        flag: Type:str, "all" 中英文标点分句，"zh" 中文标点分句，"en" 英文标点分句
        limit: 默认单句最大长度为510个字符
    Returns: Type:list
    """
    sent_list = []
    try:
        if flag == "zh":
            document = re.sub('(?P<quotation_mark>([。？！](?![”’"\'])))', r'\g<quotation_mark>\n', document)  # 单字符断句符
            document = re.sub('(?P<quotation_mark>([。？！])[”’"\'])', r'\g<quotation_mark>\n', document)  # 特殊引号
        elif flag == "en":
            document = re.sub('(?P<quotation_mark>([.?!](?![”’"\'])))', r'\g<quotation_mark>\n', document)  # 英文单字符断句符
            document = re.sub('(?P<quotation_mark>([?!.]["\']))', r'\g<quotation_mark>\n', document)  # 特殊引号
        else:
            document = re.sub('(?P<quotation_mark>([。？！….?!](?![”’"\'])))', r'\g<quotation_mark>\n', document)  # 单字符断句符
            document = re.sub('(?P<quotation_mark>(([。？！.!?]|…{1,2})[”’"\']))', r'\g<quotation_mark>\n',
                              document)  # 特殊引号

        sent_list_ori = document.splitlines()
        for sent in sent_list_ori:
            sent = sent.strip()
            if not sent:
                continue
            else:
                while len(sent) > limit:
                    temp = sent[0:limit]
                    sent_list.append(temp)
                    sent = sent[limit:]
                sent_list.append(sent)
    except:
        sent_list.clear()
        sent_list.append(document)
    return sent_list


def get_predictions_seq2label(sentences, model, batch_size=1):
    subsents = []
    s_map = []
    for i, sent in enumerate(sentences):  # 将篇章划分为子句，分句预测再合并
        subsent_list = split_sentence(sent, flag="zh")
        s_map.extend([i for _ in range(len(subsent_list))])
        subsents.extend(subsent_list)
    assert len(subsents) == len(s_map)

    predictions = []
    cnt_corrections = 0
    batch = []
    for sent in subsents:
        batch.append(split_char(sent))
        # batch.append(sent.split())
        if len(batch) == batch_size:  # 如果数据够了一个batch的话，
            preds, cnt = model.handle_batch(batch)
            assert len(preds) == len(batch)
            predictions.extend(preds)
            cnt_corrections += cnt
            batch = []

    if batch:
        preds, cnt = model.handle_batch(batch)
        assert len(preds) == len(batch)
        predictions.extend(preds)
        cnt_corrections += cnt

    assert len(subsents) == len(predictions)

    results = ["" for _ in range(len(sentences))]
    for i, ret in enumerate(predictions):
        ret_new = [tok.lstrip("##") for tok in ret]
        ret = cc.convert("".join(ret_new))
        results[s_map[i]] += cc.convert(ret)
    predictions = results

    return predictions, cnt_corrections


def predict_seq2label(inputs, model, logger, model_name, batch_size):
    try:
        predictions, cnt_corrections = get_predictions_seq2label(inputs, model, batch_size)
        # logger.info("预测结果:"+predictions[0])
        logger.info(model_name + "成功处理用户请求！")
        # # 将模型model转到cpu
        # model = model.to('cpu')
        # # 删除模型，也就是删除引用
        # del model
        return predictions, cnt_corrections
    except Exception as e:
        logger.error("处理用户请求失败！")
        logger.error(str(e.message))


def get_predictions_seq2seq(model, tokenizer, sentences, device, batch_size=1, logger=None):
    predictions = []
    cnt_corrections = 0
    logger.info("   开始预测...")

    predict_dataset = MyDataset(sentences, tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    predict_dataloader = DataLoader(dataset=predict_dataset, collate_fn=data_collator, batch_size=batch_size)

    # input_ids = torch.tensor(tokenizer.encode(input, return_tensors='pt', add_special_tokens=False), device=device)
    for i, batch in enumerate(predict_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        generated_tokens = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                          num_beams=12, max_length=100)
        generated_tokens = generated_tokens.cpu().numpy()
        results = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        predictions += [cc.convert("".join(result).replace("##", "")) for result in results]

    # for idx, input in enumerate(sentences):
    #     input_ids = torch.tensor(tokenizer.encode(input, return_tensors='pt', add_special_tokens=False), device=device)
    #     pred_ids = model.generate(input_ids, num_beams=12, max_length=100)
    #     result = tokenizer.convert_ids_to_tokens(pred_ids[0], skip_special_tokens=False)[2:-1]
    #     predictions.append(cc.convert("".join(result).replace("##", "")))

    assert len(sentences) == len(predictions)
    return predictions, cnt_corrections


def predict_seq2seq(model, tokenizer, inputs, device, batch_size, logger, model_name):
    try:
        logger.info("   seq2seq预测：")
        predictions, cnt_corrections = get_predictions_seq2seq(model, tokenizer, inputs, device, batch_size, logger)
        logger.info(model_name + "成功处理用户请求！")
        # 将模型model转到cpu
        model = model.to('cpu')
        # 删除模型，也就是删除引用
        del model
        # for input, pred in zip(inputs, predictions):
        #     print(input)
        #     print(pred)
        return predictions, cnt_corrections
    except Exception as e:
        logger.info("处理用户请求失败！")
        logger.info(e)
        # print("处理用户请求失败")



def main():
    # get all paths
    model_dir = "./pretrain_weights/real_learner_bart_CGEC_new"
    inputs = ["我在超市里工作的是第一次的，所以我工作前很可怕，工作的速度也很慢。",
              "提高了定位的精度，并且相比GIoU对两个目标框的面积关系进行优化，DIoU直接优化两个目标框的距离，收敛还要更快[8]。",
              "而FN与FP都是预测错误的情形，FN是错将positive预测为negative，这也被称为统计学上的第一类错误(TypeIError)，FP则是错将negative预测为positive，这被称为统计学上的第二类错误(TypeIIError)。",
              ]
    predictions, cnt_corrections = predict_seq2seq(model_dir, inputs)
    srcs, tgts = postprocess(inputs, predictions, batch_size=3)
    srcs = ["".join(src.split(" ")).replace("##", "") for src in srcs]
    tgts = ["".join(tgt.split(" ")).replace("##", "") for tgt in tgts]
    # for src, tgt in zip(srcs, tgts):
    #     print("".join(src.split(" ")).replace("##", ""))
    #     print("".join(tgt.split(" ")).replace("##", ""))


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()