import random

# from model.config import *
# from model.align import ZhAnnotator
# from model.gector.gec_model import GecBERTModel
from flask import Flask, request, render_template, make_response
from flask_caching import Cache
import json
import re
import logging
import time, datetime
from flask_cors import CORS
import sqlite3
import copy
import requests
import os
# import torch
import argparse
import base64
import subprocess
# from concurrent.futures import ThreadPoolExecutor
from model.preprocess.preprocess import preprocess
from pprint import pprint
import difflib
from model.postprocess.postprocess import postprocess
from model.predict import predict_seq2seq, predict_seq2label
# import multiprocessing
from multiprocessing.pool import ThreadPool
# from multiprocessing import Pool
# from transformers import BartForConditionalGeneration, BertTokenizer
from utils.sendEmail import sendEmail, sendVerifyEmail
from utils.cache import Cache

# from utils.GPU_autochoice import GPUManager

processes = 5
pool = ThreadPool(processes=processes)
# pool = multiprocessing.Pool(processes=2)

# executor = ThreadPoolExecutor(6)
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = os.urandom(24)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

cache = Cache()
cache.set('all_task', list())  # 设置一个缓存对象

logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)
start_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
handler = logging.FileHandler('./logs/log_2023.txt'.format(str(start_time)))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("日志系统启动")
# annotator = ZhAnnotator.create_default(annotator_id=0)
logger.info("进程队列初始化")

# all_task = list()   # 收集当前运行进程
d = difflib.Differ()
# BATCH_SIZE = 6
# DEVICE = [0]
# GM = GPUManager(DEVICE)

status = {
    0: "上传成功，前方剩余待纠错文件:",
    1: "纠错完成",
    2: "出现未知错误，请重新提交",
    3: "模型纠错中"
}


# def set_model(vocab_path, model_paths, weights_names, device, bert_dim=1024):
#     model = GecBERTModel(vocab_path=vocab_path,
#                          model_paths=model_paths,
#                          weights_names=weights_names,
#                          max_len=max_len,
#                          min_len=min_len,
#                          iterations=iteration_count,
#                          min_error_probability=min_error_probability,
#                          min_probability=min_probability,
#                          lowercase_tokens=0,
#                          special_tokens_fix=0,
#                          log=False,
#                          confidence=additional_confidence,
#                          is_ensemble=is_ensemble,
#                          weigths=None,
#                          cuda_device=device,
#                          mode=mode,
#                          pinyin=pinyin,
#                          crf=crf,
#                          mtl_pos=mtl_pos,
#                          bert_dim=bert_dim,
#                          )
#     logger.info("模型创建完成")
#     return model

@app.route('/login', methods=['POST'])  # 登录
def login():
    # 传参：用户名UserName，密码PassWord
    now_time = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now() + datetime.timedelta(hours=8))
    username = request.values.get("UserName")
    password = request.values.get("PassWord")
    if len(password) < 6:
        return {
            'code': 402,
            'message': "密码长度应不少于6位！"
        }
    conn = sqlite3.connect('./database/mydata.db')
    cur = conn.cursor()
    correct_password = cur.execute("SELECT password FROM user WHERE username = ?", (username,)).fetchone()
    res = {
        'code': 200,
        'message': "登录成功！"
    }
    if correct_password is None:
        res['code'] = 400
        res['message'] = "邮箱不存在！"
        return res
    elif correct_password[0] != password:
        res['code'] = 401
        res['message'] = "密码错误！"
        return res
    else:
        now_time = str(now_time)
        cur.execute("UPDATE user SET time = ? WHERE username = ?", (now_time, username))  # 更新最近登录时间
    conn.commit()
    conn.close()
    return res


@app.route('/register', methods=['POST'])  # 注册
def register():
    # 传参：用户名UserName，密码PassWord，身份Identity，验证码VerificationCode
    # 账号限制为学校认证的邮箱，注册时需要邮箱验证。
    now_time = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now() + datetime.timedelta(hours=8))
    username = request.values.get("UserName")
    password = request.values.get("PassWord")
    identity = request.values.get("Identity")
    verification_code = request.values.get("VerificationCode")
    logger.info(username)
    logger.info(password)
    logger.info(identity)
    logger.info(verification_code)
    conn = sqlite3.connect('./database/mydata.db')
    cur = conn.cursor()
    correct_verification_code = \
    cur.execute("SELECT verification_code FROM verify WHERE username = ?", (username,)).fetchone()[0]
    if verification_code is None or len(verification_code) != 6 or verification_code != correct_verification_code:
        return {
            'code': 405,
            'message': "请输入正确验证码！"
        }
    if not is_valid_email(username):
        return {
            'code': 403,
            'message': "邮箱格式有误！"
        }
    if len(username) < 6 or len(password) < 6:
        return {
            'code': 402,
            'message': "用户名与密码长度不少于6位！"
        }
    correct_password = cur.execute("SELECT password FROM user WHERE username = ?", (username,)).fetchone()
    if correct_password != None:
        return {
            'code': 404,
            'message': "用户名已存在！"
        }
    cur.execute("INSERT INTO user VALUES (?,?,?,?,?,?)", (username, password, now_time, None, identity, None))
    conn.commit()
    conn.close()
    res = {
        'code': 200,
        'message': "注册成功！"
    }
    return res


# def gec_task(file_id, json_path, doc=0):
#     pool = ThreadPool(processes=5)
#     start_time = time.time()
#     inputs = preprocess(file_id, json_path, doc)
#     logger.info("子进程开始纠错, file_id:" + str(file_id))
#     logger.info("句子数量：" + str(len(inputs)))
#     # logger.info(inputs)
#     try:
#         # seq2seq
#         # seq2seq_models = ["real_learner_bart_CGEC_new", "real_learner_bart_CGEC_exam", "real_learner_bart_CGEC"]
#         seq2seq_models = []
#
#         predictions_models = []
#         pool_results = []
#         for model_name in seq2seq_models:
#             device_id = GM.auto_choice()
#             device = torch.device('cuda:%d' % device_id if device_id >= 0 else 'cpu')
#             logger.info("加载模型:" + model_name + " GPU编号：" + str(device_id))
#             model_dir = "model/pretrain_weights/" + model_name
#             tokenizer = BertTokenizer.from_pretrained(model_dir)
#             model = BartForConditionalGeneration.from_pretrained(model_dir)
#             logger.info("分配GPU资源")
#             model.to(device)
#             model.eval()
#             # predictions, cnt_corrections = pool.apply_async(predict_seq2seq, (model1, tokenizer1, inputs, device, BATCH_SIZE, logger)).get()
#             # predictions, cnt_corrections = predict_seq2seq(model1, tokenizer1, inputs, device, BATCH_SIZE, logger)
#             pool_results.append(pool.apply_async(predict_seq2seq, (model, tokenizer, inputs, device, BATCH_SIZE, logger, model_name)))
#
#         # seq2edit
#         seq2edit_models = [
#             # ["output_vocabulary_chinese_char_hsk+lang8_5",
#             #  'Best_Model_Stage_2.th',
#             #  'structbert-large-zh',
#             # 1024],
#             # ["output_vocabulary_chinese_char_hsk+lang8_10",
#             #  'bert_wwm_ext_BT_Best_Model_Stage_3.th',
#             #  'chinese-bert-wwm-ext',
#             #  768],
#             ["output_vocabulary_chinese_char_hsk+lang8_10",
#              'macbert_BT_Best_Model_Stage_3.th',
#              'chinese-macbert-large',
#              1024],
#             # ["output_vocabulary_chinese_char_hsk+lang8_5",
#             #  'MB_DA_FT_EP20.th',
#             #  'chinese-struct-bert',
#             #  1024],
#             # ["output_vocabulary_chinese_char_hsk+lang8_10",
#             #  'RB_DA_FT_EP20.th',
#             #  'chinese-struct-bert',
#             #  768],
#             # ["output_vocabulary_chinese_char_hsk+lang8_10",
#             #  'SB_DA_FT_EP20.th',
#             #  'chinese-struct-bert',
#             #  768],
#         ]
#         for model_args in seq2edit_models:
#             device_id = GM.auto_choice()
#             model_name = model_args[2] + " " + model_args[1]
#             logger.info("加载模型:" + model_name + " GPU编号：" + str(device_id))
#             model = set_model(vocab_path="model/data/" + model_args[0],
#                               model_paths=['model/model_files/' + model_args[1]],
#                               weights_names=['model/pretrain_weights/' + model_args[2]],
#                               device=device_id,
#                               bert_dim=model_args[3])
#             # predictions, _ = predict_seq2label(inputs, model, logger, model_name)
#             pool_results.append(pool.apply_async(predict_seq2label, (inputs, model, logger, model_name)))
#         pool.close()  # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用
#         pool.join()  # 等待进程池中的所有进程执行完毕
#         for pool_result in pool_results:
#             predictions_models.append(pool_result.get()[0])
#
#         inputs_models = [inputs] * (len(seq2seq_models) + len(seq2edit_models))
#         srcs, tgts = postprocess(inputs_models, predictions_models, batch_size=BATCH_SIZE, min_vote=1, logger=logger)
#         predictions = ["".join(tgt.split(" ")).replace("##", "") for tgt in tgts]
#         logger.info("file_id:" + str(file_id) + " 纠错成功，保存纠错结果到数据库中...")
#         # logger.info(predictions)
#         conn = sqlite3.connect('./database/mydata.db')
#         cur = conn.cursor()
#         # assert len(inputs) == len(predictions), logger.info(len(inputs), len(predictions))
#         # logger.info("len(inputs), len(predictions)" + str(len(inputs)) + str(len(predictions)))
#         for i in range(len(inputs)):
#             # logger.info(inputs[i])
#             # logger.info(predictions[i])
#             if inputs[i] != predictions[i]:
#                 if inputs[i] not in ["对本文的研究作出重要贡献的个人和集体，均已在文中以明确方式标明。", "本设计（论文）属在年月解密后适用本规定。"]:
#                     # logger.info("====:" + predictions[i])
#                     cur.execute("INSERT INTO errors VALUES (?, ?, ?, ?, ?, ?)",
#                                 (None, file_id, 0, inputs[i], predictions[i], None))
#             conn.commit()
#
#         # 纠错完成，更新status
#         logger.info("file_id:" + str(file_id) + "保存完成!")
#         cur.execute("UPDATE commit_record SET status = 1 WHERE file_id = %s" % file_id)
#         conn.commit()
#         flag = cur.execute("SELECT is_email FROM commit_record WHERE file_id = %s" % file_id).fetchone()[0]
#         if flag:
#             username = cur.execute("SELECT username FROM commit_record WHERE file_id = %s" % file_id).fetchone()[0]
#             email = cur.execute("SELECT email FROM user WHERE username = '{}'".format(username)).fetchone()[0]
#             send_email(email, username, file_id)
#             logger.info("提醒邮件发送成功！")
#         conn.close()
#
#         train_seconds = int(time.time() - start_time)
#         logger.info('训练总耗时={0}'.format(str(train_seconds)))
#     except Exception as e:
#         logger.info("子进程出错")
#         logger.info(e)
#
#     # # 将模型model转到cpu
#     # model = model.to('cpu')
#     # # 删除模型，也就是删除引用
#     # del model
#     # 释放GPU。
#     torch.cuda.empty_cache()
#     return


@app.route('/save_file', methods=['POST'])  # 用户文件上传
def save():
    # todo   限制用户提交次数（初步定为每10分钟只能交三次）   数据库记录上次提交时间，3分钟内不予提交
    # todo   预测任务改为发送给一个GPU服务器处理，此项目在cpu服务器上运行，作为跳板机    能否在cpu服务器和gpu服务器各部署一个flask，通过url来传数据？待定
    all_task = cache.get('all_task')

    def Except(e):
        print(e)
        logger.info(e)
        return e

    # 传参：文件file、用户名UserName、时间Date
    now_time = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now() + datetime.timedelta(hours=8))
    data = request.form
    logger.info("收到用户上传请求!")
    username = data.get("UserName")
    Date = data.get("Date")
    file = request.files.get("file")
    if file is None:  # 表示没有发送文件
        return json.dumps({
            'code': 500,
            'message': "文件上传失败",
            'Date': Date,
            'Status': "上传失败",
            'FileName': "",
            'FileId': -1,
        })
    conn = sqlite3.connect('./database/mydata.db')
    cur = conn.cursor()
    file_name = file.filename.replace(" ", "")
    logger.info("获取上传文件的名称为：" + file_name)
    file_id = cur.execute("SELECT max(file_id) FROM commit_record").fetchall()[0][0]
    if file_id is not None:
        file_id += 1
    else:
        file_id = 0
    doc = 0
    if file_name.endswith('.pdf'):
        file_path = f"./articles/{file_id}.pdf"  # 用file_id命名，防止同一文件保存时导致覆盖
        file.save(file_path)  # 保存文件
    elif file_name.endswith('.doc'):
        file_path = f"./articles/{file_id}.doc"
        file.save(file_path)  # 保存文件
        # output = subprocess.check_output(["soffice",
        #                                   "--headless",
        #                                   "--invisible",
        #                                   "--convert-to",
        #                                   "docx",
        #                                   file_path,
        #                                   "--outdir", "/data3/hcjiang/backend/articles"])
        file_path = f"./articles/{file_id}.docx"
        doc = 1
    else:
        file_path = f"./articles/{file_id}.docx"
        file.save(file_path)  # 保存文件
    json_path = f"./jsons/{file_id}.json"

    cur.execute("insert into commit_record values (?, ?, ?, ?, ?, ?, ?, ?)",
                (file_id, username, file_name, file_path, json_path, now_time, 0, 0))
    conn.commit()
    logger.info("文件已保存到数据库")
    # 更新纠错队列信息
    for task in all_task[:]:
        label = cur.execute("SELECT status FROM commit_record WHERE file_id = %s" % task).fetchone()[0]
        if label == 1 or label == 2:
            logger.info("移除进程，file_id:" + str(task))
            all_task.remove(task)
    if len(all_task) < processes:
        Status = status[3]
    else:
        Status = status[0] + str(len(all_task) - processes + 1) + "，请耐心等待"
    res = {
        'code': 200,
        'message': "文件上传成功",
        'Date': Date,
        "Status": Status,
        'FileName': file_name,
        'FileId': file_id,
    }
    # 1. 直接在本机处理
    # pool.apply_async(gec_task, (file_id, json_path, doc), error_callback=Except)
    # all_task.append(file_id)
    # logger.info("新的纠错进程已加入队列,file_id:" + str(file_id))
    # cache.set('all_task', all_task)
    # return json.dumps(res)

    # 2. 分发任务到GPU服务器
    inputs = preprocess(file_id, json_path, doc)
    data = {
        "inputs": inputs,
        "file_id": file_id
    }
    url = 'http://192.168.134.31:5000/gec_task'  # 请求接口
    req = requests.post(url, data=json.dumps(data))  # 发送请求
    all_task.append(file_id)
    cache.set('all_task', all_task)
    return json.dumps(res)


@app.route('/receive', methods=['POST'])  # 请求论文提交记录
def reveive():
    args = json.loads(request.data)
    ans = args["ans"]
    file_id = args["file_id"]
    conn = sqlite3.connect('./database/mydata.db')
    cur = conn.cursor()
    for i in range(len(ans)):
        cur.execute("INSERT INTO errors VALUES (?, ?, ?, ?, ?, ?)",
                    (None, file_id, 0, ans[i][0], ans[i][1], None))
    conn.commit()

    # 纠错完成，更新status
    logger.info("file_id:" + str(file_id) + "保存完成!")
    cur.execute("UPDATE commit_record SET status = 1 WHERE file_id = %s" % file_id)
    conn.commit()
    # flag = cur.execute("SELECT is_email FROM commit_record WHERE file_id = %s" % file_id).fetchone()[0]
    # if flag:
    #     username = cur.execute("SELECT username FROM commit_record WHERE file_id = %s" % file_id).fetchone()[0]
    #     email = cur.execute("SELECT email FROM user WHERE username = '{}'".format(username)).fetchone()[0]
    #     send_email(email, username, file_id)
    #     logger.info("提醒邮件发送成功！")
    conn.close()
    return "200"


@app.route('/record', methods=['GET'])  # 请求论文提交记录
def record():
    # 传参：用户名UserName

    all_task = cache.get('all_task')
    username = request.args.get("UserName")
    conn = sqlite3.connect('./database/mydata.db')
    cur = conn.cursor()
    ans = cur.execute("SELECT * FROM commit_record WHERE username = ?", (username,)).fetchall()
    # print(ans)
    resp = []
    rec = dict()
    # 更新进程队列信息
    for task in all_task[:]:
        label = cur.execute("SELECT status FROM commit_record WHERE file_id = %s" % task).fetchone()[0]
        if label == 1 or label == 2:
            logger.info("移除进程，file_id:" + str(task))
            all_task.remove(task)
    logger.info("当前进程池：")
    logger.info([i for i in all_task])
    for index, item in enumerate(ans):
        flag = 0
        thread_pool_count = 0
        for task in all_task:
            if task != item[0]:
                thread_pool_count += 1
            else:
                flag = 1  # 任务队列中没有出现该file_id，若status==0则说明该任务意外退出，需要修改status
                break
        # print("file_id:", item[0], "排队数:", thread_pool_count)

        rec["UserName"] = item[1]
        rec["FileId"] = " " + str(item[0])
        rec["Date"] = item[5]
        rec["FileName"] = item[2]
        rec["Status"] = status[item[6]]
        if item[6] == 0:  # 未纠错完成，需要判断下是否还在纠错，其他情况不用改变状态
            if flag == 0:  # 无该纠错进程
                logger.info(str(item[0]) + "进程意外出错，进程池：")
                logger.info([i for i in all_task])
                cur.execute("UPDATE commit_record SET status = ? WHERE file_id = ?", (2, item[0]))
                conn.commit()
                rec["Status"] = status[2]  # 意外出错
            else:
                if thread_pool_count < processes:
                    rec["Status"] = status[3]
                else:
                    rec["Status"] = status[0] + str(thread_pool_count - processes + 1) + "，请耐心等待"

        # if item[6] == 2 and flag == 1:
        #     cur.execute("UPDATE commit_record SET status = ? WHERE file_id = ?", (0, item[0]))
        #     conn.commit()
        #     rec["Status"] = status[0]   # 上传成功

        resp.append(copy.deepcopy(rec))
    resp.reverse()
    cache.set('all_task', all_task)
    conn.commit()
    conn.close()
    return resp


@app.route('/correct', methods=['GET'])  # 提取纠错记录
def correct():
    all_task = cache.get('all_task')
    # 传参：论文编号FileId
    if request.method == 'GET':
        now_time = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now() + datetime.timedelta(hours=8))
        logger.info("收到用户纠错请求!")
        conn = sqlite3.connect('./database/mydata.db')
        cur = conn.cursor()
        file_id = request.args.get('FileId')
        ans = cur.execute("SELECT * FROM errors WHERE file_id = %s" % file_id).fetchall()
        file_name = cur.execute("SELECT file_name FROM commit_record WHERE file_id = %s" % file_id).fetchone()[0]
        status1 = cur.execute("SELECT status FROM commit_record WHERE file_id = %s" % file_id).fetchone()[0]
        Errors = []
        error = dict()

        for index, item in enumerate(ans):
            source = item[3]
            target = item[4]
            l = list(d.compare(source, target))
            tgt = []
            for i in l:
                if i[0] == ' ':
                    tgt.append([i[-1], 0])
                elif i[0] == '-':
                    tgt.append([i[-1], 1])
                elif i[0] == '+':
                    tgt.append([i[-1], 2])
            error["id"] = item[0]
            error["page"] = item[2]
            error["src"] = source
            error["tgt"] = tgt
            Errors.append(copy.deepcopy(error))
        # 更新纠错队列信息
        for task in all_task[:]:
            label = cur.execute("SELECT status FROM commit_record WHERE file_id = %s" % task).fetchone()[0]
            if label == 1 or label == 2:
                logger.info("移除进程，file_id:" + str(task))
                all_task.remove(task)
        logger.info("当前进程池：")
        logger.info([i for i in all_task])
        thread_pool_count = 0
        logger.info("提取进程：" + str(file_id))
        for task in all_task:
            if str(task) != str(file_id):
                thread_pool_count += 1
            else:
                break
        logger.info("进程前队列长度：" + str(thread_pool_count))
        if status1 == 2:  # 进程意外出错
            logger.info(str(file_id) + "进程意外出错，进程池：")
            logger.info([i for i in all_task])
            Status = status[2]
        elif status1 == 0:
            if thread_pool_count < processes:
                Status = status[3]
            else:
                Status = status[0] + str(thread_pool_count - processes + 1) + "，请耐心等待"
        elif status1 == 1:
            Status = "纠错完成"
        cache.set('all_task', all_task)
        conn.commit()
        conn.close()
        if len(Errors) == 0:
            resp = [now_time, file_name, None, Status]
        else:
            resp = [now_time, file_name, Errors, Status]
        return resp


@app.route('/judge', methods=['POST'])  # 用户反馈
def judge():
    # 传参：纠错文本编号id, judge
    id = request.values.get('id')
    judge = request.values.get('judge')
    # print(id, judge)
    conn = sqlite3.connect('./database/mydata.db')
    cur = conn.cursor()
    cur.execute("UPDATE errors SET judge = ? WHERE id = ?", (judge, id))
    conn.commit()
    conn.close()
    return {
        'code': 200,
        'message': "反馈成功！"
    }


@app.route('/information', methods=['POST'])  # 查看个人信息
def information():
    # 传参：用户名username
    username = request.values.get('UserName')
    conn = sqlite3.connect('./database/mydata.db')
    cur = conn.cursor()
    ans = cur.execute("SELECT * FROM user WHERE username = ?", (username,)).fetchone()
    conn.close()
    return {
        'code': 200,
        'UserName': ans[0],
        'PassWord': ans[1],
        'Identity': ans[4],
        'Message': "修改成功！"
    }
    # ruleform = [200, ans[0], ans[1], ans[3], ans[4], ans[5], "修改成功！"]
    # return ruleform


@app.route('/update_information', methods=['POST'])  # 个人信息维护
def update_information():
    # todo 把这五个都更新到数据库
    username = request.values.get('UserName')
    password = request.values.get('PassWord')
    identity = request.values.get('Identity')
    # if not is_valid_email(username):
    #     return {
    #         'code': 500,
    #         'message': "邮箱格式有误！"
    #     }
    # print(id, judge)
    conn = sqlite3.connect('./database/mydata.db')
    cur = conn.cursor()
    cur.execute("UPDATE user SET password = ?, identity = ? WHERE username = ?", (password, identity, username))
    conn.commit()
    conn.close()
    return {
        'code': 200,
        'message': "修改成功！"
    }


@app.route('/verify', methods=['POST'])  # 发送验证码
def verify():
    # 传参：用户名UserName
    now_time = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now() + datetime.timedelta(hours=8))
    username = request.values.get("UserName")
    logger.info(username)
    if not is_valid_email(username):
        return {
            'code': 403,
            'message': "邮箱格式有误！"
        }
    arrs = [random.randint(0, 9) for _ in range(6)]  # 随机生成6位数验证码
    verification_code = ''.join(list(map(str, arrs)))
    conn = sqlite3.connect('./database/mydata.db')
    cur = conn.cursor()
    is_exist_data = cur.execute("SELECT * FROM verify WHERE username = ?", (username,)).fetchone()
    if not is_exist_data:
        cur.execute("INSERT INTO verify VALUES (?,?,?)", (username, verification_code, now_time))
    # todo 数据库表中加入count，seconds限制扩大到10min，次数限制为三次
    else:
        last_time = cur.execute("SELECT last_time FROM verify WHERE username = ?", (username,)).fetchone()[0]
        last_time = datetime.datetime.strptime(last_time, "%Y-%m-%d %H:%M:%S")
        now_time = datetime.datetime.strptime(now_time, "%Y-%m-%d %H:%M:%S")
        seconds = (now_time - last_time).seconds
        if seconds < 60:
            return {
                'code': 600,
                'message': "发送验证码过于频繁，请1分钟后再试。"
            }
        else:
            cur.execute("UPDATE verify SET verification_code = ?, last_time = ? WHERE username = ?",
                        (verification_code, now_time, username))
    conn.commit()
    receivers = [username]
    logger.info(receivers, verification_code)
    sendVerifyEmail(receivers, verification_code, logger)
    conn.close()
    return {
        'code': 200,
        'message': "发送成功！"
    }


# @app.route('/mail', methods=['GET'])  # 发送邮件提醒
def send_email(receiver, username, file_id):
    # username = request.args.get('UserName')
    # file_id = request.args.get('FileId')
    conn = sqlite3.connect('./database/mydata.db')
    cur = conn.cursor()
    file_name = cur.execute("SELECT file_name FROM commit_record WHERE file_id = %s" % file_id).fetchone()[0]
    receivers = [receiver]
    sendEmail(receivers, username, file_name)
    conn.close()


def is_valid_email(email):  # 检验邮箱格式
    # todo 限制为学校认证的邮箱   暂定修改为edu.cn结尾
    pattern = re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+edu.cn$')
    return pattern.match(email)


@app.route('/')
def hello():
    return "Hello"


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    app.run(host='0.0.0.0', port=81, debug=True)
