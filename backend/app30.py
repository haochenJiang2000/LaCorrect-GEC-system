import random
import hashlib
# from model.config import *
# from model.align import ZhAnnotator
# from model.gector.gec_model import GecBERTModel
from flask import Flask, session, request, render_template, make_response
from flask.sessions import SecureCookieSessionInterface
from flask_session import Session
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
from model.preprocess.split_data import split_data
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
from flask import send_file, jsonify, send_from_directory

processes = 20
ensamble_processes = 1
pool = ThreadPool(processes=processes)
# pool = multiprocessing.Pool(processes=2)

# executor = ThreadPoolExecutor(6)
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask-session'
app.config['SESSION_KEY'] = os.urandom(24)
# app.secret_key = "hello,lacorrect"
# session.permanent = True
# app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=15)
app.permanent_session_lifetime = datetime.timedelta(minutes=15)  # 设置session的保存时间。
# CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
CORS(app, supports_credentials=True)  # 解决跨域问题

cache = Cache()
cache.set('all_task', list())  # 设置一个缓存对象
cache.set('ensamble_task', list())  # 设置一个缓存对象
Session(app)
session_cookie = SecureCookieSessionInterface().get_signing_serializer(app)

logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)
start_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
handler = logging.FileHandler('./logs/log_2023.txt'.format(str(start_time)))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("日志系统启动")
logger.info("进程队列初始化")

d = difflib.Differ()

status = {
    0: "上传成功，前方剩余待纠错文件:",
    1: "纠错完成",
    2: "出现未知错误，请重新提交",
    3: "模型纠错中"
}

mode = {
    '速度优先': 0,
    '均衡': 1,
    '质量优先': 2
}

def gec_task(file_id, json_path, doc, GEC_mode):
    inputs = preprocess(file_id, json_path, doc)
    data = {
        "inputs": inputs,
        "file_id": file_id,
        "GEC_mode": GEC_mode,
    }
    url = 'http://192.168.134.32:5000/gec_task'  # 请求接口
    requests.post(url, data=json.dumps(data))  # 发送请求

@app.route('/login', methods=['POST'])  # 登录
def login():
    # 传参：用户名UserName，密码PassWord
    now_time = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
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
    md5 = hashlib.md5()  # md5对象，md5不能反解，但是加密是固定的

    if correct_password is None:
        res['code'] = 400
        res['message'] = "邮箱不存在！"
        return res
    else:
        salt = cur.execute("SELECT salt FROM user WHERE username = ?", (username,)).fetchone()[0]
        if salt:
            password = password + salt
            md5.update(password.encode('utf-8'))  # 要对哪个字符串进行加密，就放这里
            password = md5.hexdigest()  # 拿到加密字符串
        if correct_password[0] != password:
            res['code'] = 401
            res['message'] = "密码错误！"
            return res
        else:
            session['username'] = username
            # set_session("username", username)
            print("登录：", session.get('username'))
            # 设置Session存活时间
            # session.permanent = True
            cur.execute("UPDATE user SET time = ? WHERE username = ?", (now_time, username))  # 更新最近登录时间
    conn.commit()
    conn.close()
    return res


@app.route('/register', methods=['POST'])  # 注册
def register():
    # 传参：用户名UserName，密码PassWord，身份Identity，验证码VerificationCode
    # 账号限制为学校认证的邮箱，注册时需要邮箱验证。
    now_time = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    username = request.values.get("UserName")
    password = request.values.get("PassWord")
    identity = request.values.get("Identity")
    verification_code = request.values.get("VerificationCode")
    # print(username, password, identity, verification_code)
    if username == "" or password == "" or identity == "" or verification_code == "":
        return {
            'code': 600,
            'message': "请完善注册信息！"
        }
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
    salt = v_code(20)
    password = password + salt
    md5 = hashlib.md5()  # md5对象，md5不能反解，但是加密是固定的
    # update需要一个bytes格式参数
    md5.update(password.encode('utf-8'))  # 要对哪个字符串进行加密，就放这里
    password = md5.hexdigest()  # 拿到加密字符串
    cur.execute("INSERT INTO user VALUES (?,?,?,?,?,?,?)", (username, password, now_time, None, identity, 0, salt))
    conn.commit()
    conn.close()
    res = {
        'code': 200,
        'message': "注册成功！"
    }
    print("注册成功")
    return res

@app.route('/test', methods=['POST'])  # 测试
def test():
    now_time = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    username = request.values.get("UserName")
    sentences = request.values.get("Sentences")
    # GEC_mode = ""
    GEC_mode = request.values.get("GEC_mode_test")
    logger.info("GEC_mode:"+GEC_mode)
    if not valid_username(username):
        return {
            'code': 707,
            'message': "请先登录"
        }
    conn = sqlite3.connect('./database/mydata.db')
    cur = conn.cursor()
    ans = cur.execute("SELECT * FROM test WHERE username = ?", (username,)).fetchall()
    if len(ans) > 1:
        last_time = datetime.datetime.strptime(ans[-1][4], "%Y-%m-%d %H:%M:%S")
        # now_time = datetime.datetime.strptime(now_time, "%Y-%m-%d %H:%M:%S")
        seconds = (datetime.datetime.now() - last_time).seconds
        if seconds < 1:
            return {
                'code': 500,
                'message': "提交太频繁，请稍等再试。"
            }
    if len(ans) > 20:
        last_time = datetime.datetime.strptime(ans[-20][4], "%Y-%m-%d %H:%M:%S")
        # now_time = datetime.datetime.strptime(now_time, "%Y-%m-%d %H:%M:%S")
        seconds = (datetime.datetime.now() - last_time).seconds
        if seconds < 600:
            return {
                'code': 501,
                'message': "10分钟内只能提交20次。"
            }

    inputs = split_data([sentences])
    data = {
        "inputs": inputs,
        "GEC_mode": GEC_mode,
    }
    url = 'http://192.168.134.32:5000/test_gec_task'  # 请求接口
    logger.info("输入句：")
    logger.info(inputs)
    req = requests.post(url, data=json.dumps(data)).text  # 发送请求
    req = json.loads(req)["ans"][0]
    logger.info("处理成功！纠错结果：")
    logger.info(req)
    source = req[0]
    target = req[1]
    id = cur.execute("SELECT max(id) FROM test").fetchall()[0][0]
    if id is not None:
        id += 1
    else:
        id = 0
    cur.execute("INSERT INTO test VALUES (?, ?, ?, ?, ?, ?)", (id, username, ''.join(source), ''.join(target), now_time, 0))
    conn.commit()
    l = list(d.compare(source, target))
    tgt = []
    error = {}
    for i in l:
        if i[0] == ' ':
            tgt.append([i[-1], 0])
        elif i[0] == '-':
            tgt.append([i[-1], 1])
        elif i[0] == '+':
            tgt.append([i[-1], 2])
    error["src"] = source
    error["tgt"] = tgt
    # Errors = []
    # for i in range(len(source)):
    #     l = list(d.compare(source[i], target[i]))
    #     tgt = []
    #     error = {}
    #     for i in l:
    #         if i[0] == ' ':
    #             tgt.append([i[-1], 0])
    #         elif i[0] == '-':
    #             tgt.append([i[-1], 1])
    #         elif i[0] == '+':
    #             tgt.append([i[-1], 2])
    #     error["src"] = source
    #     error["tgt"] = tgt
    #     Errors.append(copy.deepcopy(error))
    # print(error)
    # return error
    logger.info("纠错结果返回前端")
    return {
        'code': 200,
        'message': "纠错完成",
        'error': error,
    }

@app.route('/save_file', methods=['POST'])  # 用户文件上传
def save():
    all_task = cache.get('all_task')
    ensamble_task = cache.get('ensamble_task')
    # task = cache.get('all_task')
    # GEC_mode = ""
    GEC_mode = request.values.get("GEC_mode")
    def Except(e):
        print(e)
        logger.info(e)
        return e

    # 传参：文件file、用户名UserName、时间Date
    now_time = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    data = request.form
    logger.info("收到用户上传请求!")
    username = data.get("UserName")
    Date = data.get("Date")
    file = request.files.get("file")
    logger.info("GEC_mode:"+GEC_mode)
    # GEC_mode = request.files.get("GEC_mode")

    # if not valid_username(username):
    #     return {
    #         'code': 707,
    #         'message': "请先登录"
    #     }
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
    ans = cur.execute("SELECT * FROM commit_record WHERE username = ?", (username,)).fetchall()
    if len(ans) > 3:
        last_time = datetime.datetime.strptime(ans[-3][5], "%Y-%m-%d %H:%M:%S")
        now_time = datetime.datetime.strptime(now_time, "%Y-%m-%d %H:%M:%S")
        seconds = (now_time - last_time).seconds
        if seconds < 600:
            return {
                'code': 501,
                'message': "10分钟内只能提交3篇论文！"
            }
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
        file_path = f"./articles/{file_id}.docx"
        doc = 1
    else:
        file_path = f"./articles/{file_id}.docx"
        file.save(file_path)  # 保存文件
    json_path = f"./jsons/{file_id}.json"

    cur.execute("insert into commit_record values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (file_id, username, file_name, file_path, json_path, now_time, 0, 0, 0, mode[GEC_mode]))
    conn.commit()
    logger.info("文件已保存到数据库")
    if GEC_mode != "质量优先":
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
    else:
        # 更新纠错队列信息
        for task in ensamble_task[:]:
            label = cur.execute("SELECT status FROM commit_record WHERE file_id = %s" % task).fetchone()[0]
            if label == 1 or label == 2:
                logger.info("移除进程，file_id:" + str(task))
                ensamble_task.remove(task)
        if len(ensamble_task) < ensamble_processes:
            Status = status[3]
        else:
            Status = status[0] + str(len(ensamble_task) - ensamble_processes + 1) + "，请耐心等待"
    res = {
        'code': 200,
        'message': "文件上传成功",
        'Date': Date,
        "Status": Status,
        'FileName': file_name,
        'FileId': file_id,
    }

    # 2. 分发任务到GPU服务器
    pool.apply_async(gec_task, (file_id, json_path, doc, GEC_mode), error_callback=Except)

    if GEC_mode != "质量优先":
        all_task.append(file_id)
    else:
        ensamble_task.append(file_id)
    cache.set('all_task', all_task)
    cache.set('ensamble_task', ensamble_task)
    return json.dumps(res)

@app.route("/download", methods=['POST'])
def download_file():
    username = request.values.get("UserName")
    file_id = request.values.get("FileId")
    if not valid_username(username):
        return {
            'code': 707,
            'message': "请先登录"
        }
    conn = sqlite3.connect('./database/mydata.db')
    cur = conn.cursor()
    filename = cur.execute("SELECT file_path FROM commit_record WHERE file_id = ?", (file_id,)).fetchone()[0].split("/")[-1]
    directory = os.getcwd() + "/articles"
    # print(directory, filename)
    conn.close()

    file_path = directory + "/" + filename
    if os.path.exists(file_path):
        return send_from_directory(directory, filename, as_attachment=True)
    else:
        return jsonify({
            'message': '文件不存在',
        })

    # return send_from_directory(directory, filename, as_attachment=True)

@app.route('/receive', methods=['POST'])  # 请求论文提交记录
def receive():
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
    conn.close()
    return "200"

@app.route('/error', methods=['POST'])  # 返回报错file_id，更新进程池
def error():
    args = json.loads(request.data)
    all_task = cache.get('all_task')
    ensamble_task = cache.get('ensamble_task')
    file_id = args["file_id"]
    if file_id in all_task:
        all_task.remove(file_id)
        cache.set('all_task', all_task)
    else:
        ensamble_task.remove(file_id)
        cache.set('ensamble_task', ensamble_task)
    return "200"


@app.route('/record', methods=['POST'])  # 请求论文提交记录
def record():
    # 传参：用户名UserName
    all_task = cache.get('all_task')
    ensamble_task = cache.get('ensamble_task')
    username = request.values.get("UserName")
    # print(get_session("username"))
    print("session:", session.get('username'))
    if not valid_username(username):
        return {
            'code': 707,
            'message': "请先登录"
        }
    conn = sqlite3.connect('./database/mydata.db')
    cur = conn.cursor()
    ans = cur.execute("SELECT * FROM commit_record WHERE username = ?", (username,)).fetchall()
    ans = [i for i in ans if i[7] == 0]  # 不显示已经删除的论文信息
    # print("论文提交记录：")
    # pprint(ans)
    resp = []
    rec = dict()
    # 更新进程队列信息
    for task in all_task[:]:
        label = cur.execute("SELECT status FROM commit_record WHERE file_id = %s" % task).fetchone()[0]
        if label == 1 or label == 2:
            logger.info("移除进程，file_id:" + str(task))
            all_task.remove(task)
    for task in ensamble_task[:]:
        label = cur.execute("SELECT status FROM commit_record WHERE file_id = %s" % task).fetchone()[0]
        if label == 1 or label == 2:
            logger.info("移除进程，file_id:" + str(task))
            ensamble_task.remove(task)

    logger.info("当前进程池：")
    logger.info([i for i in all_task])
    for index, item in enumerate(ans):
        flag = 0
        thread_pool_count = 0
        gec_mode = item[9]
        if gec_mode != 2:
            for task in all_task:
                if task != item[0]:
                    thread_pool_count += 1
                else:
                    flag = 1  # 任务队列中没有出现该file_id，若status==0则说明该任务意外退出，需要修改status
                    break
        else:
            for task in ensamble_task:
                if task != item[0]:
                    thread_pool_count += 1
                else:
                    flag = 1  # 任务队列中没有出现该file_id，若status==0则说明该任务意外退出，需要修改status
                    break
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
                if gec_mode != 2:
                    if thread_pool_count < processes:
                        rec["Status"] = status[3]
                    else:
                        rec["Status"] = status[0] + str(thread_pool_count - processes + 1) + "，请耐心等待"
                else:
                    if thread_pool_count < ensamble_processes:
                        rec["Status"] = status[3]
                    else:
                        rec["Status"] = status[0] + str(thread_pool_count - ensamble_processes + 1) + "，请耐心等待"

        resp.append(copy.deepcopy(rec))
    resp.reverse()
    cache.set('all_task', all_task)
    cache.set('ensamble_task', ensamble_task)
    conn.commit()
    conn.close()
    return resp


@app.route('/correct', methods=['POST'])  # 提取纠错记录
def correct():
    all_task = cache.get('all_task')
    ensamble_task = cache.get('ensamble_task')
    # 传参：论文编号FileId
    now_time = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    logger.info("收到用户纠错请求!")
    conn = sqlite3.connect('./database/mydata.db')
    cur = conn.cursor()
    file_id = request.values.get('FileId')
    username = request.values.get('UserName')
    if not valid_username(username):
        return {
            'code': 707,
            'message': "请先登录"
        }
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
    for task in ensamble_task[:]:
        label = cur.execute("SELECT status FROM commit_record WHERE file_id = %s" % task).fetchone()[0]
        if label == 1 or label == 2:
            logger.info("移除进程，file_id:" + str(task))
            ensamble_task.remove(task)
    gec_mode = cur.execute("SELECT gec_mode FROM commit_record WHERE file_id = ?", (file_id,)).fetchone()[0]
    thread_pool_count = 0
    if gec_mode != 2:
        logger.info("进程池：")
        logger.info([i for i in all_task])
        logger.info("提取进程：" + str(file_id))
        for task in all_task:
            if int(task) != int(file_id):
                thread_pool_count += 1
            else:
                break
        logger.info("进程前队列长度：" + str(thread_pool_count))
    else:
        logger.info("进程池：")
        logger.info([i for i in ensamble_task])
        logger.info("提取进程：" + str(file_id))
        for task in ensamble_task:
            if int(task) != int(file_id):
                thread_pool_count += 1
            else:
                break
        logger.info("进程前队列长度：" + str(thread_pool_count))
    if status1 == 2:  # 进程意外出错
        logger.info(str(file_id) + "进程意外出错，进程池：")
        logger.info([i for i in all_task])
        Status = status[2]
    elif status1 == 0:
        if gec_mode != 2:
            if thread_pool_count < processes:
                Status = status[3]
            else:
                Status = status[0] + str(thread_pool_count - processes + 1) + "，请耐心等待"
        else:
            if thread_pool_count < ensamble_processes:
                Status = status[3]
            else:
                Status = status[0] + str(thread_pool_count - ensamble_processes + 1) + "，请耐心等待"
    elif status1 == 1:
        Status = "纠错完成"
    cache.set('all_task', all_task)
    cache.set('ensamble_task', ensamble_task)
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
    if not valid_username(username):
        return {
            'code': 707,
            'message': "请先登录"
        }
    conn = sqlite3.connect('./database/mydata.db')
    cur = conn.cursor()
    ans = cur.execute("SELECT * FROM user WHERE username = ?", (username,)).fetchone()
    conn.close()
    Privacy = {
        0: "可公开使用",
        1: "乐改团队内部使用",
        2: "不可使用"
    }
    return {
        'code': 200,
        'UserName': ans[0],
        'PassWord': ans[1],
        'Identity': ans[4],
        'Privacy': Privacy[ans[5]],
        'Message': "获取信息成功！"
    }
    # ruleform = [200, ans[0], ans[1], ans[3], ans[4], ans[5], "修改成功！"]
    # return ruleform


@app.route('/update_information', methods=['POST'])  # 个人信息维护
def update_information():
    username = request.values.get('UserName')
    identity = request.values.get('Identity')
    privacy = request.values.get('Privacy')
    if not valid_username(username):
        return {
            'code': 707,
            'message': "请先登录"
        }
    Privacy = {
        "可公开使用": 0,
        "乐改团队内部使用": 1,
        "不可使用": 2
    }
    conn = sqlite3.connect('./database/mydata.db')
    cur = conn.cursor()
    cur.execute("UPDATE user SET identity = ?, privacy = ? WHERE username = ?", (identity, Privacy[privacy], username))
    conn.commit()
    conn.close()
    return {
        'code': 200,
        'message': "修改成功！"
    }

@app.route('/file_privacy', methods=['POST'])  # 修改论文隐私权限
def file_privacy():
    file_id = request.values.get('FileId')
    file_privacy = request.values.get('FilePrivacy')
    if not session.get("username"):
        return {
            'code': 707,
            'message': "请先登录"
        }
    Privacy = {
        "可公开使用": 0,
        "乐改团队内部使用": 1,
        "不可使用": 2
    }
    conn = sqlite3.connect('./database/mydata.db')
    cur = conn.cursor()
    cur.execute("UPDATE commit_record SET file_privacy = ? WHERE file_id = ?", (Privacy[file_privacy], file_id))
    conn.commit()
    conn.close()
    return {
        'code': 200,
        'message': "修改成功！"
    }

@app.route('/update_password', methods=['POST'])  # 修改密码
def update_password():
    username = request.values.get('UserName')
    password = request.values.get('PassWord')
    new_password = request.values.get('newPassWord')
    if not valid_username(username):
        return {
            'code': 707,
            'message': "请先登录"
        }
    if password == new_password:
        return {
            'code': 700,
            'message': "新密码不能和原密码一样"
        }
    conn = sqlite3.connect('./database/mydata.db')
    cur = conn.cursor()
    correct_password = cur.execute("SELECT password FROM user WHERE username = ?", (username,)).fetchone()[0]
    salt = cur.execute("SELECT salt FROM user WHERE username = ?", (username,)).fetchone()[0]
    if salt:
        md5 = hashlib.md5()  # md5对象，md5不能反解，但是加密是固定的
        new_password += salt
        md5.update(new_password.encode('utf-8'))  # 要对哪个字符串进行加密，就放这里
        new_password = md5.hexdigest()  # 拿到加密字符串
        md5 = hashlib.md5()  # md5对象，md5不能反解，但是加密是固定的
        password += salt
        md5.update(password.encode('utf-8'))  # 要对哪个字符串进行加密，就放这里
        password = md5.hexdigest()  # 拿到加密字符串
    if password != correct_password:
        return {
            'code': 701,
            'message': "原密码填写错误"
        }
    cur.execute("UPDATE user SET password = ? WHERE username = ?", (new_password, username))
    conn.commit()
    conn.close()
    return {
        'code': 200,
        'message': "修改成功！"
    }


@app.route('/verify', methods=['POST'])  # 发送验证码
def verify():
    # 传参：用户名UserName
    now_time = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    username = request.values.get("UserName")
    logger.info(username)
    if not is_valid_email(username):
        return {
            'code': 403,
            'message': "邮箱格式有误！"
        }
    verify_count = session.get("verify_count") if session.get("verify_count") else 0
    verify_count += 1
    session["verify_count"] = verify_count
    if verify_count > 5:
        return {
            'code': 604,
            'message': "发送次数过多，请15分钟后再试。"
        }
    arrs = [random.randint(0, 9) for _ in range(6)]  # 随机生成6位数验证码
    verification_code = ''.join(list(map(str, arrs)))
    conn = sqlite3.connect('./database/mydata.db')
    cur = conn.cursor()
    if cur.execute("SELECT * FROM user WHERE username = ?", (username,)).fetchone():
        return {
            'code': 601,
            'message': "邮箱已被注册，请切换邮箱。"
        }
    is_exist_data = cur.execute("SELECT * FROM verify WHERE username = ?", (username,)).fetchone()
    if not is_exist_data:
        cur.execute("INSERT INTO verify VALUES (?,?,?)", (username, verification_code, now_time))
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
    logger.info(receivers[0])
    logger.info(verification_code)
    flag = 0
    try:
        sendVerifyEmail(receivers, verification_code, logger)
    except:
        flag = 1
    conn.close()
    if flag == 1:
        return {
            'code': 602,
            'message': "服务器网络不稳定！"
        }
    return {
        'code': 200,
        'message': "验证码发送成功！"
    }

@app.route('/delete', methods=['POST'])  # 删除论文
def delete():
    # 传参：文件名FileId，是否删除本地论文whether_delete_data
    now_time = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    logger.info(request.values)
    file_id = request.values.get("FileId")
    whether_delete_data = request.values.get("whether_delete_data")
    logger.info(file_id)
    logger.info(whether_delete_data)
    conn = sqlite3.connect('./database/mydata.db')
    cur = conn.cursor()
    status = cur.execute("SELECT status FROM commit_record WHERE file_id = ?", (file_id,)).fetchone()[0]
    if status != 1 and status != 2:
        return {
            'code': 401,
            'message': "论文纠错中无法删除！"
        }
    cur.execute("UPDATE commit_record SET is_delete = ? WHERE file_id = ?", (1, file_id))
    conn.commit()
    if whether_delete_data == "true":  # 删除本地论文文件和纠错数据
        file_path, json_path = cur.execute("SELECT file_path, json_path FROM commit_record WHERE file_id = ?", (file_id,)).fetchone()
        try:
            os.remove(file_path)
            os.remove(json_path)
            # todo 删除数据库相应内容
            cur.execute("DELETE FROM errors WHERE file_id = ?", (file_id,))
            conn.commit()
        except Exception as e:
            return {
                'code': 400,
                'message': "删除失败！"
            }
    conn.close()
    return {
        'code': 200,
        'message': "删除成功！"
    }


# @app.route('/mail', methods=['GET'])  # 发送邮件提醒
def send_email(receiver, username, file_id):
    # username = request.args.get('UserName')
    # file_id = request.args.get('FileId')
    conn = sqlite3.connect('./database/mydata.db')
    cur = conn.cursor()
    file_name = cur.execute("SELECT file_name FROM commit_record WHERE file_id = ?", (file_id,)).fetchone()[0]
    receivers = [receiver]
    sendEmail(receivers, username, file_name)
    conn.close()


def is_valid_email(email):  # 检验邮箱格式
    # todo 限制为学校认证的邮箱   暂定修改为edu.cn结尾
    pattern = re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]*edu.cn$')
    # pattern = re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]*$')
    return pattern.match(email)


def v_code(n=20):
    ret = ""
    for i in range(n):
        num = random.randint(0, 9)
        # num = chr(random.randint(48,57))#ASCII表示数字
        letter = chr(random.randint(97, 122))  # 取小写字母
        Letter = chr(random.randint(65, 90))  # 取大写字母
        s = str(random.choice([num, letter, Letter]))
        ret += s
    return ret

def valid_username(username):
    session_username = session.get("username")
    if username == session_username:
        print("验证成功")
        return True
    else:
        print("验证失败，非本人操作")
        return False

@app.route('/')
def hello():
    return "Hello"

@app.route('/set_session')
def set_session():
    session["username"] = "111111"
    session.permanent = True
    return "Session设置成功！"

@app.route('/get_session')
def get_session():
    username = session.get("username")
    return username or "Session为空！"

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    app.run(host='0.0.0.0', port=8080, debug=True)
