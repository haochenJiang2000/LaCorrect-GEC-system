import smtplib
from email.mime.text import MIMEText
#设置服务器所需信息
#163邮箱服务器地址
mail_host = 'smtp.163.com'
#163用户名
mail_user = 'nonename8'
#密码(部分邮箱为授权码)
mail_pass = 'QUZHMSBRWCABSWSR'
#邮件发送方邮箱地址
sender = 'nonename8@163.com'
# #邮件接受方邮箱地址，注意需要[]包裹，这意味着你可以写多个邮件地址群发
# receivers = ['1927405039@stu.suda.edu.cn']

def sendEmail(receivers, username, filename):
    #设置email信息
    # 邮件接受方邮箱地址，注意需要[]包裹，这意味着你可以写多个邮件地址群发)
    #邮件内容设置
    message = MIMEText(f'用户{username}您好，您提交的论文《{filename}》已经纠错成功，请您上线查看！\nhttp://192.168.134.31/', 'plain', 'utf-8')
    #邮件主题
    message['Subject'] = '纠错完成提醒'
    #发送方信息
    message['From'] = sender
    #接受方信息
    message['To'] = receivers[0]

    #登录并发送邮件
    try:
        smtpObj = smtplib.SMTP()
        #连接到服务器
        smtpObj.connect(mail_host, 25)
        #登录到服务器
        smtpObj.login(mail_user, mail_pass)
        #发送
        smtpObj.sendmail(
            sender, receivers, message.as_string())
        #退出
        smtpObj.quit()
        return
    except smtplib.SMTPException as e:
        print('error', e) #打印错误
    return

def sendVerifyEmail(receivers, verification_code, logger):
    #设置email信息
    # 邮件接受方邮箱地址，注意需要[]包裹，这意味着你可以写多个邮件地址群发)
    #邮件内容设置
    message = MIMEText(f'您好！您的注册验证码为{verification_code}。若您并未注册本系统，请忽略。', 'plain', 'utf-8')
    #邮件主题
    message['Subject'] = '注册验证'
    #发送方信息
    message['From'] = sender
    #接受方信息
    message['To'] = receivers[0]

    #登录并发送邮件
    try:
        smtpObj = smtplib.SMTP()
        #连接到服务器
        smtpObj.connect(mail_host, 25)
        #登录到服务器
        smtpObj.login(mail_user, mail_pass)
        #发送
        smtpObj.sendmail(
            sender, receivers, message.as_string())
        #退出
        smtpObj.quit()
        return
    except smtplib.SMTPException as e:
        print('error', e) #打印错误
    return
