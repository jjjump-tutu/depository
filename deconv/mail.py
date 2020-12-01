import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import re

def mail_it(email_address, subject, text):
    email_pattern = r'^[_A-Za-z0-9\u4e00-\u9fa5]+@[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+)+$'
    if not re.match(email_pattern, email_address):
        print("Invalid email address!")
        return
    sender = 'jiang_x@qq.com'
    receivers = [email_address,]
    message = MIMEMultipart()
    message['From'] = Header("JiangXue", 'utf-8')
    message['To'] = Header("TUTU", 'utf-8')
    message['Subject'] = Header(subject, 'utf-8')

    message.attach(MIMEText(text, 'plain', 'utf-8'))

    try:
        server=smtplib.SMTP_SSL("smtp.qq.com", 465)
        server.login(sender, "hrqrvocxospfbbaf")
        server.sendmail(sender, receivers, message.as_string())
        server.quit()
    except smtplib.SMTPException:
        print("An error occurred while sending email.")

if __name__ == "__main__":
    mail_it("jiang_x@qq.com","Test Mail","Test!")
