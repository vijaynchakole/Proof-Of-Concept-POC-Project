# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 06:28:10 2020

@author: Vijay Narsing Chakole

Topic : Process Monitor 

Automation script which accepts time interval from user and create log file in that
Marvellous directory which contains information of all running processes.
After creating the log file send that log file through mail.
    
    
"""

import os
import time
import psutil
from urllib.request import urlopen
import smtplib
import schedule
from sys import *
from email import encoders
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart

os.chdir("C:\\Users\\hp\\Desktop\\practicals_ml\\proof of concept (POC) project\\Process Monitor")
current_dir = os.getcwd()


def is_connected():
    try:
        connected = urlopen("https://www.google.com/", timeout=1)
        return True
    except URLError as err:
        return False

def MailSender(filename, time):
    try:
        fromaddr = "vijaychakole23@gmail.com"
        toaddr = "vijaynarsingchakole10@gmail.com"
        
        msg = MIMEMultipart()
        
        msg['From'] = fromaddr
        msg['To'] = toaddr
        
        body = """
        
        Hello %s,
        Welcome to Process Monitoring information
        Please find attached document which contains Log of Running process.
        Log file is created at : %s
         
        This is auto generated mail.
         
        Thanks and Regards,
        Vijay Narsing Chakole
        """%(toaddr,time)

        Subject = """
        Process Log Generated at : %s
         """%(time)
        
        msg['Sunbect'] = Subject
        msg.attach(MIMEText(body, 'pain'))
        attachment = open(filename, 'rb')
        p = MIMEBase('application', 'octet-stream')
        p.set_payload((attachment).read())
        encoders.encode_base64(p)
        
        p.add_header('Content-Disposition',"attachment; filename = %s"%filename)
        msg.attach(p)
        
        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        s.login(fromaddr, "BigRevenge@2020")
        text = msg.as_string()
        s.sendmail(fromaddr, toaddr, text)
        s.quit()
        
        print("Log file Successfully sent through Mail")
        
    except Exception as E:
        print("Unable to send mail", E)
        
        
        
def ProcessLog(log_dir = "Log Monitoring folder"):
    listprocess = []
    
    if not os.path.exists(log_dir):
        try:
            os.mkdir(log_dir)
        except:
            pass
        
    os.chdir((os.path.join(current_dir, log_dir)))

    separator = "-"*80
    #log_path = os.path.join(log_dir,"shreeLog%s.txt"%(time.ctime()))
    log_path = "Log_File.txt"
    fd = open(log_path,'w')
    fd.write(separator+"\n")
    fd.write("Process Logger : "+time.ctime() + "\n")
    fd.write(separator+"\n")
    fd.write("\n")       
        
    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=['pid','name','username'])
            vms = proc.memory_info().vms / (1024 * 1024)
            pinfo['vms'] = vms
            listprocess.append(pinfo)

        except (psutil.NoSuchProcess,psutil.AccessDenied,psutil.ZombieProcess):
            pass

    for elements in listprocess:
        fd.write("%s\n"%elements)

    print("Log file successfully generated at location %s"%(log_path))
    
    os.chdir(current_dir)
    
    connected = is_connected()

    if connected:
        startTime = time.time()
        MailSender(log_path,time.ctime())
        endTime = time.time()
        print("Log_File sent Successfully")
        print("Took %s seconds to send mail"%(endTime - startTime))
    else:
        print("There is no internet connection")       
       
        

def main():
    # input aurgument : provide time in minute (integer) 
    # for example, 1
    print("inside main \n")
    
    print("Application Name :",argv[0])
    
    try:
        schedule.every(int(argv[1])).minutes.do(ProcessLog)
        while True:
            schedule.run_pending()
            time.sleep(1)

    except ValueError :
        print("Error : invalid datatype of input")

    except Exception as E :
        print("Error : invalid input ",E)


if __name__ == "__main__":
    main()

       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
         
    
        