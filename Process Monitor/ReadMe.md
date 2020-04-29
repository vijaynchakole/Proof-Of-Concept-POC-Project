### Project : Process Monitor

Automation script which accepts time interval from user and create log file in that
Log Monitoring folder directory which contains information of all running processes.
After creating the log file send that log file through mail.


#### Project file :  Process_Monitor_With_Periodic_Memory_Log_and_Mail_Sender.py

#### code Explanation :

##### dependencoes : 

**psutil** :  It is useful mainly for system monitoring, profiling and limiting process resources and management of running processes.

psutil (process and system utilities) is a cross-platform library for retrieving information on running processes and system utilization (CPU, memory, disks, network, sensors) in Python


**os** : file system related operations 

This module provides a portable way of using operating system dependent functionality.

**time** : This module provides various time-related functions. For related functionality .


**urllib** : for opening urls


**sys** : for commandline argument

**smtplib** : The smtplib module defines an SMTP client session object that can be used to send mail to any Internet machine with an SMTP or ESMTP listener daemon.

**schedule** : Schedule Library is used to schedule a task at a particular time every day or a particular day of a week. We can also set time in 24 hours format that when a task should run.

**email** : to read, write, and send simple email messages, as well as more complex  MIME (Multipurpose Internet Mail Extensions) messages.


### **Function Names and their working** :

1). **Function Name** : *is_connected( )*

Working : check internet connection


2).**Function Name** : MailSender()

input : log file name, time interval

working : attached log file and send mail

3). **Function Name** : *ProcessLog( )*

input : folder name

working : create log file and store running process information in that log file.

4). **Function Name** : *main( )* 

working : calling above mentioned function

Note : entry point function
