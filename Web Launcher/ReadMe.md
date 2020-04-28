## Project : Web Launcher
  Topic : Automation script which accept file name. 
Extract all URL’s from that file and connect to
that URL’s through Webbrowser.

#### Project file name : Web_Launcher.py

code Explanation :

first step : create new folder WebLauncher and save the both files ( Web_Launcher.py and Website_names.txt ) in that folder.

second step : check your internet connection .

connect your laptop to internet


#### How to execute the code : 

open commandPrompt then change working dirctory to WebLauncher folder in which these both files are saved.

for window users ,

command for changing directory : cd folder_path

once you the set working directory to that folder then type below mentioned command on your commandPrompt:

command :  *python Web_Launcher.py Website_names.txt*

#### output : opens all urls  which are in Website_names.txt file in your default webbrowser in separate tabs.

#### Libraries used in this project :


1). **urllib** : for opening provided url

2). **re** : regular expression : for pattern match , finding url pattern string in given file 

3). **webbrowser** : for opening fond url in web-browser

4). **os** : for file system related operations

5). **sys** : for commandline argument 


#### Function Names and their working :

1). **Function Name** : *is_connected( )* :

working : check whether your PC is connected to internet or not. in short , it's checks internet connectivity

2). **Function Name** : *Find( )*

working : it finds url pattern in given text file and return found url string to function *WebLauncher( )* 

3). **Function Name** : *WebLauncher( )*

Input : folder name

working : it takes file name which contains urls and then it opens file and passes line by line string to function *Find( )* which
finds url pattern in that string with the help of re module( regular expression ) and then return url to  *WebLauncher( )* .
*WebLauncher( )* function takes that url and open it in webbroswer by using webbroswer library

4). **Function Name**  : *Main( )*

working : calling *WebLauncher( )* function

Note : Entry Point function


##### Contact Person :

feel free to contact me if you have any doubt or query related to project code : vijaychakole23@gmail.com
