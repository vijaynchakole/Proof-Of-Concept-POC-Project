# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:30:14 2020

@author: Vijay Narsing Chakole

Topic : directory watcher

Absolute and Relative Pathnames in UNIX
A path is a unique location to a file or a folder in a file system of an OS.A path to a file is a combination of / and alpha-numeric characters.

Absolute Path-name

An absolute path is defined as the specifying the location of a file or directory from the root directory(/).
To write an absolute path-name:



Start at the root directory ( / ) and work down.
Write a slash ( / ) after every directory name (last one is optional)


An absolute path is defined as specifying the location of a file or directory from the root directory(/).
In other words,we can say that an absolute path is a complete path from start of actual file system from / directory.


*******************************************************************************
Relative path

Relative path is defined as the path related to the present working directly(pwd).
It starts at your current directory and never starts with a / .
 

Topic :
    

 
"""

from sys import *
import os



# set current working to "C:\\Users\\hp\\Desktop\\practicals_ml" and 
# then it will find given folder name in this directory
#if given folder is exists then it will show all file names
# otherwise throws exception
# very important line
# you must give folder name which are already exists in below mentioned directory path

# set current working directory
os.chdir("C:\\Users\\hp\\Desktop\\practicals_ml")


# shows subfolder names and file names of given directory
def DirectoryWatcher(path):
    flag = os.path.isabs(path)

    if flag == False:
        path = os.path.abspath(path)

    exists = os.path.isdir(path)

    if exists:
        for foldername, subfolder, filename in os.walk(path):
            print("Current folder is ", foldername )
            for subfold in subfolder :
                print("sub Folder of " + foldername +" is "+ subfold)
            for file_name in filename:
                print("File inside "+foldername+" is "+file_name)
            print('')
    else:
        print("Directory Not Found ")



def main():
    print("inside main")
    #path = "college"
    print("Application Name "+argv[0])
    
    string = argv[1].lower()
    
    if (len(argv)<2):
        print("Invalid Number of arguments ")
    
        
    if(string == '-h'):
        print("Script is designed for traverse specific directory ")
        exit()
    
    if (string == '-u'):
        print("Usage : Application Name Absolutepath directory")
        exit()
    
                
    try:
        path = argv[1]
        DirectoryWatcher(path)
    except ValueError:
        print("Error : invalid datatype of input ")
    except Exception:
        print("Error : invalid input ")
        
    

if __name__ == "__main__":
    main()











