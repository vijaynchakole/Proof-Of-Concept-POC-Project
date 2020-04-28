# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:30:14 2020

@author: Vijay Narsing Chakole

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
import hashlib



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


###########################################################################
# calculate checkSum of  given file
def hashfile(file_path, blocksize = 1024):
    afile = open(file_path, 'rb')
    hasher = hashlib.md5()
    buffer = afile.read(blocksize)
    while(len(buffer)>0):
        hasher.update(buffer)
        buffer = afile.read(blocksize)
    
    afile.close()
    return hasher.hexdigest()

#########################################################################

# display CheckSum of file
def DisplayChecksum(path):
    flag = os.path.isabs(path)
    
    if flag == False :
        path = os.path.abspath(path)
    
    
    exists = os.path.isdir(path)
    
    if exists:
        for folder, subfolder,file in os.walk(path):
            print("Current folder is "+ folder)
            for filename in file :
                path = os.path.join(folder, filename)
                file_hash = hashfile(path)
                print(path)
                print(file_hash)
                print(" ")
    else:
        print("Invalid Path")

##########################################################################################

# Find duplicate file in given directory

def FindDuplicate(path):
    flag = os.path.isabs(path)
    
    if flag == False :
        path = os.path.abspath(path)
        
    
    exists = os.path.isdir(path)
    
    duplicate = {}
    
    if exists: 
        for folder, subfolder, filenames in os.walk(path):
            for file in filenames :
                path = os.path.join(folder, file)
                hash_file = hashfile(path)
                if hash_file in duplicate :
                    duplicate[hash_file].append(path)
                else:
                    duplicate[hash_file] = [path]
        return duplicate                          
    else:
        print("invalid path")
        

###########################################################

# print duplicate file names

def PrintDuplicate(dict1):
    results = list(filter(lambda x : len(x)>1, dict1.values()))
    type(results) #list of list containing duplicate file names if found duplicate
    if len(results) > 0 :
        print("Duplicate Found")
        print("Following Files are identical ")
        icnt = 0
        for result in results :
           # print(result)
            #print(icnt)
           # icnt = 0
            for subresult in result:    #result[-1] original file name
                icnt = icnt +1
                #print(icnt)
                if icnt >= 2 :
                    print(icnt,"\t\t%s"%subresult)
                    
    else:
        print("NO duplicate file Found")
        

##########################################################


def DeleteFiles(dict1):
    results = list(filter(lambda x : len(x)>1, dict1.values()))
    
    icnt = 0
    
    if len(results) > 0 :
        print("duplicate file found")
        
        for result in results:
            for subresult in result :
                icnt = icnt + 1
                if icnt>=2:
                    os.remove(subresult)
            icnt = 0
            
    else:
        print("Duplicate file not found")


###############################################################

def PrintResults(dict1):
    results = list(filter(lambda x: len(x)>1 , dict1))
    
    if len(results)>0:
        print("duplicate file found")
        
        print("followings are duplicate files")
        
        for result in results :
            for subresult in result:
               # print(subresult)
                print("\t\t%s"%subresult)
               # print("\t\t%s" % subresult)
    
        
    else:
        print("No Duplicate file found")




def main():
    print("inside main")
    #path = "college"
    print("Application Name : "+argv[0])
    
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
        #DirectoryWatcher(path)
        #DisplayChecksum(path)
        
        arr = FindDuplicate(path)
        PrintDuplicate(arr)
        
        # find duplicate file and remove it 
        #path = "libraries"
        arr = FindDuplicate(path)
        PrintResults(arr)
        DeleteFiles(arr)
        
    except ValueError:
        print("Error : invalid datatype of input ")
    except Exception:
        print("Error : invalid input ")
        
    

if __name__ == "__main__":
    main()











