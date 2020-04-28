# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 03:27:53 2020

@author: Vijay Narsing Chakole

Topic : duplicate file removal

"""

import os 
from sys import *
import hashlib 
import time

# set current working directory
#os.chdir("C:/Users/hp/PycharmProjects")
os.chdir("C:/Users/hp/Desktop")

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


###############################################################################
        
def hashfile(path, blocksize = 1024):
    
    fd = open(path, 'rb')
    hasher = hashlib.md5()
    buffer = fd.read(blocksize)
    
    while len(buffer) > 0 :
        hasher.update(buffer)
        buffer = fd.read(blocksize)
        
    fd.close()
    return hasher.hexdigest()

###############################################################################

def FindDuplicateFile(path):
    
    flag = os.path.isabs(path)
    
    if flag == False:
        path = os.path.abspath(path)
        
    exists = os.path.isdir(path)
    
    duplicate = {}
    if exists :
        
        for folder,subfolder,files in os.walk(path):
            for file_name in files :
                path = os.path.join(folder, file_name)
                file_hash = hashfile(path)
                
                if file_hash in duplicate:
                    duplicate[file_hash].append(path)
                    
                else:
                    duplicate[file_hash] = [path]
                    
        return duplicate
                    
    else:
        print("invalid path")
                
###############################################################################

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
            
    
###############################################################################

def main():
    
    print("inside main")
    #path = "college"
    print("Application Name "+argv[0])
    
    string = argv[1].lower()
    
    if (len(argv)<2):
        print("Invalid Number of arguments ")
    
        
    if(string == '-h'):
        print("Script is designed for traverse specific directory and delete the duplicate file ")
        exit()
    
    if (string == '-u'):
        print("Usage : Application Name Absolutepath directory Extention")
        exit()
    
    
     
    try:
        path = argv[1]
       # path = "demo12"
        arr = {}
        start_time = time.time()
        arr = FindDuplicateFile(path)
        PrintResults(arr)
        DeleteFiles(arr)
        end_time = time.time()
        
        print(f"took {(end_time - start_time)} seconds to evaluate")
    
         
    except ValueError:
        print("invalid datatype input")
    except Exception as E :
        print("ERROR : invalid input ", E)
     
     
     
     
     
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    