# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 01:32:20 2020

@author: Vijay Narsing Chakole

Topic : duplicate file detector

"""

from sys import *
import os
import hashlib

# set current working directory
os.chdir("C:/Users/hp/PycharmProjects")
#os.chdir("C:/Users/hp/Desktop")

def hashfile(path, blocksize = 1024):
    fd =open(path, 'rb')
    hasher = hashlib.md5()
    buffer = fd.read(blocksize)
    
    while len(buffer)>0:
        hasher.update(buffer)
        buffer = fd.read(blocksize)
        
    
    fd.close()    
    return hasher.hexdigest()


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
        
        

def main():
    print("inside main")
    #path = "college"
    print("Application Name "+argv[0])
    
    string = argv[1].lower()
    
    if (len(argv)<2):
        print("Invalid Number of arguments ")
    
        
    if(string == '-h'):
        print("Script is designed for traverse specific directory and display size of file ")
        exit()
    
    if (string == '-u'):
        print("Usage : Application Name Absolutepath directory Extention")
        exit()
    
                
    try:
        arr = {}
        path = argv[1]
        
        #path = "college"
        arr = FindDuplicate(path)
        PrintDuplicate(arr)
    except ValueError:
        print("Error : invalid datatype of input ")
    except Exception:
        print("Error : invalid input ")
        
    


if __name__ == "__main__":
    main()

    
        
                    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            