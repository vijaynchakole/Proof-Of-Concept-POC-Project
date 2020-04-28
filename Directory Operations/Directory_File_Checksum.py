# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 01:02:26 2020

@author: Vijay Narsing Chakole


MD5 hash in Python
Cryptographic hashes are used in day-day life like in digital signatures, message authentication codes, manipulation detection, fingerprints, checksums (message integrity check), hash tables, password storage and much more. They are also used in sending messages over network for security or storing messages in databases.

There are many hash functions defined in the “hashlib” library in python.
Here we are talking about explanation and working of MD5 hash.

MD5 Hash
This hash function accepts sequence of bytes and returns 128 bit hash value, usually used to check data integrity but has security issues.

Functions associated :

encode() : Converts the string into bytes to be acceptable by hash function.
digest() : Returns the encoded data in byte format.
hexdigest() : Returns the encoded data in hexadecimal format.

Explanation : The below code takes file data string and
converts it into the byte equivalent using encode() 
so that it can be accepted by the hash function. 
The md5 hash function encodes it and then using hexdigest(), 
hexadecimal equivalent encoded string is printed.

"""

from sys import *
import os
import hashlib

# set current working directory
os.chdir("C:/Users/hp/PycharmProjects")


def hashfile(file_path, blocksize = 1024):
    afile = open(file_path, 'rb')
    hasher = hashlib.md5()
    buffer = afile.read(blocksize)
    while(len(buffer)>0):
        hasher.update(buffer)
        buffer = afile.read(blocksize)
    
    afile.close()
    return hasher.hexdigest()


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
        
        
        
#path = "college"
#DisplayChecksum(path)    


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
        DisplayChecksum(path)
    except ValueError:
        print("Error : invalid datatype of input ")
    except Exception:
        print("Error : invalid input ")
        
    


if __name__ == "__main__":
    main()
