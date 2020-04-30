# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 07:11:20 2020

@author: Vijay Narsing Chakole

problem statement : 
Crawl popular websites & create a database of Indian movie celebrities
containing their images and personality traits.

"""
# import required libraries
import os
from urllib.request import urlopen
import bs4
import wget
import mysql.connector
from mysql.connector import Error

#set current working directory
os.chdir("C:\\Users\\hp\\Desktop\\Bipolar Factory Project\\")
#get current working directory
current_dir = os.getcwd()

# creating data structures    
image_ls = []
name_ls = []
movie_name = []
profession = []
paragraph_ls = []
gender = ""
conn = None


def get_information(url): 
    connection = urlopen(url)
    html = connection.read()
    soup = bs4.BeautifulSoup(html, "html.parser")
    # extracting class which contain images
    image_container = soup.findAll("div",{"class":"lister-item-image"})
    
    for image in image_container :
        image_ls.append(image.img["src"])
        name_ls.append(image.img["alt"])
            
    # extracting class which contains information about celebraties
    data_container = soup.findAll("div", {"class":"lister-item-content"})
    
    for info in data_container :
        paragraph_ls.append( (info.find("p", class_="").get_text()).strip() )
        partition(info.find("p", class_="text-muted text-small").get_text())


def partition(text):
    # partition text so that we can get profession and movie name separetly
    target_text = text.partition("|")  
    # adding to list (and removing space by using strip())
    profession.append(target_text[0].strip())
    movie_name.append(target_text[2].strip())


def download_images(image_ls):
    for image_url in image_ls:
        wget.download(image_url)
    

def create_folder(folder):
    if os.path.exists(folder):
        return folder
    else:
        os.mkdir(folder)
        return folder
    
        
def remove_img(folder):    
    path = os.path.join(current_dir,folder)   
    os.chdir(path) 
    [os.remove(file) for file in os.listdir(path) if file.endswith(('.jpg', '.png'))]
    os.chdir(current_dir) 


def create_database_connection():
    global conn
    
    try:
        conn= mysql.connector.connect(host = "localhost", database = "indian_celebraties",
                                             user = "root",
                                             password = "shreeRam")    
        
        if conn.is_connected():        
            db_Info = conn.get_server_info()
            print(f"Connected to sql server version : {db_Info}")
            
            cursor = conn.cursor()
            cursor.execute("select database();")
        
            record = cursor.fetchone()
            print(f"you are connected to database {record}")
            
            return conn
            
    except Error as err :
        print("Error while connecting to MySQL", err)  
 
    
 
def InsertVariablesIntoTable(name, gender, profession, movie, image,details):
    global conn
    
    try:    
        if conn.is_connected():    
            cursor = conn.cursor()
            
            query = """insert into celebraty_information
                        (`celebraty_Name`, `Gender`, `profession`, `best_movie`, `image`, `details`)
                       values(%s, %s, %s, %s, %s, %s);"""
            
            record_tuple = (name, gender, profession, movie, image, details)
            cursor.execute(query, record_tuple)
            
            #  We also used commit() method to make changes persistent in the database.
            conn.commit()        
            print(cursor.rowcount, "Record inserted successfully into celebraty_information table")
    
    except Error as err :
        print("Error while connecting to MySQL", err)
    
    finally:
        if conn.is_connected():
            cursor.close()
           
        
def main():
    print("inside main")
    
    url = "https://www.imdb.com/list/ls002913270/"
    
    # calling function for data extractiion from url
    get_information(url)
    
    # creating folder for saving images
    image_folder = create_folder("photos")
    
    #set working directory new created folder for saving images
    os.chdir(image_folder)
    
    #downloading images from url
    download_images(image_ls)
    
    #set working directory to its original place
    os.chdir(current_dir)
    
    # if we want to delete all downloaded images
    #remove_img(image_folder)
    
    # creating database connection (indian celebraties database)
    conn = create_database_connection()

    # inserting data into celebraty information table
    for i in range(len(image_ls)):
        if profession[i] == "Actor" :
            gender = "M"
        else:
            gender = "F"
        
        InsertVariablesIntoTable(name_ls[i], gender, profession[i], movie_name[i], image_ls[i],paragraph_ls[i])

    print("successfully done...!!")


if __name__=="__main__":
    main()
