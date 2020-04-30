##Project - Crawl popular websites and create a database of Indian movie celebrities containing their images and personality traits.

plan of attack :

I have selected website https://www.imdb.com/list/ls002913270/ for web scrapping. 
IMDB website provides top 100 indian celebraties list with their best movie work, images and some personal information.

I have scrapped all  information ralated celebraties. 
list of scrapped information as per below

1) Celebraty Name :
2) Celebraty image :
3) profession :
4) best work movie :
5) personal information :

For Example :

1) Celebraty Name : Shah Rukh Khan

2) Celebraty image : https://m.media-amazon.com/images/M/MV5BZDk1ZmU0NGYtMzQ2Yi00N2NjLTkyNWEtZWE2NTU4NTJiZGUzXkEyXkFqcGdeQXVyMTExNDQ2MTI@._V1_UY209_CR3,0,140,209_AL_.jpg

3) profession : Actor

4) best work movie : Don 2

5) personal information : Shahrukh Khan was born on 2 November 1965 in New Delhi, India. He married Gauri Khan on 25 October 1991. They have three children, son Aryan Khan (b. 1997), son AbRam (b.2013) and daughter Suhana (b. 2000). Khan started out his career by appearing in several television serials during 1988-1990. ...

By the same way I have extracting top 100 indian celebraties information from IMDB website :

****************************************************************************************************************************************

Code Explaination :

file Name : indian_celebraties_info.py

# required libraries :

os : for file system related operation

urllib : for opening website url

bs4 : for extracting data from html page

wget : for downloading images

mysql : for creating database in mysql workBench and inserting data into it.

****************************************************************************************************************************************

# data structure

image_ls : it is list which contains celebraties image url

name_ls : it is list which contains celebraties names

movie_name : it is list which contains best movie name 

profession : it is list which contains celebraties's profession

paragraph_ls : it is list which contains personal information of celebraties


****************************************************************************************************************************************
# functional approach

# function Name  and their working
1)
Function Name : get_information()

input : website url

working : it takes url and open webpage with the help of urlopen library. once url opens it brings data in html format.
then with the help of BeautifulSoup from bs4 library we have read data and extract important information from raw html page.
extracted information is : celebraties's name , their movie name, their profession , image url, personal info.

2) 
Function Name : partition()

input : text data

working : it takes text data and separate out profession and movie name in separate list.

3)
Function Name : download_images()

input :image url list

working : it downloads images from provided url with the help of wget library

4)

Function Name : create_folder()

input :image folder name

working : it creates new folder if it is not exists in current working directory for saving downloaded images. 

output : folder name


5)
Function Name : remove_img()

input :image folder name

working : it delete images from provided image folder.

Note: if we want to delete downlaoded images  then only use this function.


6)
Function Name : create_database_connection()

input : -

working : it creates database connection with the help of mysql library.

output : returns connection variable (conn)


7)
Function Name : InsertVariablesIntoTable()

input : celebratiy name , gender, profession, movie, image, details

working : it insert data into celebraty_information table in indian_celebraties database.

8)
Function Name : main()

working : calling all above mentioned functions 

Note : it is function which is invokes first by PVM







