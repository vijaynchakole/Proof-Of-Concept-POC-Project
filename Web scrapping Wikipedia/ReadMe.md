@author: Vijay Narsing Chakole 

Topic : Wikipedia Web scrapping 

Automation script which fetch URL, title and
content headers of wikipedia page using Beautiful Soup.

Description :



# Project WorkFlow

file Name : Web_Scrapping_Wkipidea.py

code Explanation :

### libraries used in this project:

1). requests

2). bs4 (beautifulSoup)

### data structure used in this project

1). link_ls = it is list which is used to store url's

2). titles_ls = it is list which is used to store title

3). headers_ls = it is list which is used to store header



#### Functions Name and their working

1). Function Name : LinksDisplay()

    input : URL
    
    Working : open url with the help of requests module and bring html content and 
    extract all links from html page with the help of bs4 module 



2). Function Name : TitlenHeaderDisplay()
  
    input : URL
    
    Working : open url with the help of requests module and bring html content and 
    extract all headers and title from html page with the help of bs4 module 


3). Function Name : main()
  
  working : calling above mentioned functions
  
  Note : entry point function
