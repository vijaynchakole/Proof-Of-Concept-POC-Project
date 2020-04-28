# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:20:33 2020

@author: Vijay Narsing Chakole 

Topic : Wikipedia Web scraping 

Automation script which fetch URL, title and
content headers of wikipedia page using Beautiful Soup.

"""

import requests
import bs4

# create data structure
link_ls = []
titles_ls = []
headers_ls = []

def LinksDisplay(url):
    res = requests.get(url)
    res.text
    
    page_soup = bs4.BeautifulSoup(res.text, "lxml")
    links = page_soup.find_all('a', {"class" : "external text"})
    
    return links
    
    

def TitlenHeaderDisplay(url):
    res = requests.get(url)
    page_soup = bs4.BeautifulSoup(res.text, "lxml")
    
    titles = page_soup.select("title")
    for title in titles:
        print(title.text)
        titles_ls.append(title.text)
        
    header = page_soup.find_all("span", {"class" : "toctext"})
    for head in header :
        print(head.text)
        headers_ls.append(head.text)


def main():
    url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    
    links_arr = LinksDisplay(url)
    
    for link in links_arr :
        print(link['href'])
        link_ls.append(link['href'])
        
        
if __name__ == "__main__":
    main()
