import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import html5lib
import yfinance as yf

'''
names=[]
startURL='https://finrange.com/ru/company/NASDAQ?page='
for i in range(1,16):
    URL=startURL+str(i)
    response=requests.get(URL)
    #print(response.status_code)
    soup=BeautifulSoup(response.content,"html5lib")
    #print(page)
    items=soup.findAll('span','ticker')
    for item in items:
        names.append(item.text)
print(len(names))
#print(names)
'''

'''
with open('names.txt', 'w'):
    pass
    
with open('names.txt', 'w') as fp:
    fp.write(' '.join(names))
'''


