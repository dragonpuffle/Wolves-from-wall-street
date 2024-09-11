import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import html5lib
import yfinance as yf
import openpyxl

'''
#parcing data
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
#writing data in file
with open('names.txt', 'w'):
    pass
    
with open('names.txt', 'w') as fp:
    fp.write(' '.join(names))
'''

'''
#reading data from file
names=[]
with open('names.txt') as file:
    names= file.read().split(' ')
#print(names)

#downloading data from yah00
data=pd.DataFrame(columns=names)
for name in names:
    data[name]=yf.download(name,'2016-01-01','2016-12-31')['Adj Close']

#transfer data to input.txt
data.to_excel('input.xlsx', index=False)
'''

