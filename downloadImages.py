# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 20:54:00 2020

@author: kjosh
"""

from selenium import webdriver
import time
import urllib.request
import os
from selenium.webdriver.common.keys import Keys


browser = webdriver.Chrome('C:\\Users\\kjosh\\Downloads\\chromedriver_win32\\chromedriver.exe') #incase you are chrome
browser.get('https://www.google.com/')


search = browser.find_element_by_name('q')


search.send_keys('neck gaiter mask multicolor',Keys.ENTER)


elem = browser.find_element_by_link_text('Images')
elem.get_attribute('href')
elem.click()



value = 0
for i in range(20):
   browser.execute_script('scrollBy('+ str(value) +',+1000);')
   value += 1000
   time.sleep(3)
   
   
elem1 = browser.find_element_by_id('islmp')

sub = elem1.find_elements_by_tag_name('img')

try:
    os.mkdir('multicolor')
except FileExistsError:
    pass


count = 0
for i in sub:
    src = i.get_attribute('src')
    try:
        if src != None:
            src  = str(src)
            print(src)
            count+=1
            urllib.request.urlretrieve(src, os.path.join('multicolor','image'+str(count)+'.jpg'))
        else:
            raise TypeError
    except TypeError:
        print('fail')