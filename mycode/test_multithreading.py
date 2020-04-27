# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 21:23:31 2020

@author: bejin
"""

import logging
import threading
import time
import numpy as np

df = np.random.rand(16384, 4)

start = time.time()
df1 = df.copy()
duration = time.time() - start
print(duration)



data = [1]
flag_st = False

def rt_data():
    while data[0] <= 20 and not flag_st:
        time.sleep(0.7)
        data[0] = data[0] + 1
        #yield data

def thread_function(name):
    logging.info("Thread %s: starting", name)
    time.sleep(2)
    logging.info("Thread %s: finishing", name)

if __name__ == "__main__":
    '''
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    logging.info("Main    : before creating thread")
    x = threading.Thread(target=thread_function, args=(1,))
    logging.info("Main    : before running thread")
    x.start()
    logging.info("Main    : wait for the thread to finish")
    x.join()
    logging.info("Main    : all done")
    '''
    
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    x = threading.Thread(target=rt_data)
    x.start()
    
    for i in range(10):
        y = threading.Thread(target=thread_function, args=(data[0],))
        y.start()
        #y.join()
    logging.info("finish processing")
    x.join()
    logging.info("main   : all done")
    
    
    