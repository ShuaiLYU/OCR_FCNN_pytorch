import logging
import os
import time
import sys
#打印日志到控制台和log_path下的txt文件
def get_logger():
    log_path='log_path'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    timer=time.strftime("%Y-%m-%d-%H-%M-%S_", time.localtime())
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    txthandle = logging.FileHandler((log_path+'/'+timer+'log.txt'))
    txthandle.setFormatter(formatter)
    logger.addHandler(txthandle)
    #输出到屏幕
    console_handle=logging.StreamHandler()
    console_handle.setLevel(logging.INFO)
    logger.addHandler(console_handle)
    return logger

#将输入路径的上两级路径加入系统
def set_projectpath(current_path):
    curPath = os.path.abspath(current_path)
    #curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = os.path.split(curPath)[0]
    sys.path.append(rootPath)
    rootPath = os.path.split(rootPath)[0]
    sys.path.append(rootPath)
