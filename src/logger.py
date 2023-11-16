import logging
import os
from datetime import datetime

FileName=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
FilePath=os.path.join(os.getcwd(),"logs",FileName)
os.makedirs(FilePath,exist_ok=True)

LogFilePath=os.path.join(FilePath,FileName)

logging.basicConfig(
    filename=LogFilePath,
    format="[%(asctime)s]%(lineno)d %(name)s-%(levelname)s-%(message)s",
    level=logging.INFO
)