import sys
from src.logger import logging

class CustomeException(Exception):
      def __init__(self,errormessage,errordetail:sys):
        
        super().__init__(errormessage)
        self.errormessage=errormessage_details(errormessage,errordetail=errordetail)
      def __str__(self):
        return self.errormessage


def errormessage_details(errormessage,errordetail:sys):
    _,_,exc_tb=errordetail.exc_info()
    fileName =exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in the file [{0}] at line number[{1}] error message [{2}]".format(
        fileName,exc_tb.tb_lineno,str(errormessage))
    return error_message
