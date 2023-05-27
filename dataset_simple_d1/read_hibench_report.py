#-*-coding:utf-8 -*-
import json
import sys
from pandas import json_normalize


#report_file = open("/home/love1245672/桌面/大數據基準測試/HiBench-HiBench-7.0/report/hibench.report", 'r')


import csv
with open("hibench_report.csv", 'w') as csvfile:
    spamwriter = csv.writer(csvfile, dialect='excel')
    # 讀要轉換的txt檔案，檔案每行各詞間以@@@字元分隔
    with open('/home/love1245672/桌面/hibench_report/dataset_simple/hibench_simple_final.report', 'r') as filein:
        for line in filein:
            line_list =line.strip('\n').split(' ')
            line_list = [x for x in line_list if x]
            spamwriter.writerow(line_list)
 

