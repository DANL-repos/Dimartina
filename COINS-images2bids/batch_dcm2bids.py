#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:40:10 2018

@author: sray
"""

import os, sys
import pandas as pd
import numpy as np
from optparse import OptionParser
import argparse
import datetime

#Take input form user
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' %message)
        self.print_help()
        sys.exit(2)
        
def gracefulExit():
    parser.print_help()
    exit(2)

basePath=os.getcwd()

parser=MyParser(prog="DCM2BIDS")
parser.add_argument("--source_dir",dest="source", required=True, help='The directory containing all subject dicoms')
parser.add_argument("--bids_dir",dest="destination", required=True, help='The directory the bids data should be put in')
parser.add_argument("--COINS_BIDS",dest="COINS_BIDS", required=True, help='selected_scans.csv output from COINS_BIDS_setup')
parser.add_argument('--subject_list', dest='subject_list', required=True, help='List of subjects to convert to bids')

if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)

args=parser.parse_args()

if args.source:
    heads, tails=os.path.split(args.source)
    if heads=='':
        heads=basePath
        
    fullpaths=os.path.join(heads,tails)

if args.destination:
    headd, taild=os.path.split(args.destination)
    if headd=='':
        headd=basePath
        
    fullpathd=os.path.join(headd,taild)
    
if args.COINS_BIDS:
    headc,tailc=os.path.split(args.COINS_BIDS)
    if headc=='':
        headc=basePath
        
    fullpathc=os.path.join(headc,tailc)
    coins_bids=pd.read_csv(fullpathc)

if args.subject_list:
    f = open(args.subject_list)
    subject_list = f.read().splitlines()
    f.close()

path = fullpaths.split('/')

#subID=coins_bids['SubjectID']
subID=coins_bids['Scan_Subject_ID']
for i in range(len(subID)):  
    subid2 = 'sub-' + str(subID[i])
    if subid2 in subject_list:
        if os.path.exists(fullpaths + "/sub-"+ str(subID[i]) + "/" + str(subID[i]) + ".json"):
            cmd="dcm2bids -d " + fullpaths + "/sub-" + str(subID[i]) + " -p " + str(subID[i]) + " -c " + fullpaths + "/sub-"+ str(subID[i]) + "/" + str(subID[i]) + ".json" + " -o " + fullpathd
            os.system(cmd)
        else:
            f = open(fullpaths + '/error_log.txt', 'a')
            f.write('{} : {} : {} : {}\n'.format(datetime.datetime.now(), 'batch_dcm2bids', subID[i], 'subject not in source folder'))
            f.close()
