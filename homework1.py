#!/usr/bin/env python
import argparse

import cv2
from hw1_tools import *

'''
My classmate kindly shared args parser with me.
So this parser mostly written not by me
And most of its options aren`t implemented at moment.
Those which are implemented are marked by 'implemented'.
'''

parser = argparse.ArgumentParser()

parser.add_argument("-corr_linear",'--corr_linear', nargs = 2)#implemented
parser.add_argument("-corr_nonlinear",'--corr_nonlinear', nargs = 3)
parser.add_argument("-invert",'--invert', nargs = 2)
parser.add_argument("-bw",'--bw', nargs = 2)#implemented
parser.add_argument("-wb_ww",'--wb_ww', nargs = 2)
parser.add_argument("-wb_gw",'--wb_gw', nargs = 2)
parser.add_argument("-gauss",'--gauss', nargs = 3)
parser.add_argument("-box",'--box', nargs = 3)#implemented.
parser.add_argument("-median",'--median', nargs = 3)
parser.add_argument("-crop",'--crop', nargs = 6)#implemented
args = parser.parse_args()
#print(args)
if args.corr_linear:#implemented
    img=cv2.imread(args.corr_linear[-2])
    result=linear_correction(img)
    cv2.imwrite(args.corr_linear[-1],result)
    
if args.corr_nonlinear:#implemented
    img=cv2.imread(args.corr_nonlinear[-2])
    gamma=float(args.corr_nonlinear[0])
    result = nonlinear_corr(img,gamma)
    cv2.imwrite(args.corr_nonlinear[-1],result)
    
if args.invert:#implemented
    img=cv2.imread(args.invert[-2])
    result=invert(img)
    cv2.imwrite(args.invert[-1],result)
    
if args.bw:#implemented
    img=cv2.imread(args.bw[-2])
    result=to_grayscale(img)
    cv2.imwrite(args.bw[-1],result)
    
if args.wb_ww:#implemented
    img=cv2.imread(args.wb_ww[-2])
    result=WB(img,mode='ww')
    cv2.imwrite(args.wb_ww[-1],result)
    
if args.wb_gw:#implemented
    img=cv2.imread(args.wb_gw[-2])
    result=WB(img,mode='gw')
    cv2.imwrite(args.wb_gw[-1],result)
    
if args.box:#implemented
    img=cv2.imread(args.box[-2])
    result=blur(img,kernel_side=int(args.box[0]))
    cv2.imwrite(args.box[-1],result)
    
if args.gauss:#implemented
    img=cv2.imread(args.gauss[-2])
    result=blur(img,mode='gauss',gamma=int(args.gauss[0]))
    cv2.imwrite(args.gauss[-1],result)
    
if args.median:#implemented
    img=cv2.imread(args.median[-2])
    result=blur(img,kernel_side=int(args.median[0]),mode='median')
    cv2.imwrite(args.median[-1], result)
    
if args.crop:#implemented
    img=cv2.imread(args.crop[-2])
    l,t,w,h=map(int,(args.crop[0],args.crop[1],args.crop[2],args.crop[3]))
    top_left_point = (t,l)
    size = (h,w)
    result=crop(img,top_left_point,size)
    cv2.imwrite(args.crop[-1],result)
