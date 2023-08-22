import numpy as np




def get_value(line):
    value=""
    
    for c in line:
        if c.isdigit() or c==".":
            value += c
        else:
            break
    return value

def get_factors(f):

    file=open(f,"r")

    
    str1="left sideband edge"
    str2="right sideband edge"
    str3="background scaling factor"
    str4="signal scaling factor "

    for line in file:
        if str1 in line:
            left_edge=get_value(line)
        elif str2 in line:
            right_edge=get_value(line)
        elif str3 in line:
            fb=float(get_value(line))
        elif str4 in line:
            fs=float(get_value(line))

    return left_edge,right_edge,fb,fs