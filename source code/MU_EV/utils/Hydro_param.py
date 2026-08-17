# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 14:57:35 2023
This is used to calculate all the hydrodynamic parameters for different 
@author: dongh
"""

# import os,random,sys,copy
# import imageio
import numpy as np
from MU_EV.config import cfg
rho = 1000.0*cfg.EVO.SCALING
thick = 7.5E-3
comp_ratio = 1.0

def head_reactive(link_size):
    x,_,z = link_size
    M_0     = np.pi*pow(z,2)/4 *rho
    M_total = M_0*x/3
    M_s     = M_0*pow(x,2)/4
    return [M_total,M_s,0,M_0]

def head_resistive(link_size):
    x,_,z = link_size
    IH0 = z*x/2
    IH1 = z*pow(x,2)/3
    IH2 = z*pow(x,3)/4
    IH3 = z*pow(x,4)/5
    # P = np.pi*(x+z)
    IP  = np.pi*(z/2+thick)*x
    return [IH0,IH1,IH2,IH3,IP]

def body_reactive(link_size):
    x,_,z = link_size
    M_0     = np.pi*pow(z,2)/4 *rho
    M_total = M_0*x
    M_s     = M_0*pow(x,2)/2
    # M_inertia = M_0*pow(x,3)/3*rho
    return [M_total,M_s,M_0,M_0]

def body_resistive(link_size):
    x,y,z = link_size
    IH0 = z*x
    IH1 = z*pow(x,2)/2
    IH2 = z*pow(x,3)/3
    IH3 = z*pow(x,4)/4
    P = np.pi*(thick+z)
    IP  = P*x
    return [IH0,IH1,IH2,IH3,IP]

def tailt_reactive(name,pd,link_size):
    x,_,z = link_size
    if name[:7]=='tailtri':
        z = z/2
    M_0     =  np.pi*pow(pd,2)/4 *rho
    M_l     =  np.pi*pow(z, 2)/4 *rho
    M_total = (np.pi*x/12)*(pow(pd,2)+pd*z+pow(z,2)) *rho
    M_s     = (np.pi*pow(x,2)/48)*(pow(pd,2)+2*pd*z+3*pow(z,2)) *rho
    return [M_total,M_s,M_0,M_l]

def tailt_resistive(name,pd,link_size):
    x,y,z = link_size
    if name[:7]=='tailtri':
        z = 0
    IH0 = x   *(pd+z)/2
    IH1 = pow(x,2)*(pd/6+z/3)
    IH2 = pow(x,3)*(pd/12+z/4)
    IH3 = pow(x,4)*(pd/20+z/5)
    IP  = IH0*2
    return [IH0,IH1,IH2,IH3,IP]

def head_comp(x,z):
    # M_0 = np.pi*pow(z,2)/4 *comp_ratio
    M_total   = (np.pi*pow(z,2)*x/12)                                * rho * comp_ratio
    M_inertia = (np.pi*pow(z,2)*x/12)*(3*pow(z,2)/80+23/20*pow(x,2)) * rho * comp_ratio
    return [M_total,M_inertia]

def body_comp(x,z):
    M_0 = np.pi*pow(z,2)/4 *comp_ratio *rho
    M_total = M_0*x
    M_inertia = M_0*pow(x,3)/3
    return [M_total,M_inertia]

def tailt_comp(pd,x,z,tri_type=False):
    if tri_type:
        z = z/2
    # M_0 = np.pi*pow(pd,2)/4
    M_total = np.pi/12*x*(pow(pd,2)+pd*z+pow(z,2)) * rho * comp_ratio
    M_ine1 = 3/80 * (pow(x,2)*((pow(pd,3)-pow(z,3))) + (pow(pd,5)-pow(z,5))/4)/(pow(pd,3)-pow(z,3))
    M_ine2 = pow((z*(pow(pd,2)+2*pow(z,2)))/(4*(pow(pd,2)+pd*z+pow(z,2))),2)
    M_inertia = M_total * (M_ine1 + M_ine2)
    return [M_total,M_inertia]
