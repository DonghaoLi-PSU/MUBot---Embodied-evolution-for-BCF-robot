# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:50:50 2023

@author: dongh
"""
from MU_EV.utils.yacs import CfgNode as CN

# Global config object
_C = CN()

# Example usage:
#   from core.config import cfg
cfg = _C

# --------------------------------------------------------------------------- #
# Evolution Options
# --------------------------------------------------------------------------- #
_C.EVO = CN()
_C.EVO.DOCK = ''
# Number of prallel searches per node
# Population size
# Total number of mubot evolved over the course of evolution. Includes the
# initial population size. 
# _C.EVO.NUM_PROCESSES = 18 
_C.EVO.INIT_POPULATION_SIZE = 1
_C.EVO.POPULATION_GOAL = 1
_C.EVO.AGING_WINDOW_SIZE =60

# Number of mubot in a tournament
_C.EVO.TOR_PARTICIPANTS = 4
_C.EVO.SIM_TIME = 6.1


# Template files for URDF and 
_C.MUBOT_TEMPLATE = "MU_EV/template/mubot_template.urdf"
_C.LINK_TEMPLATE  = "MU_EV/template/link_template.urdf"
_C.JOINT_TEMPLATE = "MU_EV/template/joint_template.urdf"
_C.AC_COUPLING = "MU_EV/template/ac_coupling.csv"

# Morphology restriction
_C.EVO.MAX_BODY = 8
_C.EVO.EDGE_DIFF = 31E-3       # max difference between two adjacent body segments
_C.EVO.ORIEN = [False,True] # False for BCF and True for Rajiform
_C.EVO.PFIN = False # whether include pectoral fin
_C.EVO.PFIN_WIDTH = 5E-3

# Morphology parameters setting  
_C.EVO.SCALING = 1000.0
_C.EVO.ANGLE_LIMIT = [0.17, 0.35, 0.52, 0.70, 0.79] # in radians
# _C.EVO.STIFF_RATIO = [ 0.003,  0.010, 0.030,   0.100, 0.200, 0.300, 0.500,1.000]#, 33,100]
_C.EVO.STIFF_RATIO = [0.010, 0.030,   0.100,  0.300, 0.500,1.000]#, 33,100]
_C.EVO.HEAD_SIZE   = [[10E-3, 15E-3, 20E-3,  25E-3],[0.500,0.750,1.000,1.250]] #[[DEPTH(m)],[LENGTH_RATIO]]
_C.EVO.BODY_LENG   = [2.5E-3, 5E-3,  10E-3,  15E-3,  20E-3,  25E-3,  30E-3,  40E-3,  50E-3,  60E-3] #for body length
_C.EVO.BODY_DEPT   = [15E-3,  20E-3, 25E-3,  30E-3,  35E-3,  40E-3,  45E-3,  60E-3] #for body depth
_C.EVO.MAIN_THICK  = 7.5E-3     # Segment thickness for the main body, which is not mutated
_C.EVO.JOINT_SF_BASELINE = 1.0  # baseline joint size for calculating actuator coupling and spring stiffness
_C.EVO.JOINT_AC_BASELINE = 0.71    # baseline actuator coupling for calculating actuator coupling
_C.EVO.MUTATION_OPS = [
    "grow_body",
    "delete_body",
    "mutate_link",
    "mutate_joint"
]

# --------------------------------------------------------------------------- #
# Actuator Options
# --------------------------------------------------------------------------- #
_C.ACT = CN()
# Parameters of standarded actuator
_C.ACT.RESISTANCE = 90
_C.ACT.MAX_F_AMP = 12
_C.ACT.MAX_V_AMP = 20
_C.ACT.MAX_F_SIG = 0.1
_C.ACT.MAX_V_SIG = 0.1

