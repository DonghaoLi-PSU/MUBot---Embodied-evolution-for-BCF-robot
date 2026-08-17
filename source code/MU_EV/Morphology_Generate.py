# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:30:34 2023

@author: dongh
"""

import os,random,sys,copy, time
# import imageio
import numpy as np
from lxml import etree
import pandas as pd
import inspect
##
from MU_EV.utils import xml
from MU_EV.config import cfg
from MU_EV.utils import file as fu
from MU_EV.utils import Hydro_param as hyp

class MUBot_morph:

    def __init__(self, init_path,worker_ID,morph_ID=None,new_morph=False):
        ## Initialize parameters
        self.path = init_path       # /home/donghaoli/test/src/des_mubot
        self.worker_ID = worker_ID  # for multithread parrallel computing
        self.morph_ID = morph_ID    # ID for each morphology
        ## joints only have one template and all mutation value is stored in cfg
        ## link_template: 1st is body/head template and rest are tail tamplate
        temp,_ =  xml.etree_from_xml(os.path.join(self.path,cfg.LINK_TEMPLATE)) 
        self.link_temp = temp[0]  
        self.tail_temp = temp[1:]
        j_temp,_ =  xml.etree_from_xml(os.path.join(self.path,cfg.JOINT_TEMPLATE))
        self.joint_temp =  j_temp[0]
        self.joint_stiff=  j_temp[1]
        ## Load module size for different segment, info in cfg
        self.head_module = cfg.EVO.HEAD_SIZE
        self.body_lgt_module = cfg.EVO.BODY_LENG
        self.body_dpt_module = cfg.EVO.BODY_DEPT
        self.thickness   = cfg.EVO.MAIN_THICK
        ## Load constraints
        self.max_body  = cfg.EVO.MAX_BODY
        self.edge_diff = cfg.EVO.EDGE_DIFF
        self.orien_idx = cfg.EVO.ORIEN
        ## Mutation options
        self.mutation_ops = list(cfg.EVO.MUTATION_OPS)
        ##Load design constraints
        self.angle_limit = cfg.EVO.ANGLE_LIMIT
        self.stiff_ratio = cfg.EVO.STIFF_RATIO
        self.joint_sf_baseline = cfg.EVO.JOINT_SF_BASELINE
        self.joint_ac_baseline = cfg.EVO.JOINT_AC_BASELINE
        self.ac_coupling = self._ac_coupling()    #Load (actuator torque)-(joint angle) trade-off
        self.scaling = cfg.EVO.SCALING
        # inherent morphology or start from beginning        
        if not new_morph:
            self._init_from_state(morph_ID)
        else:
            self._init_new_MUBot()
        self.link_list = self.root.findall('link')
        self.joint_list = self.root.findall('joint')
        self.repeat_idx = 0
        
        return

    def _ac_coupling(self):
        """Reading ac coupling coefficient"""
        df = pd.read_csv(os.path.join(self.path,cfg.AC_COUPLING),header=None,index_col=0)
        return df.to_dict()[1]

    def _init_new_MUBot(self):
        """Construct a new MUBot from template."""
        ## Initalize xml from MUBot template
        ## Template contain one empty head, zero joint and two gazebo plugin
        self.root, self.tree = xml.etree_from_xml(os.path.join(self.path,cfg.MUBOT_TEMPLATE))
        self.joint_param = {}
        self.seg_dict = {}
        ### Construct MUBot
        # Create Head
        head_depth  = random.choice(self.head_module[0])
        head_length = random.choice(self.head_module[1])*head_depth
        self.root[0] = xml.construct_segment(self.link_temp,head_length,self.thickness,head_depth)        
        self.root[0].attrib['name']='head'
        self.seg_dict[self.root[0].attrib['name']] = self.root[0]
        # Create Bodies
        body_num = random.choice(range(2,self.max_body)) 
        ######################################################################################
        # print("DEBUG: force body num") ;         body_num = 2;
        ######################################################################################
        self._add_body_segment(body_num)
        # Added Tail
        self._add_tail()
        ## Save
        self._remapping_for_gazebo()
        if self.duplicate_checking():
            self._init_new_MUBot()
        self._cal_hydrodynamic_param()
        self._save_file()
        return

    def _init_from_state(self,morph_ID):
        """Reading existing MUBot from mubot.urdf"""
        init_path = self.path+"/Population_data/"+str(morph_ID)+"/"
        self.root, self.tree = xml.etree_from_xml(init_path+"mubot.urdf")
        df = pd.read_csv(init_path+"joint_param.csv",header=None,index_col=0)
        my_dict = df.to_dict(orient='index')
        self.joint_param = {index: tuple(values.values()) for index, values in my_dict.items()}
        links = self.root.findall('link')
        self.seg_dict = {}
        for link in links:
            self.seg_dict[link.attrib['name']] = link
        return
    
    def _add_body_segment(self,num_seg):
        """Work for body segment
           num_seg: number of segment to be added  
        """
        for i in range(num_seg):
            seg_list = [key for key in self.seg_dict.keys() if ((key[:4]=='body') or (key[:4]=='head'))] #list for parents
            b_temp = copy.deepcopy(self.link_temp)
            parent_idx = False
            while not parent_idx:  # the last segment is always available for new body segment
                temp_parent = random.choice(seg_list)
                seg_list.remove(temp_parent)
                body_depth, parent_idx,re_idc = self._choose_body_depth(temp_parent)          
            
            body_length = random.choice(self.body_lgt_module)
            new_b = xml.construct_segment(b_temp,body_length,self.thickness,body_depth)
            body_num = len(seg_list)
            while 'body_'+str(body_num) in self.seg_dict.keys():
                body_num +=1
            new_b.attrib['name'] = 'body_'+str(body_num)
            self.seg_dict[new_b.attrib['name']] = new_b
            self.root.insert(-2,new_b)
            parent = self.seg_dict[temp_parent]
            if re_idc:
                self._transfer_main_joint_parent(parent,new_b)
            self._add_main_joint(parent, new_b)
        return
                
    def _add_tail(self):
        """ ONLY CALLED WHEN INITIALIZING A NEW MUBot
        Adding a tail to MUBot
        Two types of tail in template.urdf: Triangle, Trapozoid
        """
        print("TEST: no lunatic tail")
        tail = copy.deepcopy(random.choice(self.tail_temp))
        parent = self.seg_dict[self._find_last_body()[0]]
        p_D = xml.read_size(parent)[2]
        tail = xml.tail_resize(p_D,tail)
        self.root.insert(-2,tail)
        self.seg_dict[tail.attrib['name']] = tail
        self._add_main_joint(parent, tail)
        return

    def _choose_body_depth(self,temp_parent):
        """ BCF only
            For a given parent, test whether new body can attach to it
            Return: depth,success_idx,replace_idc`
            If possible, return [depth,True]
            If no possible depth, return [NaN,False]
        """
        joint = [joint for joint in self.root.findall('joint') if joint[1].attrib['link']==temp_parent and joint[2].attrib['link'][:3] != 'fin']
        p_dep = xml.read_size(self.seg_dict[temp_parent])[2]
        if len(joint)==0: # temp_parent don't have child, no need to adjust joint
            c_dep = np.NaN
            re_idc = False
        else:
            c_dep = xml.read_size(self.seg_dict[joint[0][2].attrib['link']])[2]
            re_idc = True
        depth_list = [dp for dp in self.body_dpt_module if (not abs(dp-c_dep)>self.edge_diff) and (not abs(dp-p_dep)>self.edge_diff)]
        if len(depth_list)!=0:            
            return random.choice(depth_list),True,re_idc
        else:
            return np.NaN,False,re_idc
            
    def _find_last_body(self):
        """
        Tail can only be attached to the last body segments
        """
        b_list = [link.attrib['name'] for link in self.root.findall('link') if link.attrib['name'][:4]=='body'] # name of all current body segment
        p_list = [joint[1].attrib['link'] for joint in self.root.findall('joint') if joint[1].attrib['link'][:4]=='body']
        return list(set(b_list).difference(p_list))

    def _add_main_joint(self,parent,child,):
        """ 
        Create MAIN joints connecting head, bodies and Tail
        """
        j_len = len(self.root.findall('joint'))
        angle_limit = random.choice(self.angle_limit)
        stiff = random.choice(self.stiff_ratio)
        joint = copy.deepcopy(self.joint_temp)
        joint.attrib['name'] = 'joint_'+str(j_len)
        # joint origin and axis
        p_size = xml.read_size(parent)
        c_size = xml.read_size( child)
        joint[0].attrib['xyz'],joint[3].attrib['xyz'] = xml.main_joint_setup(p_size)
        joint = xml.joint_124_setup(joint,parent,child,angle_limit)
        joint = xml.joint_damping_setup(joint,parent,child,stiff)
        self.root.insert(-2,joint)
        self.joint_param[joint.attrib['name']] = self._cal_main_joint_param(parent,child,angle_limit,stiff)
        return
    
    def _cal_main_joint_param(self,parent,child,angle,stiff):
        """
        Calculate the stiffness(sf_param) and actuator coefficient(sf_param) 
        based on j_type, parent-child connecting edge length
        and stiff and ac_coupling
        """
        p_size = xml.read_size(parent)[2]
        c_size = xml.read_size(child)[2]
        ac_param = round(((p_size+c_size)/2)*self.joint_ac_baseline*self.ac_coupling[angle],8)
        sf_param = round(((p_size+c_size)/2)*self.joint_sf_baseline*stiff,8)
        return (ac_param,sf_param,stiff)
    
    def _add_joint_stiffness(self,joint_name,stiff):
        """
        Add spring stiffness reference in joint template for gazebo plugin to read
        """
        joint_ref = copy.deepcopy(self.joint_stiff)
        joint_ref.attrib['reference'] = joint_name
        # idx = etree.SubElement(joint_ref,"implicitSpringDamper")
        joint_ref[0][0].text= str(round(stiff*self.scaling,6))
        return joint_ref
    
    def _remapping_for_gazebo(self):
        # remap the URDF file so that it has the same order as Gazebo parse it.
        temp_root,_ = xml.etree_from_xml(os.path.join(self.path,cfg.MUBOT_TEMPLATE))
        temp_root.remove(temp_root[0])
        root = copy.deepcopy(self.root)
        links = root.findall('link')
        joints = root.findall('joint')
        new_links = []
        new_joints = []
        link_dict = {}
        for link in links:
            link_dict[link.attrib['name']] = link
        new_links.append(link_dict['head'])
        link_indices = [links[0].attrib['name']]
        while len(joints):
            # parent is the current parent segment
            parent = link_indices[-1]
            found = False
            if parent[:4]=='body' or parent[:4]=='head': # found body or head
                j_list = [joint for joint in joints if joint[1].attrib['link'] == parent]
                if len(j_list) ==0:
                    raise Exception(parent+" has no children!") #Only tail can have no children
                elif len(j_list) >1:
                    self._too_many_children(parent)
                else:
                    if len(j_list) ==1: #only child should be a body or tail
                        joint = j_list[0]
                    else:
                        raise Exception("Too many Children!")
                    new_joints.append(joint)
                    joints.remove(joint)
                    child = joint[2].attrib['link']
                    # if child not in link_dict.keys():
                    #     print("DEBUG "+self.gln()+" child not found link_dict: "+str(link_dict.keys()))
                    new_links.append(link_dict[child])
                    found = True
                    link_indices.append(child)                    
            if not found:
                link_indices.pop()
        for link in new_links:
            temp_root.insert(-2, link)
        for joint in new_joints:
            temp_root.insert(-2, joint)
        for joint in new_joints:
            joint_ref = self._add_joint_stiffness(joint.attrib['name'],self.joint_param[joint.attrib['name']][1])
            temp_root.insert(-2, joint_ref)
        self.root = temp_root
        self._morphology_ID_renamening()
        return
    
    def _too_many_children(self,parent):
        self.morph_ID = 'error_ID'
        self._save_file()
        raise Exception(parent+"Too many Children!")
    
    def _find_parent_edge(self,link):
        temp_joint = self.root.findall('joint')
        parent_joint = [joint for joint in temp_joint if joint[2].attrib['link']==link.attrib['name']]
        assert len(parent_joint)==1        
        parent_edge = xml.read_size(self.seg_dict[parent_joint[0][1].attrib['link']])[2]
        return parent_edge
        
        
    def _cal_hydrodynamic_param(self):
        """
        Calculate 
        Info_M_total, Info_M_S, Info_M_0, Info_M_l,
        Info_Cog, Info_lt Info_h_0,
        Info_IH0, Info_IH1, Info_IH2, Info_IH3, Info_IP
        """
        link_list = self.root.findall('link')
        hydro_dict = {}
        for link in link_list:
            if link.attrib['name']=='world':
                continue
            link_size = xml.read_size(link)
            if link.attrib['name'][:4]=='body' or link.attrib['name'][:4]=='head':
                temp_react = hyp.body_reactive(link_size)
                temp_resis = hyp.body_resistive(link_size)
                temp_geom  = [link_size[0]/2,link_size[0],link_size[2]]
                temp_foil  = [0,0]
                if link.attrib['name'][:3]=='hea':
                    temp_react[2] = 0
            elif link.attrib['name'][:5]=='tailt':
                p_d = self._find_parent_edge(link)
                temp_react = hyp.tailt_reactive(link.attrib['name'],p_d,link_size)
                temp_resis = hyp.tailt_resistive(link.attrib['name'],p_d,link_size)
                temp_geom  = [0,link_size[0],link_size[2]]
                if link.attrib['name'][:7]=='tailtri':
                    link_size[2]=0
                temp_geom[0] = (p_d+2*link_size[2])/3/(p_d+link_size[2])*link_size[0]
                temp_foil  = [0,0]
            elif link.attrib['name'][:5]=='taill':
                temp_react = [0,0,0,0,0]
                temp_resis = [0,0,0,0,0]
                temp_geom  = [0,0,0]
                temp_foil  = [1,1]#hyp.taill_airfoil(link)
            elif link.attrib['name'][:3]=='fin':
                temp_react = [0,0,0,0,0]
                temp_resis = [0,0,0,0,0]
                temp_geom  = [0,0,0]
                temp_foil  = [1,1]#hyp.fin_airfoil(link)
            hydro_dict[link.attrib['name']] = temp_react+temp_geom+temp_resis+temp_foil
        # if not morph_ID:
        #     self.save_ID = self.morph_ID
        # else:
        #     self.save_ID = morph_ID
        df = pd.DataFrame.from_dict(hydro_dict, orient='index')
        fu.df_to_csv(df,self.path,'hydro',self.worker_ID,self.morph_ID)
        return
    
    def _save_file(self):
        """
        Save urdf file, joint_parameter file
        Save into Database/worker_ID and Population_data/morph_ID
        """
        # if not morph_ID: #not mutated
        #     self.save_ID = self.morph_ID
        # else:
        #     self.save_ID = morph_ID
        joint_dict = {}
        for key in self.root.findall('joint'):
            joint_dict[key.attrib['name']] = self.joint_param[key.attrib['name']]
        df = pd.DataFrame.from_dict(joint_dict, orient='index')
        fu.df_to_csv(df,self.path,'joint',self.worker_ID,self.morph_ID)
        fu.urdf_write(self.root,self.path,self.worker_ID,self.morph_ID)
        return
    
    def _morphology_ID_renamening(self):
        """
        Morphology ID rule: 
        segment#_head(2digit:depth#+lengthratio#)_body(2 digit:depth#+length#)_tail(2digtt:type)_joint(2digit:angle#+stiff#)
        """
        head = self.seg_dict['head']
        body_list=[key.attrib['name'] for key in self.root.findall('link') if (key.attrib['name'][:4]=='body') ]
        tail= [element for element in self.root.findall('link') if element.attrib['name'][:4]=='tail'][0]
        joint_list= self.root.findall('joint')
        # print("DEBUG: joint list for ID: "+str([joint.attrib['name'] for joint in joint_list]))
        seg_num = len(body_list)+2
        new_ID = str(seg_num)
        head_size = xml.read_size(head)
        head_depth_idx = self.head_module[0].index(head_size[2])
        head_ratio_idx = self.head_module[1].index(round(head_size[0]/head_size[2],3))
        new_ID = new_ID + '_' + str(head_depth_idx)+str(head_ratio_idx)
        for i in range(len(body_list)):
            body_size = xml.read_size(self.seg_dict[body_list[i]])
            body_depth_idx = self.body_dpt_module.index(body_size[2])
            body_length_idx = self.body_lgt_module.index(body_size[0])
            new_ID = new_ID  + '_' + str(body_depth_idx)+str(body_length_idx)
        tail_name = tail.attrib['name']
        if tail_name[:7]=='tailtra':
            tail_type = 0
        elif tail_name[:7]=='tailtri':
            tail_type = 1
        else:
            tail_type = 2
        tail_idx = tail_type*10+int(tail_name[-1])

        new_ID = new_ID  + '_' + str(tail_idx).zfill(2)+'_'
        for joint in joint_list:
            angle = round(float(joint[4].attrib['upper']),2)
            stiff = round(self.joint_param[joint.attrib['name']][-1],3)
            angle_idx = self.angle_limit.index(angle)
            stiff_idx = self.stiff_ratio.index(stiff)
            new_ID = new_ID  + '_' + str(angle_idx)+str(stiff_idx)
        self.morph_ID = new_ID
        return

    def duplicate_checking(self):
        """"
        Check whether the current morphology already exists in the database
        by scaning the morphology ID under Population_data
        """
        morph_list = os.listdir(self.path+"/Population_data/")
        if self.morph_ID in morph_list:
            # self.repeat_idx += 1
            # print('TEST repeat test: '+str(self.repeat_idx))
            return True
        else:
            return False
        # return
    
    def mutation(self):
        op = list(self.mutation_ops)
        body_list=[element.attrib['name'] for element in self.root.findall('link') if element.attrib['name'][:4]=='body']
        blank_body_list = list(set(body_list)) 
        if len(body_list)>=self.max_body:
            try:
                op.remove("grow_body")
            except:
                pass
        if len(body_list)<=2:
            try:
                op.remove("delete_body")
            except:
                pass
        self.op = random.choice(op)
        print("Mutation start")
        if   self.op =="grow_body": # op = 0
            self._add_body_segment(random.choice(range(1,3)))
        elif self.op == "delete_body": # op = 1
            del_list = random.sample(blank_body_list,k=random.choice(range(1,3)))
            self._del_body(del_list)
        elif self.op == "mutate_link": # op = 2
            self._mutate_segment()
            # self._mutate_segment(random.choice([0,1,3]))
        elif self.op == "mutate_joint":# op = 3
            self._mutate_joint(random.choice(range(1,3)))  
        print("Mutation done")
        self._remapping_for_gazebo()
        if self.duplicate_checking():
            print("Mutation Duplicate ID found, re-mutate MUBot")
            return self.mutation()
        # while os.path.exists(self.path+"/Population_data/"+str(self.morph_ID)):
        #     os.mkdir(self.path+"/Population_data/"+str(self.morph_ID))
        self._cal_hydrodynamic_param()
        self._save_file()
        return self.morph_ID
        
    def _del_body(self,del_body):
        ### check finbase problem
        """"
        delete body segments and posterior joint
        Works for BCF
        """
        
        for i in self.root.findall('link'):
            if i.attrib['name'] in del_body:
                self.root.remove(i)
                self.seg_dict.pop(i.attrib['name'])
        for j in del_body:
            p_joint = [joint for joint in self.root.findall('joint') if joint[2].attrib['link']==j]
            c_joint = [joint for joint in self.root.findall('joint') if joint[1].attrib['link']==j]
            p_joint[0][2].attrib['link'] = c_joint[0][2].attrib['link']
            self.joint_param.pop(c_joint[0].attrib['name'])
            self.root.remove(c_joint[0])
        return
        
    def _mutate_segment(self):
        # "seg_idx=0,1,2 means mutate head,body,tail"
        seg_idx = random.choice([0,1,2]) # ['head','body','tail']
        if (seg_idx == 0):
            child_link = [joint[2].attrib['link'] for joint in self.root.findall('joint') if joint[1].attrib['link']=='head'][0]
            depth_list = [dp for dp in self.head_module[0] if (not dp==xml.read_size(self.seg_dict['head'])[2]) and (not abs(dp-xml.read_size(self.seg_dict[child_link])[2])>self.edge_diff)]
            if len(depth_list)!=0:
                head_depth  = random.choice(depth_list)
                head_length = random.choice(self.head_module[1])*head_depth
            else:
                head_depth  = xml.read_size(self.seg_dict['head'])[2]
                ratio = round(xml.read_size(self.seg_dict['head'])[0]/xml.read_size(self.seg_dict['head'])[2],3)
                head_length = random.choice([lgt for lgt in self.head_module[1] if lgt != ratio])*head_depth
            self.root[0] = xml.construct_segment(self.link_temp,head_length,self.thickness,head_depth)        
            self.root[0].attrib['name']='head'
            self.seg_dict[self.root[0].attrib['name']] = self.root[0]
            self._modify_connected_joints(self.root[0])
        elif (seg_idx==1): #head or body segment.
            # possible_segment = copy.deepcopy([element for element in self.root.findall('link') if element.attrib['name'][:3]==seg_name[seg_idx]])
            possible_segment = copy.deepcopy([element for element in self.root.findall('link') if element.attrib['name'][:4]=='body'])
            new_seg = []
            while len(new_seg)==0:
                seg_chose = random.choice(possible_segment)
                possible_segment.remove(seg_chose)
                new_seg= self._possible_segment_detect(seg_chose)
                if len(new_seg)==0 and len(possible_segment)==0:
                    curr_size = xml.read_size(seg_chose)
                    curr_dpt = curr_size[2]
                    curr_lgt = curr_size[0]
                    new_lgt = random.choice([lgt for lgt in self.body_lgt_module if lgt != curr_lgt])
                    new_seg = xml.construct_segment(copy.deepcopy(self.link_temp),new_lgt,self.thickness,curr_dpt)
            new_seg.attrib['name'] = copy.deepcopy(seg_chose.attrib['name'])
            seg_old_dict = self.seg_dict[seg_chose.attrib['name']]
            seg_old_link = [element for element in self.root.findall('link') if element.attrib['name']==seg_chose.attrib['name']][0]
            if len([element for element in self.root.findall('link') ]) != len(self.seg_dict):
                raise Exception("Segment length error")
            if seg_old_dict[1][1][0].attrib['size'] != seg_old_link[1][1][0].attrib['size']:
                raise Exception("Segment dict and link mismatch error")
            self.root.replace(seg_old_link, new_seg)
            self.seg_dict.pop(seg_old_dict.attrib['name'])
            self.seg_dict[new_seg.attrib['name']] = new_seg
            self._modify_connected_joints(new_seg)
        elif (seg_idx==2): # tail segment 
            new_seg = self._mutate_tail()
        return
        
    
    def _possible_segment_detect(self,link):
        """
           Given a segment for mutation, detect posible alternative segment diameter
           return a random chosen new segment or empty list
           If chosen peduncle??
        """
        current_joints = self.root.findall('joint')
        joint_before = [joint for joint in current_joints if joint[2].attrib['link'] == link.attrib['name']]
        joint_after = [joint for joint in current_joints if ((joint[1].attrib['link'] == link.attrib['name']) and (joint[2].attrib['link'][:4] !='tail'))]
        c_joint = joint_before + joint_after
        # c_joint = [joint for joint in self.joint_list if (((joint[1].attrib['link'] == link.attrib['name']) and (joint[1].attrib['link'][:4] !='tail') ) or (joint[2].attrib['link'] == link.attrib['name']))]
        curr_dpt = xml.read_size(link)[2]
        depth_list = copy.deepcopy(self.body_dpt_module)
        depth_list.remove(curr_dpt)
        link_temp = []
        for joint in c_joint:
            # att_link = self.seg_dict[joint[2].attrib['link']] if (self.seg_dict[joint[2].attrib['link']] != link.attrib['name']) else self.seg_dict[joint[1].attrib['link']]
            att_link = self.seg_dict[joint[2].attrib['link']] if (joint[2].attrib['link'] != link.attrib['name']) else self.seg_dict[joint[1].attrib['link']]
            att_depth= xml.read_size(att_link)[2]
            depth_list = [dp for dp in depth_list if abs(dp-att_depth)<=self.edge_diff]
            if len(depth_list)==0:
                break
        if len(depth_list) !=0:
            temp_dep = random.choice(depth_list)
            temp_lgt = random.choice(self.body_lgt_module)
            link_temp = xml.construct_segment(copy.deepcopy(self.link_temp),temp_lgt,self.thickness,temp_dep)
        return link_temp
    
    def _mutate_tail(self):
        """
        there should be only one tail and joint
        """
        tail_old  = [element for element in self.root.findall('link') if element.attrib['name'][:4]=='tail'][0]
        joint_curr = [joint for joint in self.root.findall('joint') if joint[2].attrib['link'] == tail_old.attrib['name']][0]
        parent = [element for element in self.root.findall('link') if element.attrib['name']==joint_curr[1].attrib['link']][0]
        tail_list = [element for element in self.tail_temp[:9] if element.attrib['name'] != tail_old.attrib['name']]
        tail = copy.deepcopy(random.choice(tail_list))
        if tail.attrib['name'][4]=='t':
            p_D = xml.read_size(parent)[2]
        else:
            p_D = max([xml.read_size(link)[2] for link in self.root.findall('link')])
        tail_new = xml.tail_resize(p_D,tail)

        self.root.replace(tail_old, tail_new)
        self.seg_dict[tail_new.attrib['name']] = tail_new
        self.seg_dict.pop(tail_old.attrib['name'])
        self._update_tail_joint(joint_curr,parent,tail_new)
        return 
    
    def _modify_connected_joints(self,seg_1):
        """
        Called after mutate segment
        for a given segment, find its parent and children, change the joint position in URDF
        DO NOT CHANGE PARENT OR CHILD NAME
        update joint stiffness, ac_coupling, for 
        """
        joint_list = [joint for joint in self.root.findall('joint') if (joint[1].attrib['link'] == seg_1.attrib['name']) or (joint[2].attrib['link'] == seg_1.attrib['name'])]
        for joint in joint_list:
            self._update_joint(joint,self.seg_dict[joint[1].attrib['link']],self.seg_dict[joint[2].attrib['link']])
        return
    
    def _transfer_main_joint_parent(self,seg_0,seg_1):
        """
        Called after insert a new segment
        Transfer the parent of MAIN joint from seg_0 to seg_1, and keep child
        pectoral fin fixed with old parent
        """
        joint_list = [joint for joint in self.root.findall('joint') if joint[1].attrib['link']==seg_0.attrib['name'] and joint[2].attrib['link'][:3] != 'fin']
        for joint in joint_list:
            joint[1].attrib['link'] = seg_1.attrib['name']
            self._update_joint(joint,seg_1,self.seg_dict[joint[2].attrib['link']])
            # joint[0][1].attrib['link'] = seg_1.attrib['name']            
            # self._update_joint(joint,seg_1,self.seg_dict[joint[0][2].attrib['link']])
        return
    
    def _update_joint(self,joint,parent,child):
        """
        Called when a joint need to be updated
        Joint position and axis are updated
        ac_coefficient and stiff are not changed
        """
        p_size = xml.read_size(parent)
        joint[0].attrib['xyz'],joint[3].attrib['xyz'] = xml.main_joint_setup(p_size)
        # if stf_type:
        angle = float(joint[4].attrib['upper'])
        stiff  = self.joint_param[joint.attrib['name']][-1]
        self.joint_param[joint.attrib['name']]=self._cal_main_joint_param(parent,child,angle,stiff)
        joint = xml.joint_damping_setup(joint,parent,child,stiff)
        return
    
    def _update_tail_joint(self,joint,parent,child):
        """
        Called after the tail is mutated
            Update joint position, axis, ac_param and sf_param, ac_angle and stiff are not changed
            Update joint child name
        """
        self._update_joint(joint,parent,child)
        joint[2].attrib['link'] = child.attrib['name']
        return   
    
    def _mutate_joint(self,mut_num):
        joint_list = random.sample(self.root.findall('joint'),k=mut_num)
        for joint in joint_list:
            curr_angle = float(joint[4].attrib['upper'])
            new_angle = random.choice([ang for ang in self.angle_limit if ang != curr_angle])
            new_stiff = random.choice([stf for stf in self.stiff_ratio if stf != self.joint_param[joint.attrib['name']][-1]])
            parent = self.seg_dict[joint[1].attrib['link']]
            child = self.seg_dict[joint[2].attrib['link']]
            self.joint_param[joint.attrib['name']] = self._cal_main_joint_param(parent,child,new_angle,new_stiff)
            joint[4].attrib['lower'] = str(-new_angle)
            joint[4].attrib['upper'] = str( new_angle)
            joint = xml.joint_damping_setup(joint,parent,child,new_stiff)
        return
    
    def gln(self):      
        return str(inspect.currentframe().f_back.f_lineno)