import numpy as np
from lxml import etree
import copy
from MU_EV.utils import Hydro_param as hyp
from MU_EV.config import cfg

scaling_factor = cfg.EVO.SCALING
rho = 1000.0*scaling_factor
def arr2str(array):
    """Converts a numeric array into the string format.

    Examples:
        [0, 1, 2] => "0 1 2"
    """        
    arr = ['%.6g' %a for a in array]
    return " ".join(["{}".format(x) for x in arr])

def str2arr(string):
    """Converts a array string in URDF to np.array.
    Examples:
        "0 1 2" => [0, 1, 2]
    """
    return np.array([float(x) for x in string.split(" ")])

def read_size(link):
    """
    Given a URDF link, read the size of visual part.
    Only works while Visual geometry is default type
    """
    return str2arr(link[1][1][0].attrib['size'])

def construct_segment(old_link,x,y,z):
    """
    Construct <Link> for head/body segment in URDF
    Calculate mass and moment of inertia while inertia is calculated around link origin, not CoG
    Inertial, Visual and Collision are pushed back so link Origin will be place at anterior of segment
    !!!!!!!  Need to consistent with HYDRO.CC !!!!!!!!
    """
    link = copy.deepcopy(old_link)
    if link.attrib['name'][:4]=='head':
        
        mass = max(round(x*y*z*rho,9)/3,1e-5)
        ixx = mass/20*(y*y+z*z)
        iyy = mass/20*(x*x+3*z*z)+mass*x*x/4
        izz = mass/20*(x*x+3*y*y)+mass*x*x/4
        comp_mass,comp_iner = hyp.head_comp(x,z)
    else:
        mass = max(round(x*y*z*rho,9),1e-5)
        ixx = mass/12*(y*y+z*z)
        iyy = mass/12*(x*x+z*z)+mass*x*x/4
        izz = mass/12*(y*y+x*x)+mass*x*x/4
        comp_mass,comp_iner = hyp.body_comp(x,z)
    # print(f"DEBUG: segment name: {link.attrib['name']}")
    # print(f"x: {x*1000:.1f}mm, z: {z*1000:.1f}mm")
    # print(f"mass: {mass:.6f}g comp_mass:{comp_mass:.6f}g")
    link[0][0].attrib['xyz'] = arr2str(np.array([x/2,0,0]))             #inertial CoG position
    link[0][1].attrib['value'] = arr2str(np.array([mass+comp_mass]))    #inertial mass
    link[0][2].attrib['ixx'] = arr2str(np.array([ixx]))                 #inertial moment of inertia
    link[0][2].attrib['iyy'] = arr2str(np.array([iyy]))                 #inertial moment of inertia
    link[0][2].attrib['izz'] = arr2str(np.array([izz+comp_iner]))       #inertial moment of inertia

    link[1][0].attrib['xyz'] = arr2str(np.array([x/2,0,0]))             #visual CoG position
    link[1][1][0].attrib['size'] = arr2str(np.array([x,y,z]))           #visual geometry size
    link[2][0].attrib['xyz'] = arr2str(np.array([x/2,0,0]))             #collision CoG position
    link[2][1][0].attrib['size'] = arr2str(np.array([0,0,0]))           #collision geometry size
    return link

def tail_resize(p_D,tail_temp):
    """
    Reading the p_D (peduncle diameter) and tail dimension, calculate the real mass, moment of inertia and shape
    Construct link for URDF
    Tail_temp is the template link for tail, not the real link, size info is stored in visual
    All three tails use box shape in collision
    """
    tail = copy.deepcopy(tail_temp)
    tail_ratio = read_size(tail)
    if tail.attrib['name'][4]=='t':
        # left is the dimension stored in template,    right is the equation for real shape
        # trapezoid[L_ratio,Nan,D2_ratio]:             xyz=[p_D*L_ratio,thick,p_D*D2_ratio]
        # triangle[L_ratio,Nan,1]:                     xyz=[p_D*L_ratio,thick,p_D]        
        x = p_D*tail_ratio[0]
        y = 2e-3
        z = p_D*tail_ratio[2]
    else:
        # left is the dimension stored in template,    right is the equation for real shape
        # Lunate tail[a,c,b_ratio]:                    xyz = [D/tan(a)*b_ratio/2,D*b_ratio*c,D*b_ratio] -> Length, c, depth
        x = p_D/np.tan(tail_ratio[0])/2*tail_ratio[2]
        y = p_D/np.tan(tail_ratio[0])/2*tail_ratio[2]*tail_ratio[1]
        z = p_D*tail_ratio[2]
    tail[1][1][0].attrib['size'] = arr2str(np.array([x,y,z]))
    tail[2][1][0].attrib['size'] = arr2str(np.array([0,0,0]))

    if tail.attrib['name'][:7]=='tailtri':
        tail = triangle_tail_mass(tail, p_D)
    else:# tail.attrib['name'][:7]=='tailtra':
        tail = trapezoid_tail_mass(tail, p_D)
    return tail

def triangle_tail_mass(old_tail,p_D):
    """
    Calculate mass and moment of inertia while inertia is calculated around link origin, not CoG
    Inertial, Visual and Collision are pushed back so link Origin will be place at anterior of segment
    !!!!!!!  Need to consistent with HYDRO.CC !!!!!!!!
    """
    link = copy.deepcopy(old_tail)
    y = 2e-3
    x,_,z = read_size(link)    
    l_cog = x/3
    mass = max(round(x*y*z*rho/2,9),1e-5)
    ixx = mass/24*z**2
    iyy = mass/18*x**2+mass*l_cog**2
    izz = ixx+iyy
    comp_mass,comp_iner = hyp.tailt_comp(p_D,x,z,tri_type=True)
    # print(f"DEBUG: segment name: {link.attrib['name']}")
    # print(f"x: {x*1000:.1f}mm, z: {z*1000:.1f}mm, pd: {p_D*1000:.1f}mm, y: {y*1000:.1f}mm")
    # print(f"mass: {mass:.6f}g comp_mass:{comp_mass:.6f}g")
    link[0][0].attrib['xyz'] = arr2str(np.array([l_cog,0,0]))           #inertia CoG position
    link[0][1].attrib['value'] = arr2str(np.array([mass+comp_mass]))    #inertia Mass
    link[0][2].attrib['ixx'] = arr2str(np.array([ixx]))                 #inertia Moment of inertia
    link[0][2].attrib['iyy'] = arr2str(np.array([iyy]))                 #inertia Moment of inertia
    link[0][2].attrib['izz'] = arr2str(np.array([izz+comp_iner]))       #inertia Moment of inertia
    link[1][0].attrib['xyz'] = arr2str(np.array([l_cog,0,0]))           #visual CoG position
    link[2][0].attrib['xyz'] = arr2str(np.array([l_cog,0,0]))           #collision CoG position
    return link

def trapezoid_tail_mass(old_tail,p_D):
    """
    Calculate mass and moment of inertia while inertia is calculated around link origin, not CoG
    Inertial, Visual and Collision are pushed back so link Origin will be place at anterior of segment
    !!!!!!!  Need to consistent with HYDRO.CC !!!!!!!!

    b is the length of edge that connected to peduncle
    a is the length of free edge
    """
    link = copy.deepcopy(old_tail)
    y = 2e-3
    b = p_D
    h,_,a = read_size(link)
    l_cog=h*(2*a+b)/(3*a+3*b)
    mass = max(round(h*y*(a+b)/2*rho/2,9),1e-5)
    ixx = mass/24*(a**2+b**2)
    iyy = h**3*(a**2+4*a*b+b**2)/36/(a+b)*rho*y+mass*l_cog**2
    izz = ixx+iyy    
    comp_mass,comp_iner = hyp.tailt_comp(p_D,h,a)
    # print(f"DEBUG: segment name: {link.attrib['name']}")
    # print(f"x: {h*1000:.1f}mm, z: {a*1000:.1f}mm, pd: {p_D*1000:.1f}mm, y: {y*1000:.1f}mm")
    # print(f"mass: {mass:.6f}g comp_mass:{comp_mass:.6f}g")
    link[0][0].attrib['xyz'] = arr2str(np.array([h/3*(2*a+b)/(a+b),0,0]))   #inertia CoG position
    link[0][1].attrib['value'] = arr2str(np.array([mass+comp_mass]))        #inertia Mass
    link[0][2].attrib['ixx'] = arr2str(np.array([ixx]))                     #inertia Moment of inertia
    link[0][2].attrib['iyy'] = arr2str(np.array([iyy]))                     #inertia Moment of inertia
    link[0][2].attrib['izz'] = arr2str(np.array([izz+comp_iner]))           #inertia Moment of inertia
    link[1][0].attrib['xyz'] = arr2str(np.array([h/3*(2*a+b)/(a+b),0,0]))   #visual CoG position
    link[2][0].attrib['xyz'] = arr2str(np.array([h/3*(2*a+b)/(a+b),0,0]))   #collision CoG position
    return link

def main_joint_setup(p_size):
    """

    """    
    offset = 1E-4
    position = np.zeros(3)
    axis = np.zeros(3)
    position[0] = p_size[0] + offset
    axis[-1] = 1
    return arr2str(position),arr2str(axis)
    
def joint_124_setup(joint,parent,child,angle):
    """
    Setup the [Parent, Child, rotation Angle]
    Keep else 
    """
    new_joint = joint
    new_joint[1].attrib['link'] = parent.attrib['name']
    new_joint[2].attrib['link'] = child.attrib['name']
    # angel limit
    new_joint[4].attrib['lower'] = '-'+str(angle)
    new_joint[4].attrib['upper'] = str(angle)
    return new_joint

def joint_damping_setup(joint,parent,child,stiff):
    """
    Setup the [Parent, Child, damping]
    Keep else 
    """
    new_joint = joint
    izz_c = float(child[0][2].attrib['izz'])
    izz_p = float(parent[0][2].attrib['izz'])
    izz_eff = izz_c*izz_p/(izz_c+izz_p)
    damping = 2*0.6*np.sqrt(izz_eff*stiff)
    # damping
    new_joint[5].attrib['damping'] = str(round(damping,6))
    return new_joint

def joint_spring_reference(joint_temp,joint_name,stiff):
    """
    Add spring stiffness reference in joint template for gazebo plugin to read
    """
    joint_ref = copy.deepcopy(joint_temp)
    joint_ref.attrib['reference'] = joint_name
    joint_ref[1].attrib['springStiffness'] = str(stiff*scaling_factor)
    return joint_ref

def etree_from_xml(xml, ispath=True):
    """Load xml as etree and return root and tree."""
    if ispath:
        parser = etree.XMLParser(remove_blank_text=True)
        root = etree.parse(xml, parser).getroot()
    else:
        root = etree.fromstring(xml)
    tree = etree.ElementTree(root)
    return root, tree
