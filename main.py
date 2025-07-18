import matplotlib.pyplot as plt
import numpy as np
import ansys.mapdl.core


import time 
import imageio.v2 as imageio

from ansys.mapdl.core import launch_mapdl 


from ansys.mapdl.core.examples import vmfiles

import pyvista as pv
import os
if os.path.exists("file.rth"):
    os.remove("file.rth")
if os.path.exists("file.db"):
    os.remove("file.db")
if os.path.exists("temperature_animation.gif"):
    os.remove("temperature_animation.gif")
os.makedirs("frames_temp", exist_ok=True)
os.makedirs("frames_stress", exist_ok=True)
os.makedirs("frames_strain", exist_ok=True)
os.makedirs("Simfiles", exist_ok=True)
# Launch MAPDL instance. Using a jobname ensures all created files (e.g., .rth) are linked.
mapdl = launch_mapdl(nproc = 4,run_location="C:/Users/kanis/FFF1/Simfiles", jobname="FFF_sim",override = True)

#=============================================================================================================================

mapdl.clear()
# Preprocessor activate
mapdl.prep7()
mapdl.units("MKS")
# Define element type and material
mapdl.et(1, "SOLID226")

#element solid 227 for Coupled analysis
mapdl.keyopt(1,1,11)           # DOFs: UX, UY, UZ, TEMP (coupled-field)
mapdl.keyopt(1, 3, 0)   # Element behavior: structural-thermal
mapdl.keyopt(1,10,1)

temps = [20, 40, 60, 80, 100, 120]
mapdl.mptemp("", *temps)

mat_id=1
E_vals = [3.0e9, 2.5e9, 2.0e9, 1.5e9, 1.0e9, 5.0e8,]
mapdl.mpdata("EX", mat_id, "", *E_vals)

# Poisson's ratio (constant)
mapdl.mp("NUXY", mat_id, 0.38)

# Thermal Conductivity (W/m·K)
k_vals = [0.19, 0.20, 0.21, 0.22, 0.23, 0.24]
mapdl.mpdata("KXX", mat_id, "", *k_vals)

# Specific Heat Capacity (J/kg·K)
cp_vals = [900, 950, 1000, 1050, 1100, 1150]
mapdl.mpdata("C", mat_id, "", *cp_vals)

# Density (kg/m³)
rho_vals = [1380, 1375, 1370, 1365, 1360, 1355]
mapdl.mpdata("DENS", mat_id, "", *rho_vals)


# Yield strength in Pa

emisivity = 0.85
mapdl.mp("EMIS",mat_id,emisivity)
matbed = 1

mat_id = 2

#=========================================================================

Density_rubber=1000                # kg/m^3
SpecificHeat_rubber=2093.4         # J/kg.C
ThermalExpansion_rubber=80e-6      # 1/C
Go=2*1.155e6                       # Initial shear modulus in Pa
gr=0.3                             # Relative shear modulus (unitless)
Ko=2*1000e6                        # Initial bulk modulus in Pa
tauG=0.1                           # Characteristic relaxation time (shear modulus) in s

# Define temperature points (°C)
#PLA
#Properties===================================================================================================
# Material number
temps = [20, 40, 60, 80, 100, 120, 140, 150, 170, 200]
specific_heat = [1200, 1270, 1325, 1370, 1410, 1445,3000,2750,1760,1750]# J/kg·K
thermal_cond = [0.13, 0.145, 0.16, 0.175, 0.19, 0.205,0.205,0.210,0.215,0.218]  # W/m·K
density = [1250, 1245, 1240, 1235, 1230, 1225,1210,1200,1185,1175]

# --- Define the MPTEMP points ---
for i, t in enumerate(temps):
    mapdl.mptemp(i+1, t)  # i+1 is the index (1-based)

# --- Assign Specific Heat ---
for i, c in enumerate(specific_heat):
    mapdl.mpdata("C", mat_id, i+1, c)

# --- Assign Thermal Conductivity ---
for i, k in enumerate(thermal_cond):
    mapdl.mpdata("KXX", mat_id, i+1, k)

# --- Assign Density ---
for i, d in enumerate(density):
    mapdl.mpdata("DENS", mat_id, i+1, d)

alpha_vals = [7.3e-5, 7.5e-5, 7.7e-5, 7.9e-5, 8.1e-5, 8.3e-5]
mapdl.mp("ALPX", mat_id, 7.8e-5)

mapdl.mp("NUXY", mat_id, 0.36)
youngs_modulus = 3.2e9       # Pa
yield_strength = 52.0e6        # Pa
mapdl.mp("EX", mat_id, youngs_modulus) 

emisivity = 0.75
mapdl.mp("EMIS",mat_id,emisivity)

matprint = 2
#print(mapdl.mplist())
#=======================================================================================================================
#Geomentry Creation
bed_x = 0.6#in meters
bed_y = 0.6
bed_z = 0.02

# Cube dimensions
cube_dim = 0.04
cube_z_offset = bed_z # Cube sits on top of the bed

# Printing parameters
layer_height = 0.00175
num_layers = int(cube_dim / layer_height)


#Without Bed
'''
mapdl.block(0, bed_x, 0, bed_y, 0, bed_z)
mapdl.allsel()
mapdl.vsel("S","LOC", "Z", 0, bed_z)
mapdl.cm("BED", "VOLU")
mapdl.vatt(1, 4, 1, 0)   # ET=1, MAT=1
'''
# Cube: 10x10x10 mm on top of bed

mapdl.block((bed_x - cube_dim) / 2, (bed_x + cube_dim) / 2, (bed_y - cube_dim) / 2, (bed_y + cube_dim) / 2, cube_z_offset, cube_z_offset + 0.05)
#mapdl.cylind(0.5, 1, z1=0, z2=0.1)
#mapdl.nummrg("kp")
mapdl.vsel("S", "LOC", "Z", 0, 0.1)
mapdl.cm("CUBE", "VOLU")
mapdl.vatt(2, 1, 1, 0)

#==========================================================
#Mesh Generation
#mapdl.vplot()
'''
mapdl.cmsel("S", "BED")
mapdl.lsel("S","LOC","X",0)
mapdl.lsel("R","LOC","Y",0)
mapdl.lesize(0.01)

# Coarse mesh
ms = mapdl.vmesh("ALL")

print(ms)'''

mapdl.cmsel("S", "CUBE")
mapdl.esize(0.01)     # global element size # Finer mesh
ms = mapdl.vmesh("ALL")
print(ms)

#===============================================================================================================================
# define contact between bed and print 
# not in use currently


def contact_region(mapdl,Contact,target,c_pos,TCC=1000):

    #Select and save as a named component
    mapdl.nsel('NONE')  # Clear selection
    mapdl.cmsel("S",target)
    mapdl.aslv("S")
    mapdl.asel("R","LOC","Z",0.02)
    mapdl.nsla("S")
    mapdl.cm('BED_contact', 'NODE')  # Save as named component
    
    
    mapdl.allsel()
    mapdl.cmsel("S",Contact)
    mapdl.aslv("S")
    mapdl.asel("R","LOC","Z",0.02)
    mapdl.nsla("S")
    mapdl.cm('Print_contact', 'NODE')
    mapdl.et(2, "TARGE169")
    mapdl.et(3, "CONTA175") 

    mapdl.keyopt(3, 1, 2)   # Thermal contact only
    mapdl.keyopt(3, 2, 0)   # Pure penalty method (no Lagrange multipliers)
    mapdl.keyopt(3, 2, 0)    # Asymmetric contact
    mapdl.keyopt(3, 12, 5)  # Always bonded (thermal contact always on)
    mapdl.keyopt(3, 4, 0)    # Contact detection method: 0 = pinball only
    mapdl.keyopt(3, 11, 1)   # Pinball radius control: 1 = user-defined

    # Define the real constant for pinball radius and TCC (optional)
    pinball_radius = 0.01
    tcc = 10000
    mapdl.r(1, r1=pinball_radius, r2=tcc)
    mapdl.real(1)
    mapdl.allsel()

    mapdl.allsel()
    mapdl.cmsel("S","BED_contact")
    print("targ",mapdl.mesh.nnum)
    mapdl.type(2)
    mapdl.esurf() 
    #mapdl.vatt(1, 1, 1, 0) 
    mapdl.esel("S","Type","",2)
    mapdl.eplot("S")
    numb_cont_elem = int(mapdl.get_value("ELEM",item1 = "COUNT"))
    print(f"Number of target created = {numb_cont_elem}") 

def local_contact(mapdl,el_num,geom2 = "Print_contact",TCC=1000,pin_r = 0.01):
    with mapdl.run_as_routine('PREP7'):
        '''mapdl.et(3, "CONTA174") 

        mapdl.keyopt(3, 1, 2)   # Thermal contact only
        mapdl.keyopt(3, 2, 0)   # Pure penalty method (no Lagrange multipliers)
        mapdl.keyopt(3, 2, 0)    # Asymmetric contact
        mapdl.keyopt(3, 12, 5)  # Always bonded (thermal contact always on)
        mapdl.keyopt(3, 4, 0)    # Contact detection method: 0 = pinball only
        mapdl.keyopt(3, 11, 1)   # Pinball radius control: 1 = user-defined

        # Define the real constant for pinball radius and TCC (optional)
        pinball_radius = 0.6
        tcc = 1000
        mapdl.r(1, r1=pinball_radius, r2=tcc)
        mapdl.real(1)
        mapdl.allsel()'''

        mapdl.allsel()

        mapdl.esel("S","ELEM","",el_num)
        mapdl.nsle("S")  # Select the element
        mapdl.cmsel("R",geom2)
        mapdl.type(3)
        mapdl.r(1)
        mapdl.esurf()
        mapdl.esel("S","Type","",2)
        numb_cont_elem = int(mapdl.get_value("ELEM",item1 = "COUNT"))
        print(f"Number of target created = {numb_cont_elem}")
        mapdl.esel("S","Type","",3)
        numb_cont_elem = int(mapdl.get_value("ELEM",item1 = "COUNT"))
        print(f"Number of contact created = {numb_cont_elem}")

#======================================================================================================================
def part_elem(mapdl,p_name):
    mapdl.cmsel("S",p_name)
    mapdl.eslv("S")
    part_elements = mapdl.mesh.enum
    mapdl.nsle("S")
    #print(mapdl.nlist())
    print(f"Total elements in part: {len(part_elements)}")
    return part_elements

def kill_print(mapdl,Print_body):
    mapdl.esel("S","LIVE")
    live = mapdl.get_value("ELEM", item1="COUNT")
    print("Number of live elements before kill:",live)
    mapdl.cmsel("S",Print_body)
    mapdl.eslv("S")
    mapdl.ekill("ALL")
    mapdl.esel("S","LIVE")
    live = mapdl.get_value("ELEM", item1="COUNT")
    print("Number of live elements after kill:",live)
    

#contact_region(mapdl,"CUBE","BED",0.02)
#f = part_elem(mapdl,"CUBE")

#radiation coefficientis significantly smaller than convection coefficient

def apply_bc(mapdl,geom = "CUBE",conv_coeff = 25.0, ambient_temp = 60.0):
    mapdl.allsel()
    #mapdl.cmsel("S", "BED")
    #mapdl.nsel("S", "LOC", "Z", 0,0.02)
    #mapdl.d("ALL","TEMP",60) # Ambient temperature on the bed
    mapdl.nsel("S", "LOC", "Z", 0.02)
    mapdl.d("ALL", "UZ", 0)
    mapdl.d("ALL", "UX", 0)
    mapdl.d("ALL", "UY", 0)
    mapdl.allsel()
    mapdl.nslv("S") 
    mapdl.sf("ALL","CONV", conv_coeff, ambient_temp) 
    mapdl.allsel()
    mapdl.nsel("S", "LOC", "Z", 0.02)
    #mapdl.nplot()  # Select nodes at Z = 0.02
    #mapdl.sfdele("ALL","CONV") 
    mapdl.nsel("S","Sf","CONV") # Select nodes with convection
    conv_bcnodes = mapdl.mesh.nnum
    print("Number of nodes with convection BC:", len(conv_bcnodes))

#====================================select elemnts in tool path ===================================================================
#currently not used in program

def setup_path(mapdl,elem_map_id,elem):
    for eid in elem:
        x = mapdl.get_value("ELEM",eid,"CENT","X")
        y = mapdl.get_value("ELEM",eid,"CENT","Y")
        z = mapdl.get_value("ELEM",eid,"CENT","Z")
        elem_map_id.append([x,y,z])
    np.array(elem_map_id)
    print(elem_map_id)
    for layer , loc in enumerate(generate_linear_toolpath(0.295+layer_height/2,0.305-layer_height/2,0.295+layer_height/2,0.305-layer_height/2,0.02+layer_height/2,0.03-layer_height/2,0.00175,1)):
        for index,ijk in enumerate(loc[:]):
            print(index,":",ijk)
            cluster_sel(ijk[0],ijk[1],ijk[2],index)
            id_val = index
        print("----")

def get_elements_near_point(x, y, z,radius=0.00175/2):
    target = np.array([x, y, z])
    dists = np.linalg.norm(elem_map_id - target, axis=1)
    indices = np.where(dists <= radius)[0]
    # Filter out already activated
    ids = [elem[i] for i in indices]
    return ids


def cluster_sel(x,y,z,i,radius = 0.0025):
    mapdl.esel("NONE")
    for eid in get_elements_near_point(x,y,z):
        mapdl.esel("A","ELEM","",eid)
    
    mapdl.cm(f"clust{i}_lay","ELEM")
    print(mapdl.mesh.enum)

def generate_linear_toolpath(x1,x2,y1,y2,z1,z2,step_size, num_layers):
    path_points_by_layer = []
    x_steps = np.arange(x1, x2, step_size)
    y_steps = np.arange(y1, y2, step_size)
    for i in range(num_layers):
        z_layer = z1 + i * (z2-z1) + (z2-z1) / 2
        layer_points = []
        for x in x_steps:
            for y in y_steps: # fixed y center
                layer_points.append([x, y, z_layer])
        path_points_by_layer.append(layer_points)
    return path_points_by_layer
time = []
temp_prof = np.empty((3,0))

#=========================================================================plotting========================================
# for creating the plots and show results 
from matplotlib import cm
def scatter_hist(x, y, ax, ax_histx, ax_histy,tvar=[],t = []):
    # no labels
    #ax_histx.tick_params(axis="x", labelbottom=False)
    #ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    cmap = plt.cm.get_cmap('viridis')

    # Normalize y to 0–1 for colormap
    norm_y = (y - np.min(y)) / (np.max(y) - np.min(y))

    # Stem plot returns: markerline, stemlines, baseline
    markerline, stemlines, baseline = ax.stem(x, y, bottom=1.1)

    # Apply color to each stem and marker
    cmap = cm.get_cmap('viridis')
    colors = cmap(norm_y)

# stemlines and markerline are LineCollection:
    stemlines.set_color(colors)

    # Markers are Line2D objects too:
    # markerline is a Line2D, so set color for all markers
    ax.set_ylim([np.min(y)-np.min(y)/500,np.max(y)+np.min(y)/500])
    
    
    # now determine nice limits by hand:
    binwidth = 5
    xymax = np.max(np.abs(y))
    binwidth2 = np.abs(np.min(y)-np.max(y))/10
    if binwidth2==0:
        binwidth2 = 0.5
    lim2 = (int(xymax/binwidth2) + 1) * binwidth2

    bins2 = np.arange(np.min(y)-binwidth2, lim2 + binwidth2, binwidth2)
    ax_histx.plot(t, tvar[0], linestyle='--', marker='.', color='r', label='Max Temp')
    ax_histx.plot(t, tvar[1], linestyle='-', marker='.', color='g', label='Mean Temp')
    ax_histx.plot(t, tvar[2], linestyle='-', marker='.', color='b', label='Min Temp')
    ax_histx.legend(loc='best')
    # Set xlim and ylim for the stem plot
    ax_histx.set_xlim([np.min(t), np.max(t)])
    ax_histx.set_ylim([np.min(tvar[2]), 190.2])
    ax_histy.hist(y, bins=bins2, orientation='horizontal')
    
def plot_res(mapdl,i):
    global temp_prof
    global time
    mapdl.allsel()
    mapdl.esel("S","LIVE")
    #mapdl.nsle("S")
    node_ids = mapdl.mesh.enum
    temps = []
    stress = []
    disp = []
    for id in node_ids:
        mapdl.esel("S","ELEM","",id)
        mapdl.nsle("S")
        temp1 = mapdl.post_processing.nodal_temperature()
        strain1=mapdl.post_processing.nodal_thermal_principal_strain('1')
        disp1 = mapdl.post_processing.nodal_displacement()
        temps.append(np.mean(temp1))
        stress.append(np.mean(strain1))
        disp.append(np.max(disp1))
    new_col = np.array([[np.max(temps)], [np.mean(temps)],[np.min(temps)]])
    temp_prof = np.hstack((temp_prof, new_col))
    fig, axs = plt.subplot_mosaic([['histx', '.'],
                            ['scatter', 'histy']],
                            figsize=(10, 10),
                            width_ratios=(7, 1), height_ratios=(4, 4),
                            layout='constrained')
    scatter_hist(node_ids, temps, axs['scatter'], axs['histx'], axs['histy'],temp_prof,time)
    axs['scatter'].set_xlabel('Node ID')
    axs['scatter'].set_ylabel('Temperature [C]')
    axs['scatter'].set_title('Node Temperature Scatter')
# Histogram X
    axs['histx'].set_title('Transient Temperature')
    axs['histx'].set_xlabel("Time[s]")
    axs['histx'].set_ylabel("Temperature[s]")
# Histogram Y
    axs['histy'].set_title('Temperature \nDistribution')
    
    fig.suptitle('Element Stress Analysis', fontsize=16)
    fig.savefig(f"plots/Mean_Temp_plot_{i}.png")
    plt.close("all")
    fig, axs = plt.subplot_mosaic([['histx', '.'],
                            ['scatter', 'histy']],
                            figsize=(10, 10),
                            width_ratios=(7, 1), height_ratios=(4, 4),
                            layout='constrained')
    scatter_hist(node_ids, stress, axs['scatter'], axs['histx'], axs['histy'],temp_prof,time)
    axs['scatter'].tick_params(axis = "x",top = True,bottom = False,labeltop = True,labelbottom = False)
    axs['scatter'].set_xlabel('Node ID')
    axs['scatter'].set_ylabel('Strain [C]')
    axs['scatter'].set_title('Node Strain Scatter')
# Histogram X
    axs['histx'].set_title('Node ID Distribution')
    axs['histx'].set_xlabel("Time[s]")
    axs['histx'].set_ylabel("Temperature[s]")

# Histogram Y
    axs['histy'].set_title('Strain \nDistribution')
    fig.suptitle('Element strain Analysis', fontsize=16)
    fig.savefig(f"plots/Mean_Strain_plot_{i}.png")
    plt.close("all")

#===========================================================================Solver setup=================================


def solve_temp(mapdl,el_num,deposition_temp = 190,dt=0.02,i=0):
    mapdl.allsel()
    mapdl.esel("NONE")
    mapdl.esel("S","ELEM","",el_num)
    mapdl.ealive("ALL")
    mapdl.nsle("S") 
    mapdl.d("ALL", "TEMP", deposition_temp)
    #mapdl.bf("ALL","HGEN",5)
    mapdl.sf("ALL", "CONV",75.0, 60.0)  # Apply convection to all nodes
    
    #local_contact(mapdl,el_num)
    mapdl.allsel()    
    mapdl.solve() # Clear temperature boundary conditions
    mapdl.allsel()
    mapdl.ddele("ALL","TEMP")
    
    #mapdl.nsle("S")
    '''mapdl.esel("S","LIVE")
    mapdl.esel("U","ELEM","",el_num)  # Select the element
    mapdl.nsle("S")'''
    
    plot_res(mapdl,i)

    #temp2 = temps[:len(node_ids)]  # Ensure temps matches node_ids length
    '''for nid, t in zip(node_ids, temps): 
        mapdl.ic(nid, 'TEMP', t)'''
    

def initiate_solve(mapdl,geom = "CUBE",dep_temp = 190,current_time = 0.0, element_duration=1, time_steps_per_element=5):
    dt = element_duration / time_steps_per_element
    current_time += dt
    mapdl.finish()

    mapdl.run('/solu')
    mapdl.antype("trans")
    mapdl.trnopt("FULL")
    mapdl.nropt("FULL")
    mapdl.cnvtol("F", "", "", "", 1e-6) 
    mapdl.nlgeom("ON")

    kill_print(mapdl,geom)
    mapdl.timint("ON")
    mapdl.kbc(1)
    
    mapdl.autots("ON")
    mapdl.deltim(dt)

    mapdl.time(current_time)
    #mapdl.nsubst(5)
    mapdl.tunif(60.0)
    mapdl.outres("ALL", "ALL")
    mapdl.allsel()
    apply_bc(mapdl,190,75.0,60.0)
    el_list = part_elem(mapdl,geom)
    steps = 50
    
    for i in range(0,steps):
        time.append(current_time)
        solve_temp(mapdl,el_list[i],dep_temp,dt,i)  # Activate first element
        current_time+=element_duration
        print("Progress:",'%.2f'%(current_time*100/(steps*0.1)))
        
        mapdl.time(current_time)
    

#==============================================================================Solve_==========================================
'''
#setup a tool path
# mapdl.allsel()
elem = mapdl.mesh.enum
elem_map_id = []
id_val = 0
setup_path(mapdl,elem_map_id,elem)
'''

element_duration = 0.1
time_steps_per_element = 2
dt = element_duration / time_steps_per_element
current_time = 0.0

initiate_solve(mapdl,"CUBE",190,current_time,element_duration,time_steps_per_element)
mapdl.post1()
nsets = mapdl.result.nsets 
print(f"Number of result sets: {nsets}")
mapdl.save()
print("END")
#==================================================================================================
# animate temperature result
mapdl.post1()

frames = []
frames2 = []

def res_temp_frame(mapdl, time, i):
    mapdl.esel("S", "LIVE")
    mapdl.nsle("S")
    img_path = f"frames_temp/frame_{i:03d}.png"
    mapdl.post_processing.plot_nodal_temperature(
        show_edges=True,
        cmap = "plasma",
        
        savefig=f"frames_temp/frame_{i:03d}.png",
        background="white",
        cpos=[(0.025, 0.025, 0.3),
            (0.3, 0.3, 0.025),
            (0, 0, 1)],
        scalar_bar_args={
            "title": "Temperature",
            "title_font_size": 16,
            "label_font_size": 14,
            "vertical": True
        },
        nan_color="gray",
        text_color="black",
        off_screen=True,
    )
    frames.append(imageio.imread(img_path))

#strain result 
def res_strain_frame(mapdl, time, i):
    mapdl.esel("S", "LIVE")
    mapdl.nsle("S")
    img_path = f"frames_strain/frame_{i:03d}.png"
    mapdl.post_processing.plot_nodal_thermal_principal_strain(1,
        show_edges=True,
        savefig=f"frames_strain/frame_{i:03d}.png",
        background="white",
        cpos=[(0.025, 0.025, 0.3),
            (0.3, 0.3, 0.025),
            (0, 0, 1)],
        scalar_bar_args={
            "title": f"Thermal_Strain{mapdl.post_processing.time}",
            "title_font_size": 16,
            "label_font_size": 14,
            "vertical": True
        },
        nan_color="gray",
        text_color="black",
        off_screen=True,
    )
    frames2.append(imageio.imread(img_path))
'''
ldstep = 2
substep = 1

et1 = []
et2 = []
et3 = []

es1 = []
es2 = []
es3 = []


for i in range(1,nsets):
    current_time+=dt
    if (i)%3!=0:
        mapdl.set(ldstep,substep)
        print("substep:",mapdl.post_processing.time)
        substep += 1
        res_temp_frame(mapdl,current_time,i)
        #res_strain_frame(mapdl,current_time,i)
        #temp = mapdl.post_processing.element_temperature()
        et1.append(np.mean(temp))
        et2.append(np.max(temp))
        et3.append(np.min(temp))
        
        stress = mapdl.post_processing.element_stress('EQV')
        es1.append(np.mean(stress))
        es2.append(np.max(stress))
        es3.append(np.min(stress))
    if i%3==0:
        
        mapdl.set(ldstep,substep)
        ldstep += 1
        substep = 1 
        print("loadstep:",mapdl.post_processing.time)      
#imageio.mimsave("temp_animation.gif", frames, fps = 30)
#imageio.mimsave("strain_animation.gif", frames2, fps = 30)


time = np.linspace(0.1,current_time, len(et1))
plt.plot(time, et1,linestyle='-.', marker='.', color='g')
plt.plot(time,et2,linestyle="-.",marker = ".", color = 'r')
plt.plot(time,et3,linestyle="-.",marker = ".", color = 'b')
plt.xlabel('Time [s]')
plt.ylabel('Temperature [C]')
plt.legend(['MEAN Temp','MAX Temp','MIN Temp'])
plt.title('Transient Temperature')
plt.savefig("Mean_Temp_plot.png")

time = np.linspace(0.1,current_time, len(es1))
plt.plot(time, es1,linestyle='-.', marker='.', color='g')
plt.plot(time,es2,linestyle="-.",marker = ".", color = 'r')
plt.plot(time,es3,linestyle="-.",marker = ".", color = 'b')
plt.xlabel('Time [s]')
plt.ylabel('Stress [Pa]')
plt.legend(['MEAN Stress','MAX Stress','MIN Stress'])
plt.title('Transient Stress')
plt.savefig("Mean_Stress_plot.png")

'''
mapdl.allsel()
print(mapdl.post_processing.element_temperature())
def final_results_processing(mapdl, results_data, total_elements, params):
    """Final results processing and visualization"""
    
    print(f"\nFINAL RESULTS SUMMARY:")
    print(f"- Total elements processed: {total_elements}")
    print(f"- Total simulation time: {total_elements * params['time_step']:.1f} seconds")
    print(f"- Final max temperature: {results_data[-1]['max_temp']:.1f}°C")
    print(f"- Final min temperature: {results_data[-1]['min_temp']:.1f}°C")
    print(f"- Final avg temperature: {results_data[-1]['avg_temp']:.1f}°C")
    
    # Create visualization plots
    create_results_plots(results_data, params)
    
    # Export results to files
    export_results(results_data)
    
    # Final temperature distribution plot
    mapdl.post1()
    mapdl.set("LAST")
    mapdl.esel("S","LIVE")
    mapdl.pnum("ELEM", 1)
    mapdl.plnsol("TEMP")
    
    # Save final database
    mapdl.save("final_cube_analysis", "db")
    print("✓ Final database saved as 'final_cube_analysis.db'")

def create_results_plots(results_data, params):
    """Create comprehensive results plots"""
    
    # Extract data for plotting
    steps = [r['step'] for r in results_data]
    times = [r['time'] for r in results_data]
    max_temps = [r['max_temp'] for r in results_data]
    min_temps = [r['min_temp'] for r in results_data]
    avg_temps = [r['avg_temp'] for r in results_data]
    active_counts = [r['active_elements_count'] for r in results_data]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Temperature Evolution
    ax1.plot(times, max_temps, 'r-', linewidth=2, label='Maximum')
    ax1.plot(times, avg_temps, 'g-', linewidth=2, label='Average')
    ax1.plot(times, min_temps, 'b-', linewidth=2, label='Minimum')
    ax1.axhline(y=params['deposition_temp'], color='orange', linestyle='--', alpha=0.7, label='Deposition Temp')
    ax1.axhline(y=params['ambient_temp'], color='gray', linestyle='--', alpha=0.7, label='Ambient Temp')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Temperature [°C]')
    ax1.set_title('Temperature Evolution During Element Activation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Active Elements Progress
    ax2.plot(times, active_counts, 'ko-', linewidth=2, markersize=4)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Active Elements Count')
    ax2.set_title('Element Activation Progress')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Temperature Range
    ax3.fill_between(times, min_temps, max_temps, alpha=0.3, color='lightblue', label='Temperature Range')
    ax3.plot(times, avg_temps, 'k-', linewidth=2, label='Average Temperature')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Temperature [°C]')
    ax3.set_title('Temperature Distribution Range')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Activation Rate
    activation_elements = [r['activated_element'] for r in results_data]
    ax4.bar(steps, activation_elements, alpha=0.6, color='skyblue')
    ax4.set_xlabel('Step Number')
    ax4.set_ylabel('Element ID Activated')
    ax4.set_title('Element Activation Sequence')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cube_activation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Results plots created and saved as 'cube_activation_results.png'")
def export_results(results_data):
    """Export results to text files"""
    # Export main results
    with open("cube_activation_results.txt", "w") as f:
        f.write("Step,Time[s],Activated_Element,Active_Count,Max_Temp[C],Min_Temp[C],Avg_Temp[C]\n")
        for r in results_data:
            f.write(f"{r['step']},{r['time']:.3f},{r['activated_element']},"
                    f"{r['active_elements_count']},{r['max_temp']:.2f},"
                    f"{r['min_temp']:.2f},{r['avg_temp']:.2f}\n")
    
    print("✓ Results exported to 'cube_activation_results.txt'")

mapdl.allsel()
mapdl.post1()


#imageio.mimsave("strain_animation.gif", frames2, fps = 0.5)
#imageio.mimsave("stress_animation.gif", frames3, duration=0.5)

'''
time = np.linspace(0.1,0.1+7.0, len(temp_prof))
plt.plot(time, temp_prof,linestyle='-.', marker='o', color='b')
plt.plot(time, ele2t,linestyle='-.', marker='o', color='r')
plt.plot(time, ele3t,linestyle='-.', marker='o', color='g')
plt.plot(time, ele4t,linestyle='-.', marker='o', color='y')
plt.legend(['Element 1', 'Element 2', 'Element 3', 'Element 4'])
plt.xlabel('Time [s]')
plt.ylabel('Stress [Pa]')
plt.title('Stress Profiles')
plt.show()'''


mapdl.exit() 
