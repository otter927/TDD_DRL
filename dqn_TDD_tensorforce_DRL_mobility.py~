from hetnet import HetNet

import sys
import random
import math

import random

def calc_d(s, d): 
    ans = math.sqrt(math.pow(s[0] - d[0], 2) + math.pow(s[1] - d[1], 2))
    return(ans)

def if_inside_triangle(px, py, p0x, p0y, p1x, p1y, p2x, p2y):

    Area = 0.5 *(-p1y*p2x + p0y*(-p1x + p2x) + p0x*(p1y - p2y) + p1x*p2y);
    s = 1/(2*Area)*(p0y*p2x - p0x*p2y + (p2y - p0y)*px + (p0x - p2x)*py);
    t = 1/(2*Area)*(p0x*p1y - p0y*p1x + (p0y - p1y)*px + (p1x - p0x)*py);
 
    if((0 < s < 1) and (0 < t < 1) and (0<1-s-t<1)):
        return True #Inside Triangle
    else:
        return False
    

def find_xy_inside_triangel(p0x, p0y, p1x, p1y, p2x, p2y):

    max_x = max([p0x, p1x, p2x])
    min_x = min([p0x, p1x, p2x])

    max_y = max([p0y, p1y, p2y])
    min_y = min([p0y, p1y, p2y])

    answer = False

    while answer == False:
    
        x_random = random.random()
        y_random = random.random()
    
        px = min_x + (max_x - min_x) * x_random
        py = min_y + (max_y - min_y) * y_random
    
        answer = if_inside_triangle(px, py, p0x, p0y, p1x, p1y, p2x, p2y)

    return([px, py])
    

def check_distance(pos, mbs_pos, sbs_pos):
    for i in mbs_pos:
        if calc_d(i, pos) < 75:
            return(False)
    for i in sbs_pos:
        if calc_d(i, pos) < 40:
            return(False)
    return(True)
        
def check_grid(pos, x_grid, y_grid, x_lim_r, y_lim_r, x_lim_l, y_lim_l, sbs_radius):
    near_grid = False
    if pos[0] < x_lim_l - sbs_radius/2 :
        return(False)
    if pos[0] > x_lim_r + sbs_radius/2 :
        return(False)
    if pos[1] < y_lim_l - sbs_radius/2 :
        return(False)
    if pos[1] > y_lim_r + sbs_radius/2 :
        return(False)
    
    for i in range(x_lim_l, x_lim_r+x_grid, x_grid):
        print("pos2:"+str(pos[0])+", i:"+str(i))
        if pos[0] == i:
            return(False)
        print("pos3:"+str(abs(pos[0] - i))+", "+str(sbs_radius/2))
        if abs(pos[0] - i) < (sbs_radius/2):
            near_grid = True
            print("pos4:"+str(near_grid))
    for i in range(y_lim_l, y_lim_r+y_grid, y_grid):
        if pos[1] == i:
            return(False)
        if abs(pos[1] - i) < (sbs_radius/2):
            near_grid = True
            
    return(near_grid)

def find_xy1(x, y, mbs_pos, sbs_pos, x_grid, y_grid, x_lim_r, y_lim_r, x_lim_l, y_lim_l, sbs_radius):

    p0x = x - 216.5
    p0y = y - 125
    p1x = x
    p1y = y
    p2x = x + 216.5
    p2y = y - 125
    
    answer = find_xy_inside_triangel(p0x, p0y, p1x, p1y, p2x, p2y)
    while (check_distance(answer, mbs_pos, sbs_pos) == False) or (check_grid(answer, x_grid, y_grid, x_lim_r, y_lim_r, x_lim_l, y_lim_l, sbs_radius) == False):
        answer = find_xy_inside_triangel(p0x, p0y, p1x, p1y, p2x, p2y)
    
    return(answer)

def find_xy2(x, y, mbs_pos, sbs_pos, x_grid, y_grid, x_lim_r, y_lim_r, x_lim_l, y_lim_l, sbs_radius):

    p0x = x - 216.5
    p0y = y - 125
    p1x = x
    p1y = y
    p2x = x
    p2y = y + 250
    
    answer = find_xy_inside_triangel(p0x, p0y, p1x, p1y, p2x, p2y)
    while (check_distance(answer, mbs_pos, sbs_pos) == False) or (check_grid(answer, x_grid, y_grid, x_lim_r, y_lim_r, x_lim_l, y_lim_l, sbs_radius) == False):
        answer = find_xy_inside_triangel(p0x, p0y, p1x, p1y, p2x, p2y)

    return(answer)

def find_xy3(x, y, mbs_pos, sbs_pos, x_grid, y_grid, x_lim_r, y_lim_r, x_lim_l, y_lim_l, sbs_radius):

    p0x = x + 216.5
    p0y = y - 125
    p1x = x 
    p1y = y
    p2x = x
    p2y = y +250
    
    answer = find_xy_inside_triangel(p0x, p0y, p1x, p1y, p2x, p2y)
    while (check_distance(answer, mbs_pos, sbs_pos) == False) or (check_grid(answer, x_grid, y_grid, x_lim_r, y_lim_r, x_lim_l, y_lim_l, sbs_radius) == False):
        answer = find_xy_inside_triangel(p0x, p0y, p1x, p1y, p2x, p2y)

    return(answer)
    
def if_inside_cercle(px, py, p0x, p0y, radius):

    a = px - p0x;
    b = py - p0y;
    c = math.sqrt(a * a + b * b);

    if (c <= radius/2):
        return True
    else:
        return False

def find_xy_c(p0x, p0y, radius):

    
    max_x = p0x + radius/2
    min_x = p0x - radius/2

    max_y = p0y + radius/2
    min_y = p0y - radius/2

    answer = False
    
    while answer == False:
    
        x_random = random.random()
        y_random = random.random()
    
        px = min_x + (max_x - min_x) * x_random
        py = min_y + (max_y - min_y) * y_random
    
        answer = if_inside_cercle(px, py, p0x, p0y, radius)
        answer2 = minimum_distance(px, py, p0x, p0y, radius)

        answer = answer and answer2
    
    return([px, py])

def find_xy_c_mobility(x_grid, y_grid, x_lim_l, y_lim_l, x_lim_r, y_lim_r):

    px = random.randrange(x_lim_l, x_lim_r+x_grid, x_grid)
    py = random.randrange(y_lim_l, y_lim_r+y_grid, y_grid)

    return([px, py])

def minimum_distance(px, py, p0x, p0y, radius):
    if radius == 500:
        if calc_d([px, py], [p0x, p0y]) > 35:
            return(True)
        else:
            return(False)
    else:
        if calc_d([px, py], [p0x, p0y]) > 10:
            return(True)
        else:
            return(False)





#mode = "conv"
#mode = "schD"
mode = "DRL"

sim = HetNet.Simulator(mode)

print("A")

print("B")

# Get the environment and extract the number of actions.


num_mbs_per_site = 1
#num_sbs_per_mbs = 4

#site_pos = [(1000,500), (567, 750), (1433, 750), (1000, 1000), (567, 1250), (1433, 1250), (1000, 1500)]

site_pos = [(1000,500), (567, 750), (1000, 1000)]


#num_mbs_per_site = 3
#num_sbs_per_mbs = 3

#site_pos = [(1000,1000)]


inter_site_distance = 500
small_cell_coverage = 40

#UE_per_mbs = 10
#UE_per_sbs = 5

UE_per_mbs = 4 * 3
UE_per_sbs = 2


#UE_per_mbs = 5
#UE_per_sbs = 5

system_bandwidth = 10

mbs_antenna = 4
sbs_antenna = 4
UE_antenna = 2

mbs_radius = 500
sbs_radius = 40

mbs_id = 0
sbs_id = 0
ue_id = 0

cell_id = 0

print("C")

sbs_pos = []

cell_color = 0
cell_id = 0


#x_grid = 60
#y_grid = 60
#x_lim_r = 1150
#y_lim_r = 1150
#x_lim_l = 850
#y_lim_l = 850

#x_grid = 160
#y_grid = 160
#x_lim_r = 1720
#y_lim_r = 1720
#x_lim_l = 380
#y_lim_l = 380

#x_grid = 100
#y_grid = 100
#x_lim_r = 1600
#y_lim_r = 1600
#x_lim_l = 400
#y_lim_l = 400

#x_grid = 100
#y_grid = 100
#x_lim_r = 1100
#y_lim_r = 1100
#x_lim_l = 400
#y_lim_l = 400

x_grid = 300
y_grid = 300
x_lim_r = 1200
y_lim_r = 1200
x_lim_l = 300
y_lim_l = 300


#prob_ped = 0.0 #Veh
#prob_ped = 1.0 #Ped
#prob_ped = 0.9 #Ped0.9, Veh0.1
#prob_ped = 0.6 #Ped0.6, Veh0.4
prob_ped = 0.3 #Ped0.3, Veh0.7

prob_ped_eval = 0.9

#mode = "MVG"
mode = "FG"

sim.set_learning_mode(mode)

sim.set_xy_lim(x_grid, y_grid, x_lim_r, y_lim_r, x_lim_l, y_lim_l)

print("C2", flush=True)

for i in site_pos:

    print("D")
    cell_id = cell_id + 1
    mbs_cell_id = cell_id
    for id_mbs in range(num_mbs_per_site):
        mbs = sim.add_bs_agent("MBS", mbs_id, cell_color, i[0], i[1], mbs_cell_id)
        mbs_id = mbs_id + 1
        for _ in range(UE_per_mbs):
            #ueans = find_xy_c(i[0], i[1], mbs_radius)
            ueans = find_xy_c_mobility(x_grid, y_grid, x_lim_l, y_lim_l, x_lim_r, y_lim_r)
            if random.random() < prob_ped:
                sim.add_ue("Ped", ue_id, cell_color, ueans[0], ueans[1], mbs)
            else:
                sim.add_ue("Veh", ue_id, cell_color, ueans[0], ueans[1], mbs)
            ue_id = ue_id + 1

        #for _ in range(num_sbs_per_mbs):
        #    if id_mbs % 3 == 0:
        #        ans = find_xy1(i[0], i[1], site_pos, sbs_pos, x_grid, y_grid, x_lim_r, y_lim_r, x_lim_l, y_lim_l, sbs_radius)
        #    if id_mbs % 3 == 1:
        #        ans = find_xy2(i[0], i[1], site_pos, sbs_pos, x_grid, y_grid, x_lim_r, y_lim_r, x_lim_l, y_lim_l, sbs_radius)
        #    if id_mbs % 3 == 2:
        #        ans = find_xy3(i[0], i[1], site_pos, sbs_pos, x_grid, y_grid, x_lim_r, y_lim_r, x_lim_l, y_lim_l, sbs_radius)
        #    cell_id = cell_id + 1
        #    sbs = sim.add_bs_agent("SBS", sbs_id, cell_color, ans[0], ans[1], cell_id)
        #    sbs_pos.append(ans)
        #    sbs_id = sbs_id + 1
        #    for _ in range(UE_per_sbs):
        #        #ueans = find_xy_c(ans[0], ans[1], sbs_radius)
        #        ueans = find_xy_c_mobility(x_lim_l, y_lim_l, x_lim_r, y_lim_r)
        #        sim.add_ue("Ped", ue_id, cell_color, ueans[0], ueans[1], sbs)
        #        ue_id = ue_id + 1

    cell_color = cell_color + 1

#UE

for i in range(x_lim_l, x_lim_r+x_grid, x_grid):
    for j in range(y_lim_l, y_lim_r+y_grid, y_grid):
        cell_id = cell_id + 1
        cell_color = sim.find_cell_color(i,j)
        sbs = sim.add_bs("SBS", sbs_id, cell_color, i, j, cell_id)
        #sbs_pos.append(ans)
        sbs_id = sbs_id + 1
        for _ in range(UE_per_sbs):
            #ueans = find_xy_c(ans[0], ans[1], sbs_radius)
            ueans = find_xy_c_mobility(x_grid, y_grid, x_lim_l, y_lim_l, x_lim_r, y_lim_r)
            if random.random() < prob_ped:
                sim.add_ue("Ped", ue_id, cell_color, ueans[0], ueans[1], sbs)
            else:
                sim.add_ue("Veh", ue_id, cell_color, ueans[0], ueans[1], sbs)
            ue_id = ue_id + 1

print("E", flush=True)
     

sim.execute_DRL_mobility()


print("B2")



for i in range(10000):

    if (mode == "MVG") and (i<5000) and (i % 1000 == 0) and (i != 0):
        prob_ped = 1-prob_ped
        sim.change_ped(prob_ped)
        print("prob_ped:"+str(prob_ped)+", "+str(i))
    if (i==5000):
        sim.change_ped(prob_ped_eval)
        print("prob_ped:"+str(prob_ped_eval)+", "+str(i))

    print("i:"+str(i), flush=True)
        
    sim.execute_DRL_mobility()

    if (i != 0) and (i % 100 == 0):
        ans1_1, ans1_2, ans1_3, ans2_1, ans2_2, ans2_3, ans3_1, ans3_2, ans3_3, ans4_1, ans4_2, ans4_3, ans5_1, ans5_2, ans5_3, ans6_1, ans6_2, ans6_3 = sim.sum_UE_throughput()
        #i.reset_UE_throughput()
        print("median ul throughput:"+str(ans1_1)+", "+str(ans1_2)+", "+str(ans1_3)+", dl throughput:"+str(ans2_1)+", "+str(ans2_2)+", "+str(ans2_3)+", "+"median ul mbs throughput:"+str(ans3_1)+", "+str(ans3_2)+", "+str(ans3_3)+", dl mbsthroughput:"+str(ans4_1)+", "+str(ans4_2)+", "+str(ans4_3)+", " +"median ul sbs throughput:"+str(ans5_1)+", "+str(ans5_2)+", "+str(ans5_3)+", dl sbs throughput:"+str(ans6_1)+", "+str(ans6_2)+", "+str(ans6_3))
        sim.print_buffer_num()


sim.print_interf("log_DRL")

print("F")



