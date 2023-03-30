from hetnet import HetNet

import sys
import random
import math

import random
import csv
import copy

import numpy as np


class UE_mobility():

#    def __init__(self, x, y, speed, kind):
    def __init__(self, x, y, speed):

        self.x_axis = x
        self.y_axis = y
        self.speed = speed
        #self.kind = kind

        self.x_grid = 300
        self.y_grid = 300
        self.x_lim_r = 1200
        self.y_lim_r = 1200
        self.x_lim_l = 300
        self.y_lim_l = 300
        
        self.direction = random.randint(0,3)
        
        self.turn_right_prob = 0.25
        self.turn_left_prob = 0.25
        self.go_strait_prob = 0.25

    def get_xyz(self):
        return((self.x_axis, self.y_axis, 0))

    def get_xy(self):
        return([self.x_axis, self.y_axis])


    def move(self):
        ##print("move:"+str(self.get_xyz())+" ,"+str(self.direction))
        self.flag = False
        self.move_t(self.speed)
        return([self.x_axis, self.y_axis])
    
    def move_t(self, n): #0:up, 1:right, 2:down, 3:left, 4:stay
        #print("move_t:"+str(self.direction))
        if self.direction == 0:
            self.moving_up(n)
        elif self.direction == 1:
            self.moving_right(n)
        elif self.direction == 2:
            self.moving_down(n)
        elif self.direction == 3:
            self.moving_left(n)  
        else:
            self.coner()   
    
    def moving_up(self, n): #0
        self.flag_end = False
        if self.y_axis == self.y_lim_r:
            self.y_axis = self.y_lim_l
            self.flag_end = True
            ##print("mvoeend")
        pre_y_axis = self.y_axis
        self.y_axis = self.y_axis + n
        #if self.y_axis >= self.y_lim_r:
        #    flag = True
        if ((self.flag == False) and(self.flag_end == True)) or (pre_y_axis == self.y_lim_r) or ( (self.flag == False) and (((pre_y_axis - self.y_lim_l) // self.y_grid) != ((self.y_axis - self.y_lim_l) // self.y_grid))):
            self.coner()
            tmp_y_axis = self.y_axis
            ##print("pre_y_axis:"+str(pre_y_axis))
            #self.y_axis = ((pre_y_axis - self.y_lim_l) // self.y_grid) * self.y_grid + self.y_lim_l
            self.y_axis = ((tmp_y_axis - self.y_lim_l) // self.y_grid) * self.y_grid + self.y_lim_l
            ##print("tmp_y_axis"+str(tmp_y_axis) + "self.y_axis:"+str(self.y_axis))
            self.move_t(tmp_y_axis - self.y_axis)
        self.flag = False
    
    def moving_down(self, n): #2
        self.flag_end = False
        if self.y_axis == self.y_lim_l:
            self.y_axis = self.y_lim_r
            self.flag_end = True
            ##print("mvoeend")
        pre_y_axis = self.y_axis
        self.y_axis = self.y_axis - n       
        #if self.y_axis <= self.y_lim_l:
        #    flag = True
        if ((self.flag == False) and(self.flag_end == True)) or (pre_y_axis == self.y_lim_l) or ( (self.flag == False) and (((pre_y_axis - self.y_lim_l) // self.y_grid) != ((self.y_axis - self.y_lim_l) // self.y_grid))):
            self.coner()
            tmp_y_axis = self.y_axis
            ##print("pre_y_axis:"+str(pre_y_axis))
            self.y_axis = ((pre_y_axis - self.y_lim_l)// self.y_grid) * self.y_grid + self.y_lim_l
            ##print("self.y_axis:"+str(self.y_axis) + "tmp_y_axis"+str(tmp_y_axis))
            self.move_t(self.y_axis - tmp_y_axis)
        self.flag = False
            
    def moving_right(self, n): #1
        self.flag_end = False
        if self.x_axis == self.x_lim_r:
            self.x_axis = self.x_lim_l
            self.flag_end = True
            ##print("mvoeend")
        pre_x_axis = self.x_axis
        self.x_axis = self.x_axis + n
        #if self.x_axis >= self.x_lim_r:
        #    flag = True
        if ((self.flag == False) and(self.flag_end == True)) or (pre_x_axis == self.x_lim_r) or ( (self.flag == False) and (((pre_x_axis - self.x_lim_l) // self.x_grid) != ((self.x_axis - self.x_lim_l) // self.x_grid))):
            self.coner()
            tmp_x_axis = self.x_axis
            ##print("pre_x_axis:"+str(pre_x_axis))
            #self.x_axis = ((pre_x_axis - self.x_lim_l) // self.x_grid) * self.x_grid + self.x_lim_l
            self.x_axis = ((tmp_x_axis - self.x_lim_l) // self.x_grid) * self.x_grid + self.x_lim_l
            ##print("tmp_x_axis"+str(tmp_x_axis) + "self.x_axis:"+str(self.x_axis))
            self.move_t(tmp_x_axis - self.x_axis)
        self.flag = False
    
    def moving_left(self, n): #3
        self.flag_end = False
        if self.x_axis == self.x_lim_l:
            self.x_axis = self.x_lim_r
            self.flag_end = True
            ##print("mvoeend")
        pre_x_axis = self.x_axis
        ##print("self.x_axis0:"+str(self.x_axis))
        self.x_axis = self.x_axis - n
        ##print("self.x_axis1:"+str(self.x_axis))
        #if self.x_axis <= self.x_lim_l:
        #    flag = True
        #    print("flag:True")
        ##print("self.x_axis2:"+str(self.x_axis))
        ##print("(pre_x_axis - self.x_lim_l) "+str(pre_x_axis - self.x_lim_l) )
        ##print("(self.x_axis - self.x_lim_l)"+str(self.x_axis - self.x_lim_l))
        if ((self.flag == False) and(self.flag_end == True)) or (pre_x_axis == self.x_lim_l) or ( (self.flag == False) and (((pre_x_axis - self.x_lim_l) // self.x_grid) != ((self.x_axis - self.x_lim_l) // self.x_grid))):
            self.coner()
            tmp_x_axis = self.x_axis
            self.x_axis = ((pre_x_axis - self.x_lim_l) // self.x_grid) * self.x_grid + self.x_lim_l
            ##print("self.x_axis:"+str(self.x_axis) + "tmp_x_axis"+str(tmp_x_axis))
            self.move_t(self.x_axis - tmp_x_axis)
        self.flag = False
            
    def coner(self):
        ##print("coner")
        self.flag = True
#        if self.kind == "Ped":
#            prob_ped = random.random()
#            if prob_ped < (1.0 - 1.0/300):
#                #print("stay:true,"+str(prob_ped))
#                self.direction = 4
#                return()
#        else:
#            prob_ped = random.random()
#            if prob_ped < (1.0 - 1.0/10):
#                #print("stay:true,"+str(prob_ped))
#                self.direction = 4
#                return()
            #print("stay:false,"+str(prob_ped))
                
        #if flag == True:
        #    prob = random.random() * (self.turn_right_prob + self.turn_left_prob)
        #    #print("prob:"+str(prob))
        #else:
        
        prob = random.random()
        pre_direction = self.direction
        if self.turn_right_prob >= prob:
            self.direction = (self.direction + 1) % 4
        elif self.turn_right_prob + self.turn_left_prob >= prob:
            self.direction = (self.direction - 1) % 4
            
        ##print("flag:"+str(self.flag)+", "+str(prob)+", "+str(pre_direction)+", "+str(self.direction))


x_grid = 300
y_grid = 300
x_lim_r = 1200
y_lim_r = 1200
x_lim_l = 300
y_lim_l = 300

ped_speed = 1
veh_speed = 15


#prob_ped = 0.3
#prob_ped = 0.7

#prob_ped = 0.5

#prob_ped = 1.0
#prob_ped = 0.0
#prob_ped = 0.1

#prob_ped = 0.9
#prob_ped = 0.5

#prob_ped = 0.1
#prob_ped = 0.3
#prob_ped = 0.2


args = sys.argv
prob_ped = float(args[1])

#ue_random = np.random.exponential(4,16)
##ue_random = np.random.normal(4.0, 1.0, 16)
#np.random.poisson(arrival_rate, 20) ##max_num self.arrival_rate * self.subframe_duration
##print(ue_random)
##ue_random = ue_random.astype(int)
##print(ue_random)



UE_per_coner = 4
UE_list = []
ind = 0
for i in range(x_lim_l, x_lim_r+x_grid, x_grid):
    for j in range(y_lim_l, y_lim_r+y_grid, y_grid):
        for _ in range(UE_per_coner):
        #for _ in range(ue_random[ind]):
            r_rand = random.random()
            if r_rand < prob_ped:
#                ue = UE_mobility(i, j, ped_speed, "Ped")
                ue = UE_mobility(i, j, ped_speed)
                UE_list.append(ue)
            #elif r_rand < prob_ped + uav_ped:
            #    ue = UE_mobility(i, j, random.randrange(uav_speed-15, uav_speed+15))
            #    #ue = UE_mobility(i, j, ped_speed)
            #    UE_list.append(ue)            
            else:
#                ue = UE_mobility(i, j, ped_speed, "Veh")
                ue = UE_mobility(i, j, veh_speed)
                UE_list.append(ue)
        ind = ind + 1

#UE_per_coner = 4
#UE_list = []
#for i in range(x_lim_l, x_lim_r+x_grid, x_grid):
#    for j in range(y_lim_l, y_lim_r+y_grid, y_grid):
#        for _ in range(UE_per_coner):
#            if random.random() < prob_ped:
#                ue = UE_mobility(i, j, ped_speed)
#                UE_list.append(ue)
#            else:
#                ue = UE_mobility(i, j, veh_speed)
#                UE_list.append(ue)
        
        
#position = []
##for i in range(10):
#for i in range(1000):
#    position_list = []
#    for  ue in UE_list:
#        xy = ue.move()
#        position_list.extend(xy)
#    position.append(position_list)    
#
#cposition = copy.deepcopy(position)
#for i in range(18):
#    position.extend(cposition)


#for i in range(4000):
#    for  ue in UE_list:
#        xy = ue.move()

position = []
#for i in range(10):
for i in range(500):
    position_list = []
    for  ue in UE_list:
        xy = ue.move()
        position_list.extend(xy)
    position.append(position_list)    

cposition = copy.deepcopy(position)
for i in range(36):
    position.extend(cposition)



f = open("data2/position"+str(prob_ped).replace('.', '')+".csv", 'w')
writer = csv.writer(f)
writer.writerows(position)           
