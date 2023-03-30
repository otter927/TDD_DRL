from hetnet import HetNet

import sys
import random
import math

import random




def move(self):
    print("move:"+str(self.get_xyz())+" ,"+str(self.direction)+" ,"+str(self.get_name()))
    #self.flag = False
    self.move_t(self.speed)
    
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
        print("mvoeend")
    pre_y_axis = self.y_axis
    self.y_axis = self.y_axis + n
    #if self.y_axis >= self.y_lim_r:
    #    flag = True
    if ((self.flag == False) and(self.flag_end == True)) or (pre_y_axis == self.y_lim_r) or ( (self.flag == False) and (((pre_y_axis - self.y_lim_l) // self.y_grid) != ((self.y_axis - self.y_lim_l) // self.y_grid))):
        self.coner()
        tmp_y_axis = self.y_axis
        print("pre_y_axis:"+str(pre_y_axis))
        #self.y_axis = ((pre_y_axis - self.y_lim_l) // self.y_grid) * self.y_grid + self.y_lim_l
        self.y_axis = ((tmp_y_axis - self.y_lim_l) // self.y_grid) * self.y_grid + self.y_lim_l
        print("tmp_y_axis"+str(tmp_y_axis) + "self.y_axis:"+str(self.y_axis))
        self.move_t(tmp_y_axis - self.y_axis)
    self.flag = False
    
def moving_down(self, n): #2
    self.flag_end = False
    if self.y_axis == self.y_lim_l:
        self.y_axis = self.y_lim_r
        self.flag_end = True
        print("mvoeend")
    pre_y_axis = self.y_axis
    self.y_axis = self.y_axis - n       
    #if self.y_axis <= self.y_lim_l:
    #    flag = True
    if ((self.flag == False) and(self.flag_end == True)) or (pre_y_axis == self.y_lim_l) or ( (self.flag == False) and (((pre_y_axis - self.y_lim_l) // self.y_grid) != ((self.y_axis - self.y_lim_l) // self.y_grid))):
        self.coner()
        tmp_y_axis = self.y_axis
        print("pre_y_axis:"+str(pre_y_axis))
        self.y_axis = ((pre_y_axis - self.y_lim_l)// self.y_grid) * self.y_grid + self.y_lim_l
        print("self.y_axis:"+str(self.y_axis) + "tmp_y_axis"+str(tmp_y_axis))
        self.move_t(self.y_axis - tmp_y_axis)
    self.flag = False
            
def moving_right(self, n): #1
    self.flag_end = False
    if self.x_axis == self.x_lim_r:
        self.x_axis = self.x_lim_l
        self.flag_end = True
        print("mvoeend")
    pre_x_axis = self.x_axis
    self.x_axis = self.x_axis + n
    #if self.x_axis >= self.x_lim_r:
    #    flag = True
    if ((self.flag == False) and(self.flag_end == True)) or (pre_x_axis == self.x_lim_r) or ( (self.flag == False) and (((pre_x_axis - self.x_lim_l) // self.x_grid) != ((self.x_axis - self.x_lim_l) // self.x_grid))):
        self.coner()
        tmp_x_axis = self.x_axis
        print("pre_x_axis:"+str(pre_x_axis))
        #self.x_axis = ((pre_x_axis - self.x_lim_l) // self.x_grid) * self.x_grid + self.x_lim_l
        self.x_axis = ((tmp_x_axis - self.x_lim_l) // self.x_grid) * self.x_grid + self.x_lim_l
        print("tmp_x_axis"+str(tmp_x_axis) + "self.x_axis:"+str(self.x_axis))
        self.move_t(tmp_x_axis - self.x_axis)
    self.flag = False
    
def moving_left(self, n): #3
    self.flag_end = False
    if self.x_axis == self.x_lim_l:
        self.x_axis = self.x_lim_r
        self.flag_end = True
        print("mvoeend")
    pre_x_axis = self.x_axis
    print("self.x_axis0:"+str(self.x_axis))
    self.x_axis = self.x_axis - n
    print("self.x_axis1:"+str(self.x_axis))
    #if self.x_axis <= self.x_lim_l:
    #    flag = True
    #    print("flag:True")
    print("self.x_axis2:"+str(self.x_axis))
    print("(pre_x_axis - self.x_lim_l) "+str(pre_x_axis - self.x_lim_l) )
    print("(self.x_axis - self.x_lim_l)"+str(self.x_axis - self.x_lim_l))
    if ((self.flag == False) and(self.flag_end == True)) or (pre_x_axis == self.x_lim_l) or ( (self.flag == False) and (((pre_x_axis - self.x_lim_l) // self.x_grid) != ((self.x_axis - self.x_lim_l) // self.x_grid))):
        self.coner()
        tmp_x_axis = self.x_axis
        self.x_axis = ((pre_x_axis - self.x_lim_l) // self.x_grid) * self.x_grid + self.x_lim_l
        print("self.x_axis:"+str(self.x_axis) + "tmp_x_axis"+str(tmp_x_axis))
        self.move_t(self.x_axis - tmp_x_axis)
    self.flag = False
            
def coner(self):
    print("coner:"+str(self.get_name()))
    self.flag = True
    #if self.get_UEtype() == "Ped":
    #    prob_ped = random.random()
    #    if prob_ped < (1.0 - 1.0/300):
    #        #print("stay:true,"+str(prob_ped))
    #        self.direction = 4
    #        return()
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
    print("flag:"+str(self.flag)+", "+str(prob)+", "+str(pre_direction)+", "+str(self.direction)+" ,"+str(self.get_name()))



x_grid = 300
y_grid = 300
x_lim_r = 1200
y_lim_r = 1200
x_lim_l = 300
y_lim_l = 300



