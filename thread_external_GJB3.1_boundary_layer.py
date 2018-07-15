# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 14:15:02 2018

@author: Z620
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def rotation(theta, phi):
    return theta + phi

d = 9.972
P = 1.25
H = np.sqrt(3.0)/2.0*P
rho = np.sqrt(3.0)/12.0*P
#rho = 0.0
n = 32
m = 16
segment = 7
boundary_layer = 3

H1P = 0.125
theta1 = np.sqrt(3.0)*np.pi*rho/P
theta2 = (1.0-H1P)*np.pi

print d
print P
print H
print rho
print theta1
print theta2

r_out_list = []
theta_list = []
z_list = [i*P/n for i in range(5*n+1)]
for theta in np.arange(0,np.pi,np.pi/m/2):
    if theta <= theta1:
        theta_list.append(theta)
        r_out_list.append(d/2 -(1.0-H1P)*H + 2*rho - np.sqrt(rho**2-P**2/(4*np.pi**2)*theta**2))
    elif theta <= theta2:
        theta_list.append(theta)
        r_out_list.append(H*theta/np.pi+d/2-(1.0-H1P)*H)
    elif theta <= np.pi:
        theta_list.append(theta)
        r_out_list.append(d/2)

for theta in np.arange(np.pi,2*np.pi,np.pi/m/2):
    theta = 2*np.pi - theta
    if theta <= theta1:
        theta_list.append(2*np.pi - theta)
        r_out_list.append(d/2 -(1.0-H1P)*H + 2*rho - np.sqrt(rho**2-P**2/(4*np.pi**2)*theta**2))
    elif theta <= theta2:
        theta_list.append(2*np.pi - theta)
        r_out_list.append(H*theta/np.pi+d/2-(1.0-H1P)*H)
    elif theta <= np.pi:
        theta_list.append(2*np.pi - theta)
        r_out_list.append(d/2)

#print theta_list
#print r_out_list

r_in_list = []
inner_cycle_diameter = d + 1.5*P
for theta in theta_list:
    r_in_list.append(inner_cycle_diameter/2)

#print theta_list
#print r_in_list

r_interpolated_list = [ [] for seg in range(segment+1)]

for seg in range(segment-boundary_layer+1):
    for i,theta in enumerate(theta_list):
        r_interpolated_list[seg].append((r_out_list[i]-r_in_list[i])/float(segment-boundary_layer+1)*float(seg)+r_in_list[i])

for i,theta in enumerate(theta_list):
    r_interpolated_list[segment].append(r_out_list[i])

r_out_boundary_list = r_interpolated_list[segment-boundary_layer]

for b in range(1,boundary_layer):
    for i,theta in enumerate(theta_list):
        r_interpolated_list[segment-boundary_layer+b].append((r_out_list[i]-r_out_boundary_list[i])/float(boundary_layer)*float(b)+r_out_boundary_list[i])

for r in r_interpolated_list:
    print r

node_list = []
element_list = []

count = 0
for k,z in enumerate(z_list):
    number_in_layer = 1
    for seg in range(segment+1):
        for i,r in enumerate(r_interpolated_list[seg]):
            node_list.append({})
            node_list[count]['node_layer'] = k
            node_list[count]['node_row'] = seg
            node_list[count]['r'] = r
            theta = theta_list[i] + 2*np.pi/n*k
            if theta >= 2*np.pi:
                theta -= 2*np.pi
            node_list[count]['theta'] = theta
            node_list[count]['x'] = r*np.cos(theta)
            node_list[count]['y'] = r*np.sin(theta)
            node_list[count]['z'] = z
            number_in_layer = int(seg*m*4 + round(theta/(np.pi/m/2)) + 1)
            if number_in_layer > (segment+1)*m*4:
                number_in_layer -= (segment+1)*m*4
            node_list[count]['node_number_in_layer'] = number_in_layer
            node_list[count]['node_column'] = (number_in_layer-1) % (m*4)
            node_list[count]['node_number'] = k*(segment+1)*m*4 + number_in_layer
            count += 1

count = 0
for k,z in enumerate(z_list[:-1]):
    element_in_layer = 1
    for seg in range(segment):
        for i,theta in enumerate(theta_list):
            element_list.append({})
            element_list[count]['element_layer'] = k
            element_list[count]['element_row'] = seg
            element_list[count]['element_column'] = i
            
            element_node_order = [0]*8
            
            if i+1 < len(theta_list):
#                print k
#                print max(0,int(4*n*(segment+1)*(k-1))),min(int(4*n*(segment+1)*(k+2)),len(node_list))
                for node in node_list[max(0,int(4*m*(segment+1)*(k))):min(int(4*m*(segment+1)*(k+2)),len(node_list))]:
                    if node['node_layer'] == k and node['node_row'] == seg and node['node_column'] == i:
                        element_node_order[0] = node['node_number']
                    if node['node_layer'] == k and node['node_row'] == seg+1 and node['node_column'] == i:
                        element_node_order[1] = node['node_number']
                    if node['node_layer'] == k and node['node_row'] == seg+1 and node['node_column'] == i+1:
                        element_node_order[2] = node['node_number']
                    if node['node_layer'] == k and node['node_row'] == seg and node['node_column'] == i+1:
                        element_node_order[3] = node['node_number']
                    if node['node_layer'] == k+1 and node['node_row'] == seg and node['node_column'] == i:
                        element_node_order[4] = node['node_number']
                    if node['node_layer'] == k+1 and node['node_row'] == seg+1 and node['node_column'] == i:
                        element_node_order[5] = node['node_number']
                    if node['node_layer'] == k+1 and node['node_row'] == seg+1 and node['node_column'] == i+1:
                        element_node_order[6] = node['node_number']
                    if node['node_layer'] == k+1 and node['node_row'] == seg and node['node_column'] == i+1:
                        element_node_order[7] = node['node_number']
                    

            if i+1 >= len(theta_list):
                for node in node_list[max(0,int(4*m*(segment+1)*(k))):min(int(4*m*(segment+1)*(k+2)),len(node_list))]:
                    if node['node_layer'] == k and node['node_row'] == seg and node['node_column'] == i:
                        element_node_order[0] = node['node_number']
                    if node['node_layer'] == k and node['node_row'] == seg+1 and node['node_column'] == i:
                        element_node_order[1] = node['node_number']
                    if node['node_layer'] == k and node['node_row'] == seg+1 and node['node_column'] == 0:
                        element_node_order[2] = node['node_number']
                    if node['node_layer'] == k and node['node_row'] == seg and node['node_column'] == 0:
                        element_node_order[3] = node['node_number']
                    if node['node_layer'] == k+1 and node['node_row'] == seg and node['node_column'] == i:
                        element_node_order[4] = node['node_number']
                    if node['node_layer'] == k+1 and node['node_row'] == seg+1 and node['node_column'] == i:
                        element_node_order[5] = node['node_number']
                    if node['node_layer'] == k+1 and node['node_row'] == seg+1 and node['node_column'] == 0:
                        element_node_order[6] = node['node_number']
                    if node['node_layer'] == k+1 and node['node_row'] == seg and node['node_column'] == 0:
                        element_node_order[7] = node['node_number']
                    
                    

            element_list[count]['element_node_order'] = element_node_order
            element_list[count]['element_number'] = count+1
            print count
            count += 1
            
#for node in node_list:
#    print node

node_list_sort = sorted(node_list, key=lambda e: e.__getitem__('node_number'))

#for node in node_list_sort:
#    print node

filename = 'thread_external.cmf'
outfile = open(filename, 'w')
outfile.writelines('*collectorcreateonly(components,"solid","",4)' + '\n')
for node in node_list_sort:
    outfile.writelines('*createnode(' + str(node['x'])  +',' + str(node['y']) + ',' + str(node['z']) + ',0,0,0)' + '\n')
for element in element_list:
    node_order = ''
    for n in element['element_node_order']:
        node_order +=  str(n) + ' '
    outfile.writelines('*createlist(node,1)' + node_order + '\n')            
    outfile.writelines('*createelement(208,1,1,0)' + '\n')
outfile.writelines('*writefile("output.hm",1)' + '\n')
outfile.close()

#bat_outfile = open('CREAT_HM.bat', 'w')
#bat_outfile.writelines('call "C:\\Program Files\\Altair\\12.0\\hm\\bin\\win64\\hmbatch.exe" -cnodes.cmf \n')
#bat_outfile.close()

filename = 'thread_external.inp'
outfile = open(filename, 'w')

outfile.writelines('*Part, name=external' + '\n')
outfile.writelines('*Node' + '\n')

for node in node_list_sort:
    outfile.writelines('      ' + str(node['node_number']) + ',' + str(node['x']) + ',' + str(node['y']) + ',' + str(node['z']) + '\n')

outfile.writelines('*Element, type=C3D8' + '\n')
for element in element_list:
    node_order = ''
    for n in element['element_node_order']:
        node_order +=  str(n) + ','
    outfile.writelines('      '+ str(element['element_number']) + ',' + node_order[:-1] + '\n')
outfile.writelines('*End Part')
outfile.close()
                
x = []
y = []
z = []
for node in node_list:
    x.append(node['x'])
    y.append(node['y'])
    z.append(node['z'])

ax = plt.subplot(111,projection='polar')
#projection = 'polar' 指定为极坐标
ax.set_theta_zero_location('E')
ax.set_theta_direction(1)

#ax.plot(theta_list, r_out_list, linewidth=3,color='red')
#ax.plot(theta_list, r_in_list, linewidth=3,color='blue')

for seg in range(segment+1):
    ax.plot(theta_list, r_interpolated_list[seg], ls='', marker='o', color='black')
    
#第一个参数为角度，第二个参数为极径

ax.grid(True) #是否有网格

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(x,y,z)
plt.show()