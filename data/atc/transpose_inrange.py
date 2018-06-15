# Collects data from ATC atset within the given spatial range and converts it for compatibility with social attention code
# Update:
# Extracts the part of map and saves in required destination

import pandas as pd
import numpy as np
import os
import argparse
import time
from world2pix import get_pix
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

parser=argparse.ArgumentParser()
parser.add_argument('-l','--list', nargs='+', help='<Required> Spatial range to collect data from : xmin xwidth ymin ywidth', default=[-3,6,-3,6], type=float)
# Use like:
# python transpose_inrange.py -l xmin xmax ymin ymax
args=parser.parse_args()
[xmin,xwidth,ymin,ywidth]=args._get_kwargs()[0][1]
xmax=xmin+xwidth
ymax=ymin+ywidth
print(xmin,xmax,ymin,ymax)



## ---------- Extract Map ------------ ##
img = mpimg.imread('./maps/localization_grid.pgm')
size = img.shape
origin = [-60, -40]
res = 0.05
[h_min, w_min] = get_pix([xmin,ymax], origin, res, size)
[h_max, w_max] = get_pix([xmax,ymin], origin, res, size)
extracted_map = img[h_min:h_max+1,w_min:w_max+1]
plt.imsave('/home/ankur/my_pytorch/srnn-pytorch/maps/0.png', extracted_map, cmap='gray')

## ---------- Extract data ------------##
chunksize=2000000
num_chunks=1
start_chunk=0
date='20121125'
save_dir='/train'
tp = pd.read_csv('/usr/atc-dataset/atc-'+date+'.csv', header=None, iterator=True, chunksize=chunksize, low_memory=False)
data_info=' Chunksize: '+str(chunksize)+'\n No. of chunks: '+str(num_chunks)+'\n Start chunk: '+str(start_chunk)+'\n [xmin,xwidth,ymin,ywidth]: '+str([xmin,xwidth,ymin,ywidth])

i=0
arr_full=np.empty((4,0),dtype=float)

for gm_chunk in tp:
	print('Chunk no.:',i)

	if i<start_chunk:
		i+=1
		continue
	if i==num_chunks+start_chunk:	#use only mentioned number of chunks 
		break
	

	data=gm_chunk.values[:,0:4].transpose()
	#data[0,:]=data[0,:]*100
	data[[2,3],:]/=1000
	
	arr_full=np.append(arr_full,data,axis=1)
	i+=1
print('Done reading chunks')

#Remove columns outside spatial range
arr_full=arr_full[:,np.where(arr_full[2,:]>xmin)[0]]
arr_full=arr_full[:,np.where(arr_full[2,:]<xmax)[0]]
arr_full=arr_full[:,np.where(arr_full[3,:]>ymin)[0]]
arr_full=arr_full[:,np.where(arr_full[3,:]<ymax)[0]]
print('arr_full.shape',arr_full.shape,'\n')

#convert x and y so that it's around 0,0
arr_full[2,:]-=(xmax+xmin)/2
arr_full[3,:]-=(ymax+ymin)/2

#Convert timestamps and personIDs are incrementing natural numbers
arr_full[0,:] = pd.factorize(arr_full[0,:])[0]+1	#timestamp conversion. Crude because all timestamps are considered equal.
arr_full[1,:] = pd.factorize(arr_full[1,:])[0]+1

#Change IDs for pedestrians who went out and came back
org_maxID=max(arr_full[1,:])
maxID=org_maxID
print('original max ID',org_maxID)

ID=1
while(ID<=maxID):
	print('Checking ID',ID)
	pos_arr=np.where(arr_full[1,:]==ID)[0]		#positions where ID is present
	time_arr=arr_full[0,pos_arr]	#time instances where that ID is present
	diff_arr=(np.concatenate((time_arr,[0]),axis=0)-np.concatenate(([0],time_arr),axis=0))[1:-1]	#difference array
	miss_tpos_arr=np.where(diff_arr!=1)[0]	#array of missing positions (time after which it went missing) in the time_arr. These are not positions in actual array butpositions in time_arr
	#print('time_arr',time_arr)
	if(miss_tpos_arr.shape[0]):
		print('missing times-1',time_arr[miss_tpos_arr])
		#sequentially solve for each missing position deaing with only the minimum
		miss_time=min(time_arr[miss_tpos_arr])
		miss_pos=min(np.where(arr_full[0,:]>miss_time)[0])	#actual missing position in array
		pos_change_arr=pos_arr[pos_arr>=miss_pos]	#array of positions where ID needs to be changed
		arr_full[1,pos_change_arr]=maxID+1
		maxID=maxID+1
		print('maxID updated to',maxID)
	ID+=1	#move to next ID
	#ime.sleep(0.5)

#AGAIN Convert timestamps and personIDs are incrementing natural numbers
arr_full[0,:] = pd.factorize(arr_full[0,:])[0]+1	#timestamp conversion. Crude because all timestamps are considered equal.
arr_full[1,:] = pd.factorize(arr_full[1,:])[0]+1


#create directory if doesn't exist
os.makedirs(date+save_dir, exist_ok=True)

np.savetxt(date+save_dir+'/pixel_pos_interpolate.csv',arr_full[[0,1],:].astype(int),delimiter=',',fmt='%i')
with open(date+save_dir+'/pixel_pos_interpolate.csv','a') as f_handle:
    np.savetxt(f_handle,arr_full[[2,3],:],delimiter=',')
open(date+save_dir+'/data_info.txt', "w").write(data_info)