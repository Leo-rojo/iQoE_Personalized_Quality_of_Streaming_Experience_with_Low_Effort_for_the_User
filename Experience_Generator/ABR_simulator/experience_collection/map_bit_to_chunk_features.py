import numpy as np
import csv
import os
import re

th_exp=np.load('./exp_th.npy')
bb_exp=np.load('./exp_bb.npy')
mpc_exp=np.load('./exp_mpc.npy')

#format
#[st,bit][st,bit][st,bit]...30
def map_bit_to_features(bit,chunk_pos):
    VIDEO_BIT_RATE = [0.235, 0.375, 0.560, 0.750, 1.050, 1.750, 2.350, 3, 4.3, 5.8, 8.1, 11.6, 16.8]
    for file in os.listdir('./chunk_features'):
        if file[-3:]=='npy':
            bit_name=float(file.split('_')[-1].replace('k.npy', ''))
            if bit*1e3==bit_name:
                file_feat=np.load('./chunk_features/'+file)
                #1 if best otherwise 0
                if bit_name==16800:
                    isb=1
                else:
                    isb=0
                rep=VIDEO_BIT_RATE.index(bit)
                ff=file_feat[chunk_pos]
                file_feat_converted=list(map(float, ff))
                #from file feat take pos
                return file_feat_converted + [rep]+[isb]

#th
th_exp_ordered=[]
for exp_nr in th_exp:
    th_exp_ordered_temp=[]
    for nr_chunk in range(len(exp_nr)):
        rebuff=exp_nr[nr_chunk][0]
        feat=map_bit_to_features(exp_nr[nr_chunk][1],nr_chunk)
        th_exp_ordered_temp.append([feat[-2],rebuff,feat[3],feat[0],feat[1],feat[2],feat[-1],feat[4],feat[5],feat[6]])   #put features as WIV features: representation_index	rebuffering_duration	video_bitrate	chunk_duration	chunk_size	qp	framerate	width	height	is_best	psnr	ssim	vmaf
    th_exp_ordered.append(th_exp_ordered_temp)
#bb
bb_exp_ordered=[]
for exp_nr in bb_exp:
    bb_exp_ordered_temp=[]
    for nr_chunk in range(len(exp_nr)):
        rebuff=exp_nr[nr_chunk][0]
        feat=map_bit_to_features(exp_nr[nr_chunk][1],nr_chunk)
        bb_exp_ordered_temp.append([feat[-2],rebuff,feat[3],feat[0],feat[1],feat[2],feat[-1],feat[4],feat[5],feat[6]])   #put features as WIV features: representation_index	rebuffering_duration	video_bitrate	chunk_duration	chunk_size	qp	framerate	width	height	is_best	psnr	ssim	vmaf
    bb_exp_ordered.append(bb_exp_ordered_temp)
#mpc
mpc_exp_ordered=[]
for exp_nr in mpc_exp:
    mpc_exp_ordered_temp=[]
    for nr_chunk in range(len(exp_nr)):
        rebuff=exp_nr[nr_chunk][0]
        feat=map_bit_to_features(exp_nr[nr_chunk][1],nr_chunk)
        mpc_exp_ordered_temp.append([feat[-2],rebuff,feat[3],feat[0],feat[1],feat[2],feat[-1],feat[4],feat[5],feat[6]])   #put features as WIV features: representation_index	rebuffering_duration	video_bitrate	chunk_duration	chunk_size	qp	framerate	width	height	is_best	psnr	ssim	vmaf
    mpc_exp_ordered.append(mpc_exp_ordered_temp)

total_experiences=th_exp_ordered+bb_exp_ordered+mpc_exp_ordered

np.save('experiences_with_features',total_experiences)


