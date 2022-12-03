import numpy as np
import os
import sys
from readPatchOutput import *


def get_FreqComp(data, ifreq):
    if ifreq == 0:
        raise Exception('just use mean for zero comp')
    ndata = len(data)
    Fcomp = np.sum(data * np.exp(-2*np.pi*1j*ifreq*np.arange(ndata)/ndata))/ndata
    return np.array([np.abs(Fcomp)*2, np.angle(Fcomp, deg = True)])

def generate_file(output_suffix0,output_suffix,res_suffix,conLGN_suffix,conV1_suffix,res_fdr,setup_fdr,data_fdr,TF,ori,bin_prefix):
    res_suffix = "_" + res_suffix
    conLGN_suffix = "_" + conLGN_suffix
    conV1_suffix = "_" + conV1_suffix
    _output_suffix = "_" + output_suffix
    res_fdr = res_fdr + '/'
    setup_fdr = setup_fdr + '/'
    data_fdr = data_fdr + '/'
    LGN_V1_sFn = setup_fdr + "LGN_V1_sList" + conLGN_suffix + ".bin"
    LGN_V1_idFn = setup_fdr + "LGN_V1_idList" + conLGN_suffix + ".bin"
    LGN_spFn = data_fdr + "LGN_sp" + _output_suffix
    parameterFn = data_fdr + "patchV1_cfg" +_output_suffix + ".bin"
    
    LGN_V1_s = readLGN_V1_s0(LGN_V1_sFn)
    LGN_V1_ID, nLGN_V1 = readLGN_V1_ID(LGN_V1_idFn)
    LGN_V1_idFn = setup_fdr + "LGN_V1_idList" + conLGN_suffix + ".bin"
    LGN_V1_ID, nLGN_V1 = readLGN_V1_ID(LGN_V1_idFn) # size为5120    
    
    filename1 = data_fdr  + 'sample_spikeCount_' + output_suffix + '_' + str(ori) + '.bin'
    filename2 = data_fdr + bin_prefix + output_suffix + '_' + str(ori) + '.bin'  
    
    with open(filename1) as f:
        sampleSize = np.fromfile(f, 'u4', 1)[0] #一共5120
        sample_t0, sample_t1 = np.fromfile(f, 'f4', 2)#t0是开始时间，t1是结束时间，我一共跑了1s
        nt = np.fromfile(f, 'u4', 1)[0]
        nLGN = np.fromfile(f, 'u4', 1)[0]
        LGN_spike_time = np.fromfile(f, 'u4', nLGN*nt)
        sampleID = np.fromfile(f, 'u4', sampleSize)#id排序是顺序的
        sample_spikeCount = np.fromfile(f, 'u4', sampleSize)
    LGN_spike_time = LGN_spike_time.reshape((nt,nLGN)).T
    
    V1_input_pre = np.zeros((sampleSize,nt))
    for i in range(sampleSize):
        tmp_id = sampleID[i]
        nLGN_ = LGN_V1_ID[tmp_id].size
        for j in range(nLGN_):
            V1_input_pre[i] += LGN_spike_time[LGN_V1_ID[tmp_id][j]]*LGN_V1_s[tmp_id][j]
            
    tTF = 1000/TF
    dt = (sample_t1-sample_t0)/nt
    cycle = int(tTF/dt)
    F0_ls = []
    F1_ls = []
    for tsp in V1_input_pre:
        F0 = np.mean(tsp)
        tsp = np.mean(tsp.reshape((cycle,-1)),axis=1)
        tsp = np.mean(tsp.reshape((-1,25)),axis=0)
        F1 = get_FreqComp(tsp, 1)[0]
        F0_ls.append(F0)
        F1_ls.append(F1)
    F0_ls = np.array(F0_ls).astype('f4')
    F1_ls = np.array(F1_ls).astype('f4')
    
    with open(filename2,'wb') as f:
        np.array([sampleSize]).astype('u4').tofile(f)
        np.array([sample_t0, sample_t1]).astype('f4').tofile(f)
        sampleID.astype('u4').tofile(f)
        sample_spikeCount.astype('u4').tofile(f)
        F0_ls.astype('f4').tofile(f)
        F1_ls.astype('f4').tofile(f)
        
if __name__ == "__main__":
    if len(sys.argv) < 12:
        print(sys.argv)
        raise Exception('not enough argument for generate_file(output_suffix0,output_suffix,res_suffix,conLGN_suffix,conV1_suffix,res_fdr,setup_fdr,data_fdr,TF,ori,bin_prefix)')
    else:
        output_suffix0 = sys.argv[1]
        print(output_suffix0)
        output_suffix = sys.argv[2]
        print(output_suffix)
        res_suffix = sys.argv[3]
        print(res_suffix)
        conLGN_suffix = sys.argv[4]
        print(conLGN_suffix)
        conV1_suffix = sys.argv[5]
        print(conV1_suffix)
        res_fdr = sys.argv[6]
        print(res_fdr)
        setup_fdr = sys.argv[7]
        print(setup_fdr)
        data_fdr = sys.argv[8]
        print(data_fdr)
        TF = float(sys.argv[9])
        print(TF)
        ori = sys.argv[10]
        print(ori)
        bin_prefix = sys.argv[11]
        print(bin_prefix)
        
        generate_file(output_suffix0,output_suffix,res_suffix,conLGN_suffix,conV1_suffix,res_fdr,setup_fdr,data_fdr,TF,ori,bin_prefix)
    
    