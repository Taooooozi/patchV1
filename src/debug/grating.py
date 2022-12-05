import numpy as np
import cv2 as cv
import functools
from ext_signal import *
import matplotlib.pyplot as plt
#TODO: heterogeneous buffers, to save texture memory

def generate_grating(amp, spatialFrequency, temporalFrequency, direction, npixel, c1, c2, fname, time, phase, sharpness, frameRate = 120, ecc = 2.5, buffer_ecc = 0.25, gtype = 'drifting', neye = 2, bar = False, center = np.pi/2, wing = np.pi/2, mask = None, maskData = None, inputLMS = False, genMovie = True):
    """
        spatialFrequency: cycle per degree
        temporalFrequency: Hz
        direction: 0-2pi in rad
        phase: 0-2pi in rad
        a: width of the half image in pixels 
        b: height of the image in pixels 
        c1, c2: the two opposite color in rgb values
        sharpness:  y = A/(1+exp(-sharpness*(x-0.5)) + C, y=x when sharpness = 0
        buffer_ecc: buffering area, to avoid border problems in texture memory accesses
        neye == 2:
            frame from 2 visual fields: (ecc+2*buffer_ecc) x 2(ecc+buffer_ecc+buffer_ecc(unused)) (width x height)
            each has a temporal axis: [-buffer_ecc, ecc], and vertical axis [-ecc-buffer_ecc, ecc+buffer_ecc] in degree
        neye == 1:
            frame from a single visual fields: origin at the center 2(ecc+buffer_ecc) x 2(ecc+buffer_ecc) (width x height)
        mask replace all masked pixels with the masked value
    """
    if np.mod(npixel,2) != 0:
        raise Exception("need even pixel")
    if neye == 1:
        a = npixel
    else:
        a = npixel//2  
        if neye != 2:
            print('neye need to be 1 or 2')
            return
        if np.mod(npixel,2) != 0:
            print('failed: npixel need to be even for neye == 2')
            return

    print(f'{npixel} degree per pixel')

    b = npixel  
    if genMovie:
        FourCC = cv.VideoWriter_fourcc(*'HFYU')
        #FourCC = cv.VideoWriter_fourcc(*'MP4V')
        output = cv.VideoWriter(fname+'.avi', FourCC, frameRate, (npixel,npixel), True)

    if isinstance(time, (list, tuple, np.ndarray)):
        nseq = len(time)
    else:
        nseq = 1
        time = np.array([time])

    if isinstance(amp, (list, tuple, np.ndarray)):
        assert(len(amp) == nseq)
    else:
        amp = np.zeros(nseq) + amp

    if isinstance(spatialFrequency, (list, tuple, np.ndarray)):
        assert(len(spatialFrequency) == nseq)
    else:
        spatialFrequency = np.zeros(nseq) + spatialFrequency

    if isinstance(temporalFrequency, (list, tuple, np.ndarray)):
        assert(len(temporalFrequency) == nseq)
    else:
        temporalFrequency = np.zeros(nseq) + temporalFrequency

    if isinstance(direction, (list, tuple, np.ndarray)):
        assert(len(direction) == nseq)
    else:
        direction = np.zeros(nseq) + direction
        
    if isinstance(phase, (list, tuple, np.ndarray)):
        assert(len(phase) == nseq)
    else:
        phase = np.zeros(nseq) + phase

    if isinstance(sharpness, (list, tuple, np.ndarray)):
        assert(len(sharpness) == nseq)
    else:
        sharpness = np.zeros(nseq) + sharpness

    if mask is not None:
        mask = np.reshape(np.repeat(mask, 3), (nseq, b,a,3))
        if maskData is None:
            raise Exception('mask data is not provided')
        else:
            if isinstance(maskData, (list, tuple, np.ndarray)):
                if isinstance(maskData, (list, tuple)):
                    maskData = np.array(list(maskData))

                if not np.array([i == j for i, j in zip(maskData.shape, (b,a,3))]).all() and len(maskData) != 3:
                    raise Exception('maskData shape does not match with stimulus size')
                else:
                    if np.array([i == j for i, j in zip(maskData.shape, (b,a,3))]).all():
                        maskData = np.tile(maskData, (nseq, 1))
                    elif maskData.size == 3:
                        maskData = np.tile(maskData, (nseq, b, a, 1)) 
                    else:
                        raise Exception('maskData takes array of shape (b,a,3) or (3,)')
            else:
                raise Exception('maskData takes array of shape (b,a,3) or (3,)')

    ########### VIDEO encodes as BGR: 
    
    if not inputLMS: # rgb->bgr
        c1_LMS = np.matmul(sRGB2LMS, inverse_sRGB_gamma(c1.reshape(3,1)))
        c2_LMS = np.matmul(sRGB2LMS, inverse_sRGB_gamma(c2.reshape(3,1)))
        mean_value = (c1_LMS+c2_LMS)/2
        c1 = c1[::-1]
        c2 = c2[::-1]
    else:
        c1_sRGB = apply_sRGB_gamma(np.matmul(LMS2sRGB, c1))
        c2_sRGB = apply_sRGB_gamma(np.matmul(LMS2sRGB, c2))
        print(f'crest in sRGB: {c1_sRGB}')
        print(f'valley in sRGB: {c2_sRGB}')
        # if not (c1_sRGB<=1).all() or not (c1_sRGB>=0).all() or not (c2_sRGB<=1).all() or not (c2_sRGB>=0).all():
        #     raise Exception(f'crest and valley in LMS is out of the sRGB space')
        mean_value = (c1+c2)/2

    c1 = np.reshape(c1,(1,3))
    c2 = np.reshape(c2,(1,3))
    control = np.zeros(3)
    for i in range(3):
        if c1[0,i] >= c2[0,i]:
           control[i] = 1
        else: 
           control[i] = -1

    if neye == 1:
        X, Y = np.meshgrid(np.linspace(-1,1,a)*(ecc+buffer_ecc)*np.pi/180, np.linspace(-1,1,b)*(ecc+buffer_ecc)*np.pi/180)
        deg2pixel = npixel / (2*(ecc+buffer_ecc))
    else:
        X, Y = np.meshgrid((np.linspace(0,1,a)*(ecc+2*buffer_ecc)-buffer_ecc)*np.pi/180,np.linspace(-1,1,b)*(ecc+2*buffer_ecc)*np.pi/180)
        deg2pixel = npixel / (2*(ecc+2*buffer_ecc))

    print(f'{1/deg2pixel} degree per pixel')

    print(f'ecc = {ecc}, buffer_ecc = {buffer_ecc}')
    f = open(fname + '.bin', 'wb')
    np.array([-1]).astype('i4').tofile(f) 
    nFrame = np.sum(np.round(np.ceil(frameRate*time))).astype(int)
    print(nFrame)
    np.array([nFrame, npixel, npixel], dtype='i4').tofile(f)
    mean_value.astype('f4').tofile(f) # init_luminance
    np.array([buffer_ecc, ecc], dtype='f4').tofile(f)
    np.array([neye]).astype('u4').tofile(f)

    for i in range(nseq):
        t = time[i]
        nstep = int(np.round(frameRate*t))
        if not nstep == frameRate*t:
            nstep = int(np.ceil(frameRate*t))
            print(f'adjusted to {nstep} frames in total')
        else:
            print(f'exact {nstep} frames in total')

        if np.mod(nstep,2) != 0 and gtype == 'rotating':
            raise Exception(f'need even time step, current: {nstep}')

        radTF = temporalFrequency[i]*2*np.pi
        radSF = spatialFrequency[i]*180/np.pi*2*np.pi
        s = sharpness[i]
        print(f'sharpness={s}')
        #@logistic(s)
        def grating(amp, radTF, radSF, direction, a, b, c1, c2, control, s, phase, t, X, Y, bar, center = 0, wing = 0):
            return sine_wave(amp, radTF, radSF, direction, a, b, c1, c2, control, s, phase, t, X, Y, bar, center, wing)

        if gtype == 'rotating':
            half = nstep//2 
            dl = np.linspace(0,np.pi/4,half)
            dr = np.linspace(0,np.pi/4,half)
            dd = np.hstack((dl, np.flip(dr)))
            
        dt = 1.0/frameRate
        #for it in range(1):
        if gtype not in ('drifting','rotating'):
            raise Exception(f'gtype {gtype} not implemented')

        LMS_seq = np.empty((3, npixel, npixel), dtype=float)
        for it in range(nstep):
            print('it:',it)
            t = it * dt
            if neye == 1:
                if gtype == 'rotating':
                    data = grating(amp[i], radTF, radSF, direction[i]-dd[it], a, b, c1, c2, control, s, phase[i], t, X, Y, bar, center, wing)
                if gtype == 'drifting':
                    data = grating(amp[i], radTF, radSF, direction[i], a, b, c1, c2, control, s, phase[i], t, X, Y, bar, center, wing)

                if mask is not None:
                    data[mask[i,:,:,:]] = maskData[i, mask[i,:,:,:]]
            else:
                if gtype == 'rotating':
                    dataL = grating(amp[i], radTF, radSF, direction[i]-dd[it], a, b, c1, c2, control, s, phase[i], t, X, Y, bar, center, wing)
                    dataR = grating(amp[i], radTF, radSF, direction[i]+dd[it], a, b, c1, c2, control, s, phase[i], t, X, Y, bar, center, wing)
                if gtype == 'drifting':
                    dataL = grating(amp[i], radTF, radSF, direction[i], a, b, c1, c2, control, s, phase[i], t, X, Y, bar, center, wing)
                    dataR = grating(amp[i], radTF, radSF, direction[i], a, b, c1, c2, control, s, phase[i], t, X, Y, bar, center, wing)

                if mask is not None:
                    assert(dataL.shape[0] == b)
                    assert(dataL.shape[1] == a)
                    assert(dataL.shape[2] == 3)
                    dataL[mask[i,:,:,:]] = maskData[i, mask[i,:,:,:]]
                    dataR[mask[i,:,:,:]] = maskData[i, mask[i,:,:,:]]
                data = np.concatenate((dataL,dataR), axis = 1)

            if inputLMS:
                # lms->rgb->bgr
                _LMS = data.reshape(npixel*npixel,3).T
                assert((_LMS>=0).all())
                assert((_LMS<=1).all())
                _sRGB = apply_sRGB_gamma(np.matmul(LMS2sRGB, _LMS))
                if (_sRGB<0).any() or (_sRGB>1).any():
                    print('sRGB space is not enough to represent the color')
                    #print(f'{c1, c2}')
                    print(f'{np.min(_sRGB, axis = 1), np.max(_sRGB, axis = 1)}')
                    pick = _sRGB > 1
                    _sRGB[pick] = 1
                    pick = _sRGB < 0
                    _sRGB[pick] = 0
                pixelData = np.round(_sRGB*255).T.reshape(npixel,npixel,3)[:,:,::-1].astype('uint8')
                LMS_seq = _LMS.reshape((3,npixel,npixel))
            else: # input is sRGB
                pixelData = np.round(data*255).reshape(npixel,npixel,3).astype('uint8')
                # bgr->rgb->lms
                LMS_seq = np.matmul(sRGB2LMS, inverse_sRGB_gamma(data[:,:,::-1].reshape((npixel*npixel,3)).T)).reshape((3,npixel,npixel))

            if genMovie:
                output.write(pixelData)

            if it == 0:
                # fig = plt.figure()
#                 ax = fig.add_subplot(131)
#                 ax.imshow(data[:,:,0], cmap = 'Greys')
#                 ax = fig.add_subplot(132)
#                 ax.imshow(data[:,:,1], cmap = 'Greys')
#                 ax = fig.add_subplot(133)
#                 ax.imshow(data[:,:,2], cmap = 'Greys')
#                 fig.savefig(fname + '.png')
#                 plt.close(fig)
#                 print(data.shape)
#                 cv.imwrite(fname + '_1.png', pixelData)
                
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.imshow(data[:,:,0], cmap = 'Greys')
                plt.xticks([])  # 去掉x轴
                plt.yticks([])  # 去掉y轴
                plt.axis('off') 
                # plt.margins(0, 0)
                # plt.spines['top'].set_visible(False)
                # plt.spines['right'].set_visible(False)
                # plt.spines['bottom'].set_visible(False)
                # plt.spines['left'].set_visible(False)
                fig.savefig(fname + '.png',bbox_inches='tight')
                plt.close(fig)
            #pixelData = np.reshape(np.round(data*255), (b,a,3))
            #cv.imshow('linear', pixelData)
            #cv.waitKey(0)
            #pixelData = adjust_gamma(pixelData, gamma = 2.2)
            #cv.imshow('gamma', pixelData)
            #cv.waitKey(0)
            LMS_seq.astype('f4').tofile(f)

    f.close()
    if genMovie:
        output.release()
        cv.destroyAllWindows()
    return


def sine_wave(amp, radTF, radSF, direction, a, b, c1, c2, control, sharpness, phase, t, X, Y, bar, center, wing):
    phi = (np.cos(direction)*X + np.sin(direction)*Y)*radSF - radTF*t
    if bar:
        #floor_phi = np.floor(phi/(2*np.pi)).astype(int)
        #phi_tmp = phi-floor_phi*2*np.pi
        pick = np.abs(phi + (phase + center)) > wing
        phi[pick] = -phase
    rel_color = np.reshape(1+amp*np.sin(phi + phase), (a*b,1))/2
    if sharpness != 1:
        if sharpness > 0:
            exp_half = np.exp(sharpness/2)+1
            A = np.power(exp_half,2)/(np.exp(sharpness)-1)
            C = exp_half/(1-np.exp(sharpness))
            rel_color = A/(1.0 + np.exp(-sharpness*(rel_color-0.5))) + C
        else:
            rel_color[rel_color > 0.5] = 1
            rel_color[rel_color < 0.5] = 0
        
    assert((rel_color <= 1.0).all())
    assert((rel_color >= 0.0).all())
    color = np.matmul(np.ones((a*b,1)), c1) + np.matmul(rel_color, (c2-c1))
    # for i in range(3):
    #     # print(control[i])
    #     if control[i] > 0:
    #         if not((color[:,i] <= c1[0,i]).all()) or not((color[:,i] >= c2[0,i]).all()):
    #             print(color[:,i].min(),color[:,i].max())
    #             print(c1[0,i],c2[0,i])
    #         assert((color[:,i] <= c1[0,i]).all())
    #         assert((color[:,i] >= c2[0,i]).all())
    #     else:
    #         assert((color[:,i] >= c1[0,i]).all())
    #         assert((color[:,i] <= c2[0,i]).all())
        
    return color.reshape((b,a,3))


