import subprocess
import numpy as np
from scipy import io

def get_lens_shading_map_dng(path):
    exiftool = subprocess.Popen(("exiftool", "-opcodelist2", path, "-b"), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    header = subprocess.check_output(("od", "--format=u4", "-v", "-An", "--endian=big"), stdin=exiftool.stdout).split()
    header = np.array(header).astype(np.int)
    channels = header[0]
    header = header[1:].reshape((channels,-1)).transpose()
    top = header[4,:]
    left = header[5,:]
    height = header[12,0]
    width = header[13,0]
    exiftool.stdout.close()

    exiftool = subprocess.Popen(("exiftool", "-opcodelist2", path, "-b"), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    lsc_map = subprocess.check_output(("od", "-f", "-v", "-An", "--endian=big"), stdin=exiftool.stdout).split()
    lsc_map = np.array(lsc_map).astype(np.float32)
    lsc_map = lsc_map[1:].reshape((channels,-1)).transpose()
    lsc_map = lsc_map[23:,:].reshape(height, width, channels)
    exiftool.stdout.close()

    lsc = np.empty((height*2, width*2))
    lsc[top[0]::2, left[0]::2] = lsc_map[:,:,0]
    lsc[top[1]::2, left[1]::2] = lsc_map[:,:,1]
    lsc[top[2]::2, left[2]::2] = lsc_map[:,:,2]
    lsc[top[3]::2, left[3]::2] = lsc_map[:,:,3]

    lsc_map[:,:,0] = lsc[0::2,0::2]
    lsc_map[:,:,1] = lsc[0::2,1::2]
    lsc_map[:,:,2] = lsc[1::2,0::2]
    lsc_map[:,:,3] = lsc[1::2,1::2]

    return lsc_map

for dng in [
    '5a9e_20150405_165352_614',
    '6G7M_20150307_175028_814',
    'IMG_20200508_202014675',
    'IMG_20201009_123817328'
    ]:
    lsmap = get_lens_shading_map_dng('images/%s.dng'%dng)
    io.savemat('images/%s.mat'%dng, {'lens_shading_map' : lsmap})
    mat = io.loadmat('images/%s.mat'%dng)
    print(mat['lens_shading_map'].shape, mat['lens_shading_map'][0][0][0])
