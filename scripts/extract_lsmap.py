import subprocess
import numpy as np
from scipy import io

def get_lens_shading_map_dng(path):
    exiftool = subprocess.Popen(("exiftool", "-opcodelist2", path, "-b"), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    header = subprocess.check_output(("od", "--format=u4", "-v", "-An", "--endian=big"), stdin=exiftool.stdout).split()
    channels = int(header[0])
    height = int(header[13])
    width = int(header[14])
    exiftool.stdout.close()

    exiftool = subprocess.Popen(("exiftool", "-opcodelist2", path, "-b"), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    lsc_map = subprocess.check_output(("od", "-f", "-v", "-An", "--endian=big"), stdin=exiftool.stdout).split()
    lsc_map = np.array([float(i.decode("utf-8").replace(',','.')) for i in lsc_map])
    lsc_map = lsc_map[1:].reshape((channels,-1)).transpose()
    lsc_map = lsc_map[23:,:].reshape(height, width, channels)
    exiftool.stdout.close()

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
