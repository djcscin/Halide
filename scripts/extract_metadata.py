import sys
import rawpy

def print_vec(vec):
    print(*["{:.12f}".format(i) for i in vec])

def print_metadata(path_raw, path_txt):
    original_stdout = sys.stdout
    with rawpy.imread(path_raw) as raw, open(path_txt, 'w') as txt:
        sys.stdout = txt
        print_vec(raw.camera_whitebalance[:3])
        print_vec(raw.color_matrix[:3,:3].flatten())
    sys.stdout = original_stdout

for dng in [
    '5a9e_20150405_165352_614',
    '6G7M_20150307_175028_814',
    'IMG_20200508_202014675',
    'IMG_20201009_123817328'
    ]:
    print_metadata('images/%s.dng'%dng, 'images/%s.txt'%dng)
