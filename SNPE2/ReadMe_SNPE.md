# SNPE

```
python $SNPE_ROOT/models/inception_v3/scripts/setup_inceptionv3.py -a $HOME/snpe-1.40.0.2130/models/inception_v3/inception_v3_2016_08_28_frozen
python $SNPE_ROOT/models/inception_v3/scripts/setup_inceptionv3.py -a $HOME/snpe-1.40.0.2130/models/inception_v3/inception_v3_2016_08_28_frozen -r dsp
python $SNPE_ROOT/models/inception_v3/scripts/setup_inceptionv3.py -a $HOME/snpe-1.40.0.2130/models/inception_v3/inception_v3_2016_08_28_frozen -r aip
python $SNPE_ROOT/models/inception_v3/scripts/setup_inceptionv3.py -a $HOME/snpe-1.40.0.2130/models/inception_v3/inception_v3_2016_08_28_frozen -r gpu
python $SNPE_ROOT/models/inception_v3/scripts/setup_inceptionv3.py -a $HOME/snpe-1.40.0.2130/models/inception_v3/inception_v3_2016_08_28_frozen -r all -u
```

# Running the Inception v3 Model

## Run on Linux Host

```
cd models/inception_v3/
snpe-net-run --container dlc/inception_v3_CPU.dlc --input_list data/cropped/raw_list.txt
```

```
python3 $SNPE_ROOT/models/inception_v3/scripts/show_inceptionv3_classifications.py -i data/cropped/raw_list.txt -o output/ -l data/imagenet_slim_labels.txt
```

## Run on Android Target

### Select target architecture

```
export SNPE_TARGET_ARCH=aarch64-android-clang6.0
export SNPE_TARGET_STL=libc++_shared.so
```

### Push binaries to target

```
adb shell "mkdir -p /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/bin"
adb shell "mkdir -p /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib"
adb shell "mkdir -p /data/local/tmp/snpeexample/dsp/lib"

adb push $SNPE_ROOT/lib/$SNPE_TARGET_ARCH/$SNPE_TARGET_STL \
      /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib
adb push $SNPE_ROOT/lib/$SNPE_TARGET_ARCH/*.so \
      /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib
adb push $SNPE_ROOT/lib/dsp/*.so \
      /data/local/tmp/snpeexample/dsp/lib
adb push $SNPE_ROOT/bin/$SNPE_TARGET_ARCH/snpe-net-run \
      /data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/bin
```

### Set up enviroment variables

```
adb shell
export SNPE_TARGET_ARCH=aarch64-android-clang6.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib
export PATH=$PATH:/data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/bin
snpe-net-run -h
```

### Push model data to Android target

```
cd $SNPE_ROOT/models/inception_v3
mkdir data/rawfiles && cp data/cropped/*.raw data/rawfiles/
adb shell "mkdir -p /data/local/tmp/inception_v3"
adb push data/rawfiles /data/local/tmp/inception_v3/cropped
adb push data/target_raw_list.txt /data/local/tmp/inception_v3
adb push dlc/inception_v3_quantized.dlc /data/local/tmp/inception_v3
rm -rf data/rawfiles
```

## Running on Android using CPU Runtime

```
adb shell
export SNPE_TARGET_ARCH=aarch64-android-clang6.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib
export PATH=$PATH:/data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/bin
cd /data/local/tmp/inception_v3
snpe-net-run --container inception_v3_quantized.dlc --input_list target_raw_list.txt
exit
adb pull /data/local/tmp/inception_v3/output output_android
```

```
python3 scripts/show_inceptionv3_classifications.py -i data/target_raw_list.txt \
                                                   -o output_android/ \
                                                   -l data/imagenet_slim_labels.txt
```

## Running on Android using DSP Runtime

ToDo
