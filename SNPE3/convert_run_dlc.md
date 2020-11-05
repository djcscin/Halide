# SNPE

```
mkdir dlc
snpe-tensorflow-to-dlc --input_network ../SNPE2/models/tf_mlp.pb --input_dim dense_input "1,784" --out_node "dense_1/Softmax" --output_path dlc/tf_mlp.dlc
snpe-tensorflow-to-dlc --input_network ../SNPE2/models/tf1.pb --input_dim conv2d_input "1,28,28,1" --out_node "dense/Softmax" --output_path dlc/tf1.dlc --allow_unconsumed_nodes
snpe-tensorflow-to-dlc --input_network ../SNPE2/models/tf2.pb --input_dim input_1 "1,32,32,3" --out_node "dense_1/Softmax" --output_path dlc/tf2.dlc --allow_unconsumed_nodes
```

# Running the Inception v3 Model

## Run on Linux Host

```
snpe-net-run --container dlc/tf1.dlc --input_list mnist/data/image_list.txt
```

```
for i in $(seq 0 3); do python mnist/scripts/interpretRawLeNetOutput.py output/Result_$i/dense/Softmax:0.raw; done
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
adb shell "mkdir -p /data/local/tmp/mnist"
adb push mnist/data/28x28x1_raw /data/local/tmp/mnist/data/28x28x1_raw
adb push mnist/data/image_list.txt /data/local/tmp/mnist/data
adb push dlc/tf1.dlc /data/local/tmp/mnist
```

## Running on Android using CPU Runtime

```
adb shell
export SNPE_TARGET_ARCH=aarch64-android-clang6.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib
export PATH=$PATH:/data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/bin
cd /data/local/tmp/mnist
snpe-net-run --container tf1.dlc --input_list data/image_list.txt
exit
adb pull /data/local/tmp/mnist/output output_android
```

```
for i in $(seq 0 3); do python mnist/scripts/interpretRawLeNetOutput.py output_android/Result_$i/dense/Softmax:0.raw; done
```

## Running on Android using DSP Runtime

ToDo