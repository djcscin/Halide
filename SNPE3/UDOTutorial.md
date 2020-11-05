# UDO Tutorial

$SNPE_ROOT/examples/NativeCpp/UdoExample/Softmax

## Package Generation

### Dependency
```
pip install Mako
pip install six
```

```
export SNPE_UDO_ROOT=$SNPE_ROOT/share/SnpeUdo
snpe-udo-package-generator -p $SNPE_ROOT/examples/NativeCpp/UdoExample/Softmax/config/Softmax.json -o $SNPE_ROOT/models/inception_v3/
```

## Framework model Conversion to a DLC
```
snpe-tensorflow-to-dlc --input_network $HOME/snpe-1.40.0.2130/models/inception_v3/inception_v3_2016_08_28_frozen.pb --input_dim 'input' 1,299,299,3 --out_node InceptionV3/Predictions/Reshape_1 --output_path $SNPE_ROOT/models/inception_v3/dlc/inception_v3_udo.dlc --allow_unconsumed_nodes --udo $SNPE_ROOT/examples/NativeCpp/UdoExample/Softmax/config/Softmax.json
```

## Package Implementations

### CPU Implementations
cp -f $SNPE_ROOT/examples/NativeCpp/UdoExample/Softmax/src/CPU/SoftmaxImplLibCpu.cpp $SNPE_ROOT/models/inception_v3/SoftmaxUdoPackage/jni/src/CPU/
cp -f $SNPE_ROOT/examples/NativeCpp/UdoExample/Softmax/src/reg/SoftmaxUdoPackageCpuImplValidationFunctions.cpp $SNPE_ROOT/models/inception_v3/SoftmaxUdoPackage/jni/src/reg/

## Package Compilation

### x86 Host Compilation
cd $SNPE_ROOT/models/inception_v3/SoftmaxUdoPackage
make cpu_x86

cd $SNPE_ROOT/models/inception_v3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SNPE_ROOT/models/inception_v3/SoftmaxUdoPackage/libs/x86-64_linux_clang/
snpe-net-run --container dlc/inception_v3_udo.dlc --input_list data/cropped/raw_list.txt --udo_package_path SoftmaxUdoPackage/libs/x86-64_linux_clang/libUdoSoftmaxUdoPackageReg.so