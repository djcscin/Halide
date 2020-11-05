# Configure python environment

pip3 install virtualenv
## Python3.5
virtualenv --python=/usr/local/bin/python3 venv_snpe_tutorial
source venv_snpe_tutorial/bin/activate
pip3 install tensorflow_gpu==1.15.0
pip3 install pydot
pip3 install sklearn
pip3 install numpy
pip3 install sphinx
pip3 install scipy
pip3 install matplotlib
pip3 install skimage
pip3 install protobuf
pip3 install pyyaml

# Configure SNPE environment
# Download https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk
source venv_snpe_tutorial/bin/activate
export SNPE_ROOT=$HOME/snpe-1.40.0.2130
pip3 show tensorflow_gpu
# export TENSORFLOW_DIR=<path to tensorflow installation>
export TENSORFLOW_DIR=$HOME/Workspace/Halide/DL1/venv_snpe_tutorial/lib/python3.5/site-packages
# export LD_LIBRARY_PATH=<path to libpython3.5m.so.1.0>:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/snap/gnome-3-26-1604/100/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
# cp <path to libatomic>/libatomic.so.1 $SNPE_ROOT/lib/aarch64-linux-gcc4.9/
cp /usr/lib/x86_64-linux-gnu/libatomic.so.1 $SNPE_ROOT/lib/aarch64-linux-gcc4.9/
# export ANDROID_NDK_ROOT=<path to NDK>
export ANDROID_NDK_ROOT=$HOME/android-ndk-r20
cd ~/snpe-1.40.0.2130/
source bin/dependencies.sh
source bin/check_python_depends.sh
source bin/envsetup.sh -t $TENSORFLOW_DIR