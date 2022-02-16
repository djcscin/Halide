
CXX = g++

CXX_ARM = $(HOME)/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++
ARM_DIR = /data/local/tmp

GENERATOR_DEPS     = ${HALIDE_ROOT}/tools/GenGen.cpp
GENERATOR_OUTPUTS  = o,h
CXX_FLAGS          = -std=c++1z -fno-rtti
OPT_FLAGS          = -O3
INCLUDES           = -I${HALIDE_ROOT}/include -Ibin
LD_FLAGS           = -lHalide -ldl -lpthread -lz
LIBS               = -L${HALIDE_ROOT}/lib
IMAGE_IO_FLAGS     = -I${HALIDE_ROOT}/tools -ljpeg `libpng-config --cflags --ldflags`
LD_FLAGS_ARM       = -static-libstdc++ -llog
LIBS_ARM           = -L${HALIDE_ROOT_ARM}/lib -Llib
IMAGE_IO_FLAGS_ARM = -I${HALIDE_ROOT_ARM}/tools -DHALIDE_NOPNG -DHALIDE_NO_PNG -DHALIDE_NO_JPEG

ifndef AUTO_SCHEDULE
	AUTO_SCHEDULE=false
endif
ifeq ($(AUTO_SCHEDULE), true)
	ifndef AUTO_SCHEDULER
		AUTO_SCHEDULER=mullapudi2016
	endif
	AUTO_SCHEDULER_PARS=auto_schedule=true -p $(HALIDE_ROOT)/lib/libautoschedule_$(AUTO_SCHEDULER).so
	ifdef MACHINE_PARAMS
		AUTO_SCHEDULER_PARS+=machine_params=$(MACHINE_PARAMS)
	endif
endif

TARGET:=arm-64-android
ifdef DESKTOP
	ifeq ($(DESKTOP), true)
		TARGET:=host
	endif
endif

# HVX hvx_128 ou hvx (a partir da versão 11 do Halide)
# HVX para usar VTCM hvx_128-hvx_v65 ou hvx-hvx_v65
# OpenCL opencl
# Cuda cuda - indisponível para o celular
ifdef ACCELERATOR
	TARGET:=$(TARGET)-$(ACCELERATOR)
endif

# Para usar o profiler no telefone execute o seguinte comando:
#  adb logcat -s halide &
ifdef PROFILE
	ifeq ($(PROFILE), true)
		TARGET:=$(TARGET)-profile
	endif
endif

ifndef RUNS
	RUNS=1
endif

all: test

bin/sharp.generator: src/HalideSharp.cpp
	@mkdir -p $(@D)
	@$(CXX) $^ $(GENERATOR_DEPS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) -o $@

bin/sharp.o: bin/sharp.generator
	@mkdir -p $(@D)
	@$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -g sharp -f sharp target=$(TARGET) $(AUTO_SCHEDULER_PARS)

bin/sharp: src/sharp.cpp bin/sharp.o
	@mkdir -p $(@D)
	@$(CXX_ARM) $^ $(OPT_FLAGS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS_ARM) $(LIBS_ARM) $(IMAGE_IO_FLAGS_ARM) -o $@

bin/sharp_desktop: src/sharp.cpp bin/sharp.o
	@mkdir -p $(@D)
	@$(CXX) $^ $(OPT_FLAGS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) $(IMAGE_IO_FLAGS) -o $@

PATH_IMAGE_IN=../images/final2016_blurred.ppm
PATH_IMAGE_OUT=images_output/final2016_blurred_sharped.ppm
PATH_ARM_IN=$(ARM_DIR)/$(shell basename $(PATH_IMAGE_IN))
PATH_ARM_OUT=$(ARM_DIR)/$(shell basename $(PATH_IMAGE_OUT))
test: bin/sharp
	@mkdir -p images_output
	@adb push $(PATH_IMAGE_IN) $(PATH_ARM_IN)
	@adb push bin/sharp $(ARM_DIR)/sharp
	@adb shell chmod +x $(ARM_DIR)/sharp
	@for i in $(shell seq $(RUNS)); do \
		adb shell $(ARM_DIR)/sharp $(PATH_ARM_IN) 4 $(PATH_ARM_OUT); \
	done
	@adb pull $(PATH_ARM_OUT) $(PATH_IMAGE_OUT)

# LIBS_HEXAGON=$(HOME)/Qualcomm/Hexagon_SDK/4.3.0.0/tools/HALIDE_Tools/2.3.03/Halide/lib/host
# HEXAGON_SIM_REMOTE=$(HOME)/Qualcomm/Hexagon_SDK/4.3.0.0/tools/HALIDE_Tools/2.3.03/Halide/lib/v62/hexagon_sim_remote
LIBS_HEXAGON=$(HOME)/Hexagon/bin/host
HEXAGON_SIM_REMOTE=$(HOME)/Hexagon/bin/v65/hexagon_sim_remote
HEXAGON_TOOLS=$(HOME)/Qualcomm/Hexagon_SDK/4.3.0.0/tools/HEXAGON_Tools/8.4.11/Tools
test_desktop: bin/sharp_desktop
	@mkdir -p images_output
	@unset HL_HEXAGON_SIM_CYCLES
	@unset HL_HEXAGON_SIM_STATS
	@for i in $(shell seq $(RUNS)); do \
		HL_HEXAGON_SIM_REMOTE=$(HEXAGON_SIM_REMOTE) LD_LIBRARY_PATH=$(LIBS_HEXAGON):$(HEXAGON_TOOLS)/lib/iss:$(LD_LIBRARY_PATH) \
		$< $(PATH_IMAGE_IN) 4 $(PATH_IMAGE_OUT); \
	done

clean:
	@rm -rf bin images_output
