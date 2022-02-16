
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

ifndef SCHEDULER
	SCHEDULER=0
endif
ifeq ($(SCHEDULER), 0)
	AUTO_SCHEDULE=true
endif
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

ifndef PARALLEL_SIZE
	PARALLEL_SIZE=128
endif

TARGET:=arm-64-android
ifdef DESKTOP
	ifeq ($(DESKTOP), true)
		TARGET:=host
	endif
endif

# Para usar o profiler no telefone execute o seguinte comando:
#  adb logcat -s halide &
ifdef PROFILE
	ifeq ($(PROFILE), true)
		TARGET:=$(TARGET)-profile
	endif
endif

ifndef RUNS
	RUNS=5
endif

all: test

bin/halide.generator: src/halide.cpp
	@mkdir -p $(@D)
	@$(CXX) $^ $(GENERATOR_DEPS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) -o $@

bin/halide_func.o: bin/halide.generator
	@mkdir -p $(@D)
	@$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -g halide_func -f halide_func target=$(TARGET) parallel_size=$(PARALLEL_SIZE) scheduler=$(SCHEDULER) $(AUTO_SCHEDULER_PARS)

bin/process: src/process.cpp bin/halide_func.o
	@mkdir -p $(@D)
	@$(CXX_ARM) $^ $(OPT_FLAGS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS_ARM) $(LIBS_ARM) $(IMAGE_IO_FLAGS_ARM) -o $@

bin/process_desktop: src/process.cpp bin/halide_func.o
	@mkdir -p $(@D)
	@$(CXX) $^ $(OPT_FLAGS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) $(IMAGE_IO_FLAGS) -o $@

PATH_IMAGE_IN=../images/final2016_blurred.ppm
PATH_IMAGE_OUT=images_output/final2016_blurred_output.ppm
PATH_ARM_IN=$(ARM_DIR)/$(shell basename $(PATH_IMAGE_IN))
PATH_ARM_OUT=$(ARM_DIR)/$(shell basename $(PATH_IMAGE_OUT))
test: bin/process
	@mkdir -p images_output
	@adb push $(PATH_IMAGE_IN) $(PATH_ARM_IN)
	@adb push bin/process $(ARM_DIR)/process
	@adb shell chmod +x $(ARM_DIR)/process
	@for i in $(shell seq $(RUNS)); do \
		adb shell $(ARM_DIR)/process $(PATH_ARM_IN) $(PATH_ARM_OUT) 30 10 3; \
	done
	@adb pull $(PATH_ARM_OUT) $(PATH_IMAGE_OUT)

test_desktop: bin/process_desktop
	@mkdir -p images_output
	@for i in $(shell seq $(RUNS)); do \
		$< $(PATH_IMAGE_IN) $(PATH_IMAGE_OUT) 30 10 3; \
	done

clean:
	@rm -rf bin images_output
