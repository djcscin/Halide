
GENERATOR_DEPS     = ${HALIDE_ROOT}/tools/GenGen.cpp
GENERATOR_OUTPUTS  = o,h,stmt,schedule
CXX_FLAGS          = -std=c++1z -fno-rtti
OPT_FLAGS          = -O3
INCLUDES           = -I${HALIDE_ROOT}/include -I../../include -Ihalide
INCLUDES_TEST      = -I${HALIDE_ROOT}/include -I../../include -Ibin -Itest -I${HALIDE_ROOT}/tools
CXX                = g++
LD_FLAGS           = -lHalide -ldl -lpthread -lz
LIBS               = -L${HALIDE_ROOT}/lib

ifeq ($(DESKTOP), true)
CXX_TEST       = $(CXX)
LD_FLAGS_GL    = -lGL -lEGL
LD_FLAGS_TEST  = $(LD_FLAGS) $(LD_FLAGS_GL)
LIBS_TEST      = $(LIBS)
IMAGE_IO_FLAGS = -I${HALIDE_ROOT}/tools -ljpeg `libpng-config --cflags --ldflags`
else
CXX_TEST       = $(HOME)/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++
LD_FLAGS_GL    = -l:libGLESv2.so -l:libGLESv3.so -l:libEGL.so
LD_FLAGS_TEST  = -static-libstdc++ -llog $(LD_FLAGS_GL)
LIBS_TEST      = -L${HALIDE_ROOT_ARM}/lib
IMAGE_IO_FLAGS = -I${HALIDE_ROOT_ARM}/tools -DHALIDE_NOPNG -DHALIDE_NO_PNG -DHALIDE_NO_JPEG
endif

ifndef SCHEDULER
	SCHEDULER=1
endif
ifeq ($(SCHEDULER), 0)
	AUTO_SCHEDULE=true
endif
ifndef AUTO_SCHEDULE
	AUTO_SCHEDULE=false
endif
ifndef AUTO_SCHEDULER
	AUTO_SCHEDULER=mullapudi2016
endif
ifeq ($(AUTO_SCHEDULE), true)
	AUTO_SCHEDULER_PARS=-p $(HALIDE_ROOT)/lib/libautoschedule_$(AUTO_SCHEDULER).so
	ifdef MACHINE_PARAMS
		AUTO_SCHEDULER_PARS+=machine_params=$(MACHINE_PARAMS)
	endif
endif

ifndef SIZE_FACTOR
	SIZE_FACTOR=1
endif

TARGET=arm-64-android
ifeq ($(DESKTOP), true)
	TARGET=host
endif

# The following command is to use the Halide profiler on the mobile device:
#  adb logcat -s halide &
PROFILER=
ifeq ($(PROFILE), true)
	PROFILER=-profile
endif

# opencl openglcompute or cuda
ifdef ACCELERATOR
	TARGET:=$(TARGET)-$(ACCELERATOR)
endif

PARS=scheduler=$(SCHEDULER) size_factor=$(SIZE_FACTOR) \
 target=$(TARGET)$(PROFILER) auto_schedule=$(AUTO_SCHEDULE) $(AUTO_SCHEDULER_PARS)
DEFAULT_PARS=target=$(TARGET)$(PROFILER) scheduler=1

ifndef RUNS
	RUNS=3
endif

ifndef TEST
	TEST=isp
endif

OBJS=bin/normalization.o bin/black_level_subtraction.o bin/bilinear_resize.o bin/lens_shading_correction.o \
 bin/white_balance.o bin/demosaic.o bin/color_correction.o bin/reinhard_tone_mapping.o bin/gamma_correction.o \
 bin/denormalization.o bin/rgb_to_ycbcr.o bin/bilateral_denoise.o bin/mix.o bin/ycbcr_to_rgb.o bin/runtime.o

ifeq ($(TEST), bilateral_denoise)
	bilateral_denoise_pars=$(PARS)
endif
ifeq ($(TEST), bilinear_resize)
	bilinear_resize_pars=$(PARS)
endif
ifeq ($(TEST), black_level_subtraction)
	black_level_subtraction_pars=$(PARS)
endif
ifeq ($(TEST), color_correction)
	color_correction_pars=$(PARS)
endif
ifeq ($(TEST), demosaic)
	demosaic_pars=$(PARS)
endif
ifeq ($(TEST), denormalization)
	denormalization_pars=$(PARS)
endif
ifeq ($(TEST), gamma_correction)
	gamma_correction_pars=$(PARS)
endif
ifeq ($(TEST), lens_shading_correction)
	lens_shading_correction_pars=$(PARS)
endif
ifeq ($(TEST), mix)
	mix_pars=$(PARS)
endif
ifeq ($(TEST), normalization)
	normalization_pars=$(PARS)
endif
ifeq ($(TEST), reinhard_tone_mapping)
	reinhard_tone_mapping_pars=$(PARS)
endif
ifeq ($(TEST), rgb_to_ycbcr)
	rgb_to_ycbcr_pars=$(PARS)
endif
ifeq ($(TEST), white_balance)
	white_balance_pars=$(PARS)
endif
ifeq ($(TEST), ycbcr_to_rgb)
	ycbcr_to_rgb_pars=$(PARS)
endif
ifeq ($(TEST), isp)
	OBJS=bin/isp.o bin/runtime.o
	isp_pars=$(PARS)
endif
ifeq ($(TEST), copy_to_gpu)
	OBJS=
endif

normalization_pars += input.dim=2
denormalization_pars += input.type=float32 input.dim=3

all: test

bin/%.gen: halide/%.cpp
	@mkdir -p $(@D)
	@$(CXX) $< $(GENERATOR_DEPS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) -o $@

bin/%.o: bin/%.gen
	@mkdir -p $(@D)
	@$< -e $(GENERATOR_OUTPUTS) -o $(@D) -g $* -f $* target=$(TARGET)-no_runtime $(DEFAULT_PARS) $($*_pars)

DUMMY=isp
bin/runtime.o: bin/$(DUMMY).gen
	@mkdir -p $(@D)
	@$< -e o -o $(@D) target=$(TARGET) -r runtime

bin/test_%: test/%.cpp $(OBJS)
	@mkdir -p $(@D)
	@$(CXX_TEST) $^ $(OPT_FLAGS) $(CXX_FLAGS) $(INCLUDES_TEST) $(LD_FLAGS_TEST) $(LIBS_TEST) $(IMAGE_IO_FLAGS) -o $@
	@rm $(OBJS)
	@rm bin/$(DUMMY).gen

clean:
	@rm -rf bin images_output


IMAGES_DIR = ../../images
MOBILE_DIR = /data/local/tmp

ifndef GAMMA
	GAMMA=2.2
endif
ifndef SIGMA_SPATIAL
	SIGMA_SPATIAL=5
endif
ifndef SIGMA_RANGE
	SIGMA_RANGE=0.05
endif

IMAGES=5a9e_20150405_165352_614 6G7M_20150307_175028_814 IMG_20200508_202014675 IMG_20201009_123817328

ifeq ($(DESKTOP), true)
test: bin/test_$(TEST)
	@mkdir -p images_output
	@for f in $(IMAGES); do \
		echo $$f; \
		for i in $(shell seq $(RUNS)); do \
			$< $(IMAGES_DIR)/$$f.dng $(IMAGES_DIR)/$$f.mat $(IMAGES_DIR)/$$f.txt $(GAMMA) $(SIGMA_SPATIAL) \
				$(SIGMA_RANGE) images_output/$$f.ppm; \
			sleep 5; \
		done; \
	done
	@rm $<
else
test: bin/test_$(TEST)
	@mkdir -p images_output
	@adb push $< $(MOBILE_DIR)/process
	@adb shell chmod +x $(MOBILE_DIR)/process
	@for f in $(IMAGES); do \
		echo $$f; \
		adb push $(IMAGES_DIR)/$$f.dng $(MOBILE_DIR)/$$f.dng; \
		adb push $(IMAGES_DIR)/$$f.mat $(MOBILE_DIR)/$$f.mat; \
		adb push $(IMAGES_DIR)/$$f.txt $(MOBILE_DIR)/$$f.txt; \
		for i in $(shell seq $(RUNS)); do \
			adb shell $(MOBILE_DIR)/process $(MOBILE_DIR)/$$f.dng $(MOBILE_DIR)/$$f.mat $(MOBILE_DIR)/$$f.txt $(GAMMA) $(SIGMA_SPATIAL) \
				$(SIGMA_RANGE) $(MOBILE_DIR)/$$f.ppm; \
			sleep 5; \
		done; \
		adb pull $(MOBILE_DIR)/$$f.ppm images_output/$$f.ppm; \
	done
	@rm $<
endif
