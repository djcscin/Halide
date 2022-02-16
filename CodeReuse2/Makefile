
CXX = g++

GENERATOR_DEPS     = ${HALIDE_ROOT}/tools/GenGen.cpp
GENERATOR_OUTPUTS  = o,h
CXX_FLAGS          = -std=c++1z -fno-rtti
OPT_FLAGS          = -O3
INCLUDES           = -I${HALIDE_ROOT}/include -Ibin -I../include -Iinclude
LD_FLAGS           = -lHalide -ldl -lpthread -lz
LIBS               = -L${HALIDE_ROOT}/lib
IMAGE_IO_FLAGS     = -I${HALIDE_ROOT}/tools -ljpeg `libpng-config --cflags --ldflags`

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

TARGET=host
ifdef PROFILE
	ifeq ($(PROFILE), true)
		TARGET=host-profile
	endif
endif

all: images_output

bin/lsc_map.generator: src/HalideLscMap.cpp
	@mkdir -p $(@D)
	@$(CXX) $^ $(GENERATOR_DEPS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) -o $@

bin/lsc_map.o: bin/lsc_map.generator
	@mkdir -p $(@D)
	@$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -f lsc_map -g lsc_map target=$(TARGET) $(AUTO_SCHEDULER_PARS)

bin/demosaic.generator: src/HalideDemosaic.cpp
	@mkdir -p $(@D)
	@$(CXX) $^ $(GENERATOR_DEPS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) -o $@

bin/demosaic.o: bin/demosaic.generator
	@mkdir -p $(@D)
	@$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -f demosaic -g demosaic img_output_type=uint8 target=$(TARGET)-no_runtime $(AUTO_SCHEDULER_PARS)

bin/demosaic16.o: bin/demosaic.generator bin/demosaic.o
	@mkdir -p $(@D)
	@$< -e $(GENERATOR_OUTPUTS) -o $(@D) -f demosaic16 -g demosaic img_output_type=uint16 target=$(TARGET)-no_runtime $(AUTO_SCHEDULER_PARS)

bin/denoise.generator: src/HalideDenoise.cpp
	@mkdir -p $(@D)
	@$(CXX) $^ $(GENERATOR_DEPS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) -o $@

bin/denoise.o: bin/denoise.generator
	@mkdir -p $(@D)
	@$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -f denoise -g denoise img_input.type=uint16 img_output_type=uint8 target=$(TARGET)-no_runtime $(AUTO_SCHEDULER_PARS)

bin/denoise16.o: bin/denoise.generator
	@mkdir -p $(@D)
	@$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -f denoise16 -g denoise img_input.type=uint16 img_output_type=uint16 target=$(TARGET)-no_runtime $(AUTO_SCHEDULER_PARS)

bin/denoise16_chroma.o: bin/denoise.generator
	@mkdir -p $(@D)
	@$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -f denoise16_chroma -g denoise img_input.type=uint16 img_output_type=uint16 min_channel=1 num_channels=2 target=$(TARGET)-no_runtime $(AUTO_SCHEDULER_PARS)

bin/denoise16_luma.o: bin/denoise.generator bin/denoise.o bin/denoise16.o bin/denoise16_chroma.o
	@mkdir -p $(@D)
	@$< -e $(GENERATOR_OUTPUTS) -o $(@D) -f denoise16_luma -g denoise img_input.type=uint16 img_output_type=uint16 min_channel=0 num_channels=1 target=$(TARGET)-no_runtime $(AUTO_SCHEDULER_PARS)

bin/rgb_to_ycbcr.generator: src/HalideRGB2YCbCr.cpp
	@mkdir -p $(@D)
	@$(CXX) $^ $(GENERATOR_DEPS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) -o $@

bin/rgb_to_ycbcr.o: bin/rgb_to_ycbcr.generator
	@mkdir -p $(@D)
	@$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -f rgb_to_ycbcr -g rgb_to_ycbcr img_input.type=uint16 img_output_type=uint16 target=$(TARGET)-no_runtime $(AUTO_SCHEDULER_PARS)

bin/ycbcr_to_rgb.generator: src/HalideYCbCr2RGB.cpp
	@mkdir -p $(@D)
	@$(CXX) $^ $(GENERATOR_DEPS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) -o $@

bin/ycbcr_to_rgb.o: bin/ycbcr_to_rgb.generator
	@mkdir -p $(@D)
	@$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -f ycbcr_to_rgb -g ycbcr_to_rgb img_input.type=uint16 img_output_type=uint8 target=$(TARGET)-no_runtime $(AUTO_SCHEDULER_PARS)

bin/mix.generator: src/HalideMix.cpp
	@mkdir -p $(@D)
	@$(CXX) $^ $(GENERATOR_DEPS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) -o $@

bin/mix.o: bin/mix.generator
	@mkdir -p $(@D)
	@$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -f mix -g mix luma_input.type=uint16 chroma_input.type=uint16 img_mix_output.type=uint16 target=$(TARGET)-no_runtime $(AUTO_SCHEDULER_PARS)

bin/isp.generator: src/HalideISP.cpp
	@mkdir -p $(@D)
	@$(CXX) $^ $(GENERATOR_DEPS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) -o $@

bin/isp.o: bin/isp.generator
	@mkdir -p $(@D)
	@$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -f isp -g isp target=$(TARGET)-no_runtime $(AUTO_SCHEDULER_PARS) has_sigma_luma=false

bin/isp_luma.o: bin/isp.generator
	@mkdir -p $(@D)
	@$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -f isp_luma -g isp target=$(TARGET)-no_runtime $(AUTO_SCHEDULER_PARS) has_sigma_luma=true

bin/isp_debug.o: bin/isp.generator
	@mkdir -p $(@D)
	@$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -f isp_debug -g isp target=$(TARGET)-no_runtime $(AUTO_SCHEDULER_PARS) has_sigma_luma=false output_demosaic=true

bin/isp: src/isp.cpp bin/lsc_map.o bin/demosaic.o bin/demosaic16.o bin/denoise.o bin/rgb_to_ycbcr.o bin/ycbcr_to_rgb.o bin/denoise16.o bin/mix.o bin/denoise16_chroma.o bin/denoise16_luma.o bin/isp.o bin/isp_luma.o bin/isp_debug.o
	@mkdir -p $(@D)
	@$(CXX) $^ $(OPT_FLAGS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) $(IMAGE_IO_FLAGS) -o $@

images_output: bin/isp
	@mkdir -p $@
	@$^ ../images/5a9e_20150405_165352_614.dng ../images/5a9e_20150405_165352_614.mat 2.150001287460327  1.0 1.5542165040969849 5.0 0.06 1.5 0.03 $@/5a9e_20150405_165352_614.png $@/5a9e_20150405_165352_614_rgb.png $@/5a9e_20150405_165352_614_ycbcr.png $@/5a9e_20150405_165352_614_chroma.png $@/5a9e_20150405_165352_614_chroma_luma.png $@/5a9e_20150405_165352_614_gen_isp.png $@/5a9e_20150405_165352_614_gen_isp_luma.png $@/5a9e_20150405_165352_614_gen_isp_debug.png $@/5a9e_20150405_165352_614_gen_isp_debug_dms.png
	@$^ ../images/6G7M_20150307_175028_814.dng ../images/6G7M_20150307_175028_814.mat 2.2068965435028076 1.0 1.3333333730697632 5.0 0.06 1.5 0.03 $@/6G7M_20150307_175028_814.png $@/6G7M_20150307_175028_814_rgb.png $@/6G7M_20150307_175028_814_ycbcr.png $@/6G7M_20150307_175028_814_chroma.png $@/6G7M_20150307_175028_814_chroma_luma.png $@/6G7M_20150307_175028_814_gen_isp.png $@/6G7M_20150307_175028_814_gen_isp_luma.png $@/6G7M_20150307_175028_814_gen_isp_debug.png $@/6G7M_20150307_175028_814_gen_isp_debug_dms.png
	@$^ ../images/IMG_20200508_202014675.dng   ../images/IMG_20200508_202014675.mat   1.9616857767105103 1.0 1.7355931997299194 5.0 0.06 1.5 0.03 $@/IMG_20200508_202014675.png   $@/IMG_20200508_202014675_rgb.png   $@/IMG_20200508_202014675_ycbcr.png   $@/IMG_20200508_202014675_chroma.png   $@/IMG_20200508_202014675_chroma_luma.png   $@/IMG_20200508_202014675_gen_isp.png   $@/IMG_20200508_202014675_gen_isp_luma.png   $@/IMG_20200508_202014675_gen_isp_debug.png   $@/IMG_20200508_202014675_gen_isp_debug_dms.png
	@$^ ../images/IMG_20201009_123817328.dng   ../images/IMG_20201009_123817328.mat   1.6050156354904175 1.0 1.899814486503601  5.0 0.06 1.5 0.03 $@/IMG_20201009_123817328.png   $@/IMG_20201009_123817328_rgb.png   $@/IMG_20201009_123817328_ycbcr.png   $@/IMG_20201009_123817328_chroma.png   $@/IMG_20201009_123817328_chroma_luma.png   $@/IMG_20201009_123817328_gen_isp.png   $@/IMG_20201009_123817328_gen_isp_luma.png   $@/IMG_20201009_123817328_gen_isp_debug.png   $@/IMG_20201009_123817328_gen_isp_debug_dms.png

clean:
	@rm -rf bin images_output
