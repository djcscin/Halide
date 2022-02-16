
CXX = g++

GENERATOR_DEPS     = ${HALIDE_ROOT}/tools/GenGen.cpp
GENERATOR_OUTPUTS  = o,h,schedule,stmt_html
CXX_FLAGS          = -std=c++1z -fno-rtti
OPT_FLAGS          = -O3
INCLUDES           = -I${HALIDE_ROOT}/include -Ibin -I../include
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
GPARS=
ifdef SCHEDULER
	GPARS=scheduler=$(SCHEDULER)
endif
TARGET=host
ifdef PROFILE
	ifeq ($(PROFILE), true)
		TARGET=host-profile
	endif
endif

all: images_output

bin/demosaic.generator: src/HalideDemosaic.cpp
	@mkdir -p $(@D)
	@$(CXX) $^ $(GENERATOR_DEPS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) -o $@

bin/demosaic.o: bin/demosaic.generator
	@mkdir -p $(@D)
	$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -f demosaic -g demosaic target=$(TARGET) $(AUTO_SCHEDULER_PARS) $(GPARS)
	@rm $^

bin/demosaic: src/demosaic.cpp bin/demosaic.o
	@mkdir -p $(@D)
	@$(CXX) $^ $(OPT_FLAGS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) $(IMAGE_IO_FLAGS) -o $@

images_output: bin/demosaic
	@mkdir -p $@
	@$^ ../images/5a9e_20150405_165352_614.dng 2.150001287460327 1.0 1.5542165040969849 $@/5a9e_20150405_165352_614.png
	@$^ ../images/6G7M_20150307_175028_814.dng 2.2068965435028076 1.0 1.3333333730697632 $@/6G7M_20150307_175028_814.png
	@$^ ../images/IMG_20200508_202014675.dng 1.9616857767105103 1.0 1.7355931997299194 $@/IMG_20200508_202014675.png
	@$^ ../images/IMG_20201009_123817328.dng 1.6050156354904175 1.0 1.899814486503601 $@/IMG_20201009_123817328.png

clean:
	@rm -rf bin images_output
