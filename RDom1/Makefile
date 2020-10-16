
CXX = g++

GENERATOR_DEPS     = ${HALIDE_ROOT}/tools/GenGen.cpp
GENERATOR_OUTPUTS  = o,h,schedule,stmt_html
CXX_FLAGS          = -std=c++11 -fno-rtti
OPT_FLAGS          = -O3
INCLUDES           = -I${HALIDE_ROOT}/include -Ibin
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
ifdef PREDICATE
	GPARS+=predicate=$(PREDICATE)
endif
TARGET=host
ifdef PROFILE
	ifeq ($(PROFILE), true)
		TARGET=host-profile
	endif
endif

all: images_output

bin/denoise.generator: src/HalideDenoise.cpp
	@mkdir -p $(@D)
	@$(CXX) $^ $(GENERATOR_DEPS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) -o $@

bin/denoise.o: bin/denoise.generator
	@mkdir -p $(@D)
	$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -f denoise -g denoise target=$(TARGET) $(AUTO_SCHEDULER_PARS) $(GPARS)
	@rm $^

bin/denoise: src/denoise.cpp bin/denoise.o
	@mkdir -p $(@D)
	@$(CXX) $^ $(OPT_FLAGS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) $(IMAGE_IO_FLAGS) -o $@

images_output: bin/denoise
	@mkdir -p $@
	@$^ ../images/synthetic_boxes_night.png 0.5 5 $@/synthetic_boxes_night_0.5_5.png
	@$^ ../images/synthetic_boxes_night.png 0.5 10 $@/synthetic_boxes_night_0.5_10.png
	@$^ ../images/synthetic_boxes_night.png 0.5 15 $@/synthetic_boxes_night_0.5_15.png
	@$^ ../images/synthetic_boxes_night.png 1.5 5 $@/synthetic_boxes_night_1.5_5.png
	@$^ ../images/synthetic_boxes_night.png 1.5 10 $@/synthetic_boxes_night_1.5_10.png
	@$^ ../images/synthetic_boxes_night.png 1.5 15 $@/synthetic_boxes_night_1.5_15.png
	@$^ ../images/synthetic_boxes_night.png 5 5 $@/synthetic_boxes_night_5_5.png
	@$^ ../images/synthetic_boxes_night.png 5 10 $@/synthetic_boxes_night_5_10.png
	@$^ ../images/synthetic_boxes_night.png 5 15 $@/synthetic_boxes_night_5_15.png

clean:
	@rm -rf bin images_output
