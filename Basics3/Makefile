
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
	AUTO_SCHEDULE=true
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
ifdef UPDATE
	GPARS+=update=$(UPDATE)
endif
ifdef META
	GPARS+=meta=$(META)
endif
ifdef SIGMA
	GPARS+=sigma=$(SIGMA)
endif

all: images_output

bin/cd.generator: src/HalideCD.cpp
	@mkdir -p $(@D)
	$(CXX) $^ $(GENERATOR_DEPS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) -o $@

bin/cd.o: bin/cd.generator
	@mkdir -p $(@D)
	$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -f cd -g cd target=host $(AUTO_SCHEDULER_PARS) $(GPARS)
	@rm $^

bin/cd: src/cd.cpp bin/cd.o
	@mkdir -p $(@D)
	$(CXX) $^ $(OPT_FLAGS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) $(IMAGE_IO_FLAGS) -o $@

images_output: bin/cd
	@mkdir -p $@
	@time $^ ../images/synthetic_boxes_night.png $@/synthetic_boxes_night.png

clean:
	@rm -rf bin images_output
