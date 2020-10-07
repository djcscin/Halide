
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

all: images_output

bin/gradient.generator: src/HalideGradient.cpp
	@mkdir -p $(@D)
	@$(CXX) $^ $(GENERATOR_DEPS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) -o $@

bin/gradient.o: bin/gradient.generator
	@mkdir -p $(@D)
	$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -f gradient -g gradient target=host $(AUTO_SCHEDULER_PARS) $(GPARS)
	@rm $^

bin/gradient: src/gradient.cpp bin/gradient.o
	@mkdir -p $(@D)
	@$(CXX) $^ $(OPT_FLAGS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) $(IMAGE_IO_FLAGS) -o $@

images_output: bin/gradient
	@mkdir -p $@
	@$^ ../images/final2016.jpg 10 $@/edges10.png
	@$^ ../images/final2016.jpg 30 $@/edges30.png
	@$^ ../images/final2016.jpg 50 $@/edges50.png
	@$^ ../images/final2016.jpg 70 $@/edges70.png
	@$^ ../images/final2016.jpg 90 $@/edges90.png

clean:
	@rm -rf bin images_output
