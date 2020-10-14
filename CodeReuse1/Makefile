
CXX = g++

GENERATOR_DEPS     = ${HALIDE_ROOT}/tools/GenGen.cpp
GENERATOR_OUTPUTS  = o,h,schedule,stmt_html
CXX_FLAGS          = -std=c++11 -fno-rtti
OPT_FLAGS          = -O3
INCLUDES           = -I${HALIDE_ROOT}/include -Ibin -Iinclude
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

all: images_output

bin/median.generator: src/HalideMedian.cpp
	@mkdir -p $(@D)
	$(CXX) $^ $(GENERATOR_DEPS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) -o $@

bin/median_2.o: bin/median.generator
	@mkdir -p $(@D)
	$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -g median target=host $(AUTO_SCHEDULER_PARS) -f median_2 img_input.dim=2 img_input.type=uint8 img_output.dim=2 img_output.type=uint8

ifndef USE_HALIDE_MEDIAN_Y
bin/median_3.o: bin/median.generator
	@mkdir -p $(@D)
	$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -g median target=host $(AUTO_SCHEDULER_PARS) -f median_3 img_input.dim=3 img_input.type=uint8 img_output.dim=3 img_output.type=uint8
endif

bin/median_y.generator: src/HalideMedian_Y.cpp
	@mkdir -p $(@D)
	$(CXX) $^ $(GENERATOR_DEPS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) -o $@

bin/median_y.o: bin/median_y.generator
	@mkdir -p $(@D)
	$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -g median_y target=host $(AUTO_SCHEDULER_PARS) -f median_y run_only_on_luma=true

ifdef USE_HALIDE_MEDIAN_Y
bin/median_3.o: bin/median_y.generator
	@mkdir -p $(@D)
	$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -g median_y target=host $(AUTO_SCHEDULER_PARS) -f median_3 run_only_on_luma=false
endif

bin/median: src/median.cpp bin/median_2.o bin/median_3.o bin/median_y.o
	@mkdir -p $(@D)
	$(CXX) $^ $(OPT_FLAGS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) $(IMAGE_IO_FLAGS) -o $@

images_output: bin/median
	@mkdir -p $@
	time $^ ../images/final2016_salt_pepper_gray.jpg true $@/final2016_no_salt_peper_gray.jpg
	time $^ ../images/final2016_salt_pepper.jpg true $@/final2016_no_salt_peper_true.jpg
	time $^ ../images/final2016_salt_pepper.jpg false $@/final2016_no_salt_peper_false.jpg

clean:
	@rm -rf bin images_output
