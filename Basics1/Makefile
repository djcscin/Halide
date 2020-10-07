
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
ifdef VERSION
	GPARS=version=$(VERSION)
endif

# all é executado quando para make
all: images_output

# resultado: pré-requisitos
# $@ é o resultado
# $^ são os pré-requisitos
# $< é o primeiro pré-requisito
bin/gamma.generator: src/HalideGamma.cpp
	@mkdir -p $(@D)
	$(CXX) $^ $(GENERATOR_DEPS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) -o $@

bin/gamma.o: bin/gamma.generator
	@mkdir -p $(@D)
	$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -f gamma -g gamma target=host $(AUTO_SCHEDULER_PARS) $(GPARS)
	@rm $^

bin/gamma: src/gamma.cpp bin/gamma.o
	@mkdir -p $(@D)
	$(CXX) $^ $(OPT_FLAGS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) $(IMAGE_IO_FLAGS) -o $@

images_output: bin/gamma
	@mkdir -p $@
	@time $^ ../images/final2016.jpg 0.5 2.2 ../images/final2017.jpg 0.5 2.2 $@/final2020.jpg
	@time $^ ../images/final2016.jpg 0.4 0.9 ../images/final2017.jpg 0.6 0.9 $@/final2021.jpg
	@time $^ ../images/final2016.jpg 0.5 1.3 ../images/final2017.jpg 0.5 1.3 $@/final2022.jpg

clean:
	@rm -rf bin images_output
