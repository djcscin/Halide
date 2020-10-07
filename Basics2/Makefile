
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

all: images_output

bin/sharp.generator: src/HalideSharp.cpp
	@mkdir -p $(@D)
	$(CXX) $^ $(GENERATOR_DEPS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) -o $@

bin/laplacian0.o: bin/sharp.generator
	@mkdir -p $(@D)
	$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -g sharp target=host-no_runtime $(AUTO_SCHEDULER_PARS) -f laplacian0 filter=laplacian0

bin/laplacian1.o: bin/sharp.generator
	@mkdir -p $(@D)
	$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -g sharp target=host-no_runtime $(AUTO_SCHEDULER_PARS) -f laplacian1 filter=laplacian1

bin/laplacian2.o: bin/sharp.generator
	@mkdir -p $(@D)
	$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -g sharp target=host-no_runtime $(AUTO_SCHEDULER_PARS) -f laplacian2 filter=laplacian2

bin/unsharp_gauss.o: bin/sharp.generator
	@mkdir -p $(@D)
	$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -g sharp target=host-no_runtime $(AUTO_SCHEDULER_PARS) -f unsharp_gauss filter=unsharp_gauss

bin/dog.o: bin/sharp.generator
	@mkdir -p $(@D)
	$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -g sharp target=host-no_runtime $(AUTO_SCHEDULER_PARS) -f dog filter=dog

bin/gaussian.o: bin/sharp.generator
	@mkdir -p $(@D)
	$^ -e $(GENERATOR_OUTPUTS) -o $(@D) -g sharp target=host-no_runtime $(AUTO_SCHEDULER_PARS) -f gaussian filter=gaussian

bin/runtime.o: bin/sharp.generator
	@mkdir -p $(@D)
	$^ -e o -o $(@D) target=host -r runtime

bin/sharp: src/sharp.cpp bin/runtime.o bin/gaussian.o bin/laplacian0.o bin/laplacian1.o bin/laplacian2.o bin/unsharp_gauss.o bin/dog.o
	@mkdir -p $(@D)
	$(CXX) $^ $(OPT_FLAGS) $(CXX_FLAGS) $(INCLUDES) $(LD_FLAGS) $(LIBS) $(IMAGE_IO_FLAGS) -o $@

images_output: bin/sharp
	@mkdir -p $@
	@time $^ ../images/final2016.jpg 0 gaussian $@/blurred.jpg
	@time $^ ../images/final2016.jpg 0.4 laplacian0 $@/laplacian0.jpg
	@time $^ ../images/final2016.jpg 0.4 laplacian1 $@/laplacian1.jpg
	@time $^ ../images/final2016.jpg 0.2 laplacian2 $@/laplacian2.jpg
	@time $^ ../images/final2016.jpg 2.0 unsharp_gauss $@/unsharp_gauss.jpg
	@time $^ ../images/final2016.jpg 4.0 dog $@/dog.jpg
	@time $^ $@/blurred.jpg 0.4 laplacian0 $@/blurred_laplacian0.jpg
	@time $^ $@/blurred.jpg 0.4 laplacian1 $@/blurred_laplacian1.jpg
	@time $^ $@/blurred.jpg 0.2 laplacian2 $@/blurred_laplacian2.jpg
	@time $^ $@/blurred.jpg 2.0 unsharp_gauss $@/blurred_unsharp_gauss.jpg
	@time $^ $@/blurred.jpg 4.0 dog $@/blurred_dog.jpg

clean:
	@rm -rf bin images_output
