################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../center_surround_convolution.cu 

OBJS += \
./center_surround_convolution.o 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/bin/nvcc --device-debug --debug -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -ccbin g++ -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


clean: clean--2e-

clean--2e-:
	-$(RM) ./center_surround_convolution.o

.PHONY: clean--2e-

