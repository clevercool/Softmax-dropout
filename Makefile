cc = nvcc
prom = forward
deps = 
src = $(shell find ./ -name "*.cu")
obj = $(src:%.cu=%.o)


$(prom) : $(obj)
	$(cc) -o $(prom) $(obj)

%.o : %.cu $(deps)
	$(cc) -c $< -o $@

clean:
	rm $(obj) $(prom)