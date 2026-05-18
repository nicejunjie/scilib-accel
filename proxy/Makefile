FC      = pgf90
FFLAGS  = -O2 -mp -mcmodel=large
BLAS    = -Mnvpl

proxy.x: proxy.f90
	$(FC) $(FFLAGS) -o $@ $< $(BLAS)

.PHONY: clean
clean:
	rm -f proxy.x *.o
