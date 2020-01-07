CC=mpiicc
#CFLAGS= -O2 -msse4.2 -std=c99 -qopenmp
CFLAGS= -O0 -g -qopenmp 
LIB = -L/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm

ROOTDIR=.
SRCDIR=$(ROOTDIR)/src
BLDDIR=$(ROOTDIR)/obj

INSTALL_DIR=$(ROOTDIR)/bin

CFLAGSLIB=-shared

$(BLDDIR):
	@mkdir -p $(BLDDIR)


$(BLDDIR)/gw.o: | $(BLDDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/gw.c -o $@

$(BLDDIR)/main.o: | $(BLDDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/main.c -o $@
        

$(BLDDIR)/gw_mpi: $(BLDDIR)/gw.o $(BLDDIR)/main.o 
	$(CC) $(CFLAGS) -o $@ $^ $(LIB)

all: $(BLDDIR)/gw_mpi 

clean:
	rm -f $(BLDDIR)/gw.o $(BLDDIR)/main.o 
	rm -f $(BLDDIR)/gw_mpi
	rm $(BLDDIR)/*.txt

install:
	cp -f $(BLDDIR)/gw_mpi $(INSTALL_DIR)/gw_mpi

.PHONY: all
