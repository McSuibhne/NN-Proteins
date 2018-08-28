OBJECTST = Layer.o NN.o Model.o Train.o
HEADERST = Layer.h NN.h Model.h Sequence.h

CC	= g++
CFLAGS  = -O3

Train :	$(OBJECTST) 
	$(CC) $(CFLAGS) -o $@ $(OBJECTST) $(LLIBS)

Train.o : Train.cxx $(HEADERST)
	$(CC) -c $(CFLAGS) $*.cxx

Layer.o : Layer.cxx $(HEADERST)
	$(CC) -c $(CFLAGS) $*.cxx

NN.o : NN.cxx $(HEADERST)
	$(CC) -c $(CFLAGS) $*.cxx

Model.o : Model.cxx $(HEADERST)
	$(CC) -c $(CFLAGS) $*.cxx


clean:
	rm *.o Train
