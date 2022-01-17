main.exe: main.o nnets.o
	g++ -o main.exe main.o nnets.o

main.o: main.cpp nnets.h
	g++ -c main.cpp 

nnets.o: nnets.cpp nnets.h 
	g++ -c nnets.cpp

debug:
	g++ -g -o main.exe main.cpp nnets.cpp

clean:
	rm -f *.exe *.o *.stackdump *~

backup:
	test -d backups || mkdir backups
	cp *.cpp backups
	cp *.h backups
	cp Makefile backups