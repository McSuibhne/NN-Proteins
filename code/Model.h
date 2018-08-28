
#ifndef Model_h
#define Model_h 1

#include <stdlib.h>
#include <math.h>
#include "Sequence.h"
#include "NN.h"


class Model {
 public:
  int NU;
  int NY;
  int NLayers;
  int* CODE;

  int context;
  int Moore; 


  int Cseg;
  int Cwin;
  int shortcut;
  int Step;

  double* Thresholds;

  int cycles;
  double* dcycles;

  int modular;


  NN* Net;

  NN* tempo;

  int** Conf;

//  double temp_error;
  int temp_aas;
  
  int* counted;
  double squared_error;
  double error;
  int nerrors;
  int* nerrors_;

  double epsilon;

  void alloc();



  Model(Model* from);
  Model(int NU, int NY, int NLayers,  int* CODE, int context ,int Moore, 
	int Cseg, int Cwin, int Step, int shortcut, double* Thresholds, int cycles=1);
  Model(istream& is);
  ~Model();
  void read(istream& is);
  void write(ostream& os);

  void randomize(int seed);


  void copytempo(Model* from) {
	Net->copy_dW(from->Net);
	error += from->error;
  }


  void copyerrors(Model* from) {
	nerrors += from->nerrors;
	for (int y=0;y<NY;y++) {
		nerrors_[y] += from->nerrors_[y];
		for (int z=0;z<NY;z++) {
			Conf[y][z] += from->Conf[y][z];
		}
		counted[y] += from->counted[y];
	}
  }


//  void extimation(Sequence* seq);
  void extimation(DataSet* D, int seq, int pos);
  void deepExtimation(DataSet* D, int seq, int pos);
  void maximization();
  void maximizationL1();
  void maximizationClipped();

  void predict(Sequence* seq);
  void predict(Sequence* seq, int cy);
//  void predict(Sequence* seq, int W);
  double* out() {return Net->out();}
  int** getConf() {return Conf;}

  int getNErrors() { return nerrors;};

  int getNErrors_(int i) { return nerrors_[i];};
  int getClasses() { return NY;};

  int* getCounted() {return counted;}


  double* getdcycles() {return dcycles;}

   void resetNErrors() { 
	error=0;
	nerrors=0;
	memset(nerrors_,0,NY*sizeof(int));
	memset(counted,0,NY*sizeof(int));
	for (int p=0;p<NY;p++)
	  for (int y=0;y<NY;y++)
		Conf[p][y]=0;
	  for (int c=0;c<cycles;c++) {
		  dcycles[c]=0;
	  }
	};

  double get_error() { 
	return error;
	};
  double get_squared_error() { 
	return error;
	};
  void reset_squared_error() { 
	error = 0;
	  for (int c=0;c<cycles;c++) {
		  dcycles[c]=0;
	  }
	};

  void setEpsilon(double eps) { 
	epsilon = eps;
  };


};


#endif // Model_h
