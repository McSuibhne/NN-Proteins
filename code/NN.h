

#ifndef NN_h
#define NN_h 1
#include "Layer.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <cstdlib>
#include <stdlib.h>
#include <time.h>
#include <iomanip>


// NN ver. 0.0 (8/2/2014)
// Copyright (C) Gianluca Pollastri 2014
//
// N-hidden layer Feedforward neural net.
// Input: categorical data (one-hot), real valued or mixed.
// ouput: softmax.
// Cost function: the proper one
// Gradient: plain backpropagation (no momentum)



class NN
{
public:
  int NI;
  int NIr;
  int NItot;
  int Nlayers;
  int* NH;
  int NO;
  int* NK;
  int* NK2;
  int which;
  int outp;
  int inp;

  double* backprop;

  Layer_soft* upper;
  Layer_tanh** lower;


//public:



  // Constructor. Parameters:
  // Number of input attributes, number of hidden units, number of output units;
  // t_NK contains the cardinalities of the attribute spans.


NN(NN* from) {

	NI=from->NI;
	Nlayers = from->Nlayers;
	NO = from->NO;
      NItot=from->NItot;

      NK=new int[NI];
      for (int i=0; i<NI; i++) {
		NK[i]=from->NK[i];
      }

      NIr = from->NIr;
      	NK2=new int[NIr];
      	for (int i=0; i<NIr; i++)
		NK2[i]=1;

      upper= new Layer_soft(from->upper);
      upper->set_output(1);
      upper->set_ninput(2);

	lower = new Layer_tanh*[Nlayers];
	NH = new int[Nlayers];
	for (int l=0;l<Nlayers;l++) {
		NH[l] = from->NH[l];
	}
	for (int l=1;l<Nlayers;l++) {
		lower[l] = new Layer_tanh(from->lower[l]);
		lower[l]->set_ninput(2);
	}
      lower[0]= new Layer_tanh(from->lower[0]);
      lower[0]->set_ninput(0);

      outp=from->outp;
      inp=from->inp;
      which = from->which;
      backprop=new double[NItot+NIr];
}

~NN() {
//	cout << "inNNd\n" << flush;

//	if (NI>0)
		delete[] NK;
	if (NIr>0)
		delete[] NK2;
	delete upper;
	for (int l=0;l<Nlayers;l++) {
		delete lower[l];
	}
	delete[] lower;

	delete[] NH;
	delete[] backprop;
}

  NN(int t_NI, int* t_NH, int t_Nlayers, int t_NO, int* t_NK) :
    NI(t_NI), Nlayers(t_Nlayers), NO(t_NO)
    {
      NK=new int[NI];
      NItot=0;
      for (int i=0; i<NI; i++) {
		NK[i]=t_NK[i];
		NItot += NK[i];
      }

      upper= new Layer_soft(NO,NK,0,t_NH[Nlayers-1]);
      upper->set_output(1);
      upper->set_ninput(2);

	lower = new Layer_tanh*[Nlayers];
	NH = new int[Nlayers];
	for (int l=0;l<Nlayers;l++) {
		NH[l] = t_NH[l];
	}
	for (int l=1;l<Nlayers;l++) {
		lower[l] = new Layer_tanh(NH[l],NK,0,NH[l-1]);
		lower[l]->set_ninput(2);
	}
      lower[0]= new Layer_tanh(NH[0],NK,NI);
      lower[0]->set_ninput(0);
      NIr=0;
      outp=1;
      inp=1;
    };

  void copy_dW(NN* from) {
	upper->copy_dW(from->upper);
	for (int l=0;l<Nlayers;l++) {
		lower[l]->copy_dW(from->lower[l]);
	}
  }


  // Constructor for a net with mixed inputs.
  // NI = number of input attributes (categorical inputs)
  // NIr = number of inputs (real valued)
  // ..
  // outp = output or non-output network (for backprop signal)
  // inp = input or non-inpput network (for backprop signal)


  NN(int t_NI,int t_NIr, int* t_NH,  int t_Nlayers, int t_NO, int* t_NK,
	  int t_outp=1, int t_inp=1, int t_which=1) :
	NI(t_NI), NIr(t_NIr), Nlayers(t_Nlayers), NO(t_NO), outp(t_outp), inp(t_inp)
    {
      int i;
      NK=new int[NI];
      NItot=0;
      for (i=0; i<NI; i++) {
		NK[i]=t_NK[i];
		NItot += NK[i];
      }
      NK2=new int[NIr];
      for (i=0; i<NIr; i++)
	NK2[i]=1;

      which=1;
      upper= new Layer_soft(NO,NK,0,t_NH[Nlayers-1]);

      if (outp)
        upper->set_output(1);
      upper->set_ninput(2);

	lower = new Layer_tanh*[Nlayers];
	NH = new int[Nlayers];
	for (int l=0;l<Nlayers;l++) {
		NH[l] = t_NH[l];
	}
	for (int l=1;l<Nlayers;l++) {
		lower[l] = new Layer_tanh(NH[l],NK,0,NH[l-1]);
		lower[l]->set_ninput(2);
	}
      lower[0]= new Layer_tanh(NH[0],NK,NI,NIr);
      if (inp)
        lower[0]->set_ninput(0);
      backprop=new double[NItot+NIr];
    };

  // Create/read a net from file
  NN(istream& is);
  void read(istream& is);

  // Forward pass
  void forward(int* I);
  void forward(double* I);
  void forward(int* I1, double* I2);
  void forward(double* I1, double* I2);


  double f_cost(double* t) {
	return upper->f_cost(t);
  }

  // Backprop
  double backward(double* t, double weight=1.0);
  double* back_out() {return backprop;}

  double deepLearning(int* I, double* t, int lay);
  double deepLearning(double* I, double* t, int lay);
  double deepLearning(int* I1, double* I2, double* t, int lay);
  double deepLearning(double* I1, double* I2, double* t, int lay);

  // Update gradients
  void gradient(int* I, double* t);
  void gradient(double* I, double* t);
  void gradient(int* I1, double* I2, double* t);
  void gradient(double* I1, double* I2, double* t);

  // Update weights
  virtual void updateWeights(double epsilon);
  virtual void updateWeightsL1(double epsilon);
  virtual void updateWeightsClipped(double epsilon);
  void resetGradient();
  virtual void initWeights(int seed);
  inline double* out() { return upper->out(); };
  void write(ostream& os);

  void set_input(int vi) {
	  lower[0]->set_ninput(vi);
	  inp=vi;
  }
  void set_output(int vo) {
	  upper->set_output(vo);
	  outp=vo;
  }


  inline int get_NI() { return NI; };
  inline int get_NIr() { return NIr; };
  inline int get_NO() { return NO; };
  inline int get_NH(int j) { return NH[j]; };

  double dlength() {
//    return upper->dlength()+lower->dlength();
  }
};



#endif // NN_h
