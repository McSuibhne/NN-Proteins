
#include <math.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdlib.h>

#include "MModel.h"
#include "Sequence.h"

using namespace std;





void
evaluate(MModel* M, DataSet& D, char* which, double threshold)
{
        int y;

  cout << "\n counting_" << which << "_errors" << flush;
  M->resetNErrors();
  int p;
  for (p=0; p<D.length; p++) {
if (strncmp(which,"test",4)==0)
    M->predict(D.seq[p], threshold);
else
    M->predict(D.seq[p], threshold);
    if (p%20==0) cout << "." << flush;
  }

double a[128];
double all=0;

for (y=0;y<M->getClasses();y++) {
        a[y]=M->getCounted()[y];
        all += a[y];
}

cout << "Threshold :" << threshold << "\n";
//  cout << "\n\n" << which << threshold<<"_NErrors= " << M->getNErrors() << "/" << all;
//  cout << " " << (double)M->getNErrors()/(double)(all)*100.0;

//for (y=0;y<M->getClasses();y++) {
//  cout << "\nClass" << y << threshold<<"= " << M->getNErrors_(y) << "/" << a[y];
//  cout << "\t" << (double)M->getNErrors_(y)/(double)a[y]*100.0;
//}
  cout<<"\n";
for (y=0;y<M->getClasses();y++) {
	for (int z=0;z<M->getClasses();z++) {
		cout << M->getConf()[z][y] << " ";
	}
	cout << "\n";
}

  cout<<"\n";

}








int
main(int argc, char** argv)
{

// This is for predicting


  if (argc<4) {
    cerr << "Usage: " << argv[0] << " model_file protein_file nThresholds threshold1 .. thresholdN\n";
    exit(1);
  }

    char model[256];
    char prot[256];
        strcpy(model,argv[1]);
        strcpy(prot,argv[2]);

	int nThresholds;
//	sscanf(argv[3],"%i",&nThresholds);
	nThresholds = atoi(argv[3]);

	double* thresholds = new double[nThresholds];

	for (int th = 0; th<nThresholds; th++) {
		thresholds[th] = atof(argv[4+th]);
	}

    MModel* M;
    char tmp[1024];
    strcpy(tmp, model);
    ifstream mstream(tmp);
    M = new MModel(mstream);

  cout << "Reading " << prot << " .. ";
  ifstream tstream(prot);
  DataSet T(tstream);

cout << "read\n" << flush;

for (int th = 0; th<nThresholds; th++) {
	evaluate(M, T, prot, thresholds[th]);
//	strcat(prot,"F");
//	T.write(prot);
}
        return 0;
}
