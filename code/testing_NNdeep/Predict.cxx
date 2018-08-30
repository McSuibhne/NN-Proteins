
#include <math.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdlib.h>

#include "MModel.h"
#include "Sequence.h"

using namespace std;





void
evaluate(MModel* M, DataSet& D, char* which)
{
        int y;

  cout << "\n counting_" << which << "_errors" << flush;
  M->resetNErrors();
  int p;
  for (p=0; p<D.length; p++) {
    M->predict(D.seq[p]);
    if (p%20==0) cout << "." << flush;
  }

double a[128];
double all=0;

for (y=0;y<M->getClasses();y++) {
        a[y]=M->getCounted()[y];
        all += a[y];
}

  cout << "\n\n" << which <<"_NErrors= " << M->getNErrors() << "/" << all;
  cout << " " << (double)M->getNErrors()/(double)(all)*100.0;

for (y=0;y<M->getClasses();y++) {
  cout << "\nClass" << y <<"= " << M->getNErrors_(y) << "/" << a[y];
  cout << "\t" << (double)M->getNErrors_(y)/(double)a[y]*100.0;
}

  cout<<"\n";

}








int
main(int argc, char** argv)
{

// This is for predicting


  if (argc<3) {
    cerr << "Usage: " << argv[0] << " model_file protein_file\n";
    exit(1);
  }

    char model[256];
    char prot[256];
    char tmp[256];
//        char alig[256];
        strcpy(model,argv[1]);
        strcpy(prot,argv[2]);
//        strcpy(alig,argv[3]);

        double th = 0.5;


    MModel* M;
    strcpy(tmp, model);
    ifstream mstream(tmp);
    M = new MModel(mstream);

  cout << "Reading " << prot << " .. ";
  ifstream tstream(prot);
  DataSet T(tstream);

cout << "read\n" << flush;

evaluate(M, T, prot);
strcat(prot,".predictions");
T.write(prot);
        return 0;
}