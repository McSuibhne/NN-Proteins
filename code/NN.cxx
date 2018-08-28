
// NN ver. 0.0 (8/2/2014)
//
// Copyright (C) Gianluca Pollastri 2014


#include "NN.h"


NN::NN(istream& is)
{
  is >> NO >> Nlayers >> NI >> NIr >> which >> outp >> inp;

NH = new int[Nlayers];
for (int l=0;l<Nlayers;l++) {
	is >> NH[l];
}
  upper= new Layer_soft(is);
  if (outp)
    upper->set_output(1);
  upper->set_ninput(2);

lower = new Layer_tanh*[Nlayers];
for (int l=0;l<Nlayers;l++) {
  lower[l]=new Layer_tanh(is);
  lower[l]->set_ninput(2);
}
  lower[0]->set_ninput(inp);

  int i;
  NK=new int[NI];
  NItot=0;
  for (i=0; i<NI; i++) {
	NK[i]=lower[0]->get_NK()[i];
	NItot += NK[i];
  }
  NK2=new int[NIr];
  for (i=0; i<NIr; i++)
	NK2[i]=lower[0]->get_NK()[NI+i];
  backprop=new double[NItot+NIr];
}


void
NN::read(istream& is)
{
  is >> NO >> Nlayers >> NI >> NIr >> which >> outp >> inp;

for (int l=0;l<Nlayers;l++) {
	is >> NH[l];
}

  upper->read(is);
  if (outp)
  	upper->set_output(1);
  upper->set_ninput(2);

for (int l=0;l<Nlayers;l++) {
  lower[l]->read(is);
  lower[l]->set_ninput(2);
}
  lower[0]->set_ninput(inp);

  int i;
  NItot =0;
  for (i=0; i<NI; i++) {
	NK[i]=lower[0]->get_NK()[i];
	NItot += NK[i];
  }
  for (i=0; i<NIr; i++)
	NK2[i]=lower[0]->get_NK()[NI+i];
}



void
NN::forward(int* I)
{
  lower[0]->forward(I);
  for (int l=1;l<Nlayers;l++) {
	lower[l]->forward(lower[l-1]->out(),lower[l-1]->out());
  }
  upper->forward(lower[Nlayers-1]->out(),lower[Nlayers-1]->out());
}

void
NN::forward(double* I)
{
  lower[0]->forward(I);
  for (int l=1;l<Nlayers;l++) {
	lower[l]->forward(lower[l-1]->out(),lower[l-1]->out());
  }
  upper->forward(lower[Nlayers-1]->out(),lower[Nlayers-1]->out());
}

void
NN::forward(int* I1,double* I2)
{
  lower[0]->forward(I1,I2);
  for (int l=1;l<Nlayers;l++) {
	lower[l]->forward(lower[l-1]->out(),lower[l-1]->out());
  }
  upper->forward(lower[Nlayers-1]->out(),lower[Nlayers-1]->out());
}

void
NN::forward(double* I1,double* I2)
{
  lower[0]->forward(I1,I2);
  for (int l=1;l<Nlayers;l++) {
	lower[l]->forward(lower[l-1]->out(),lower[l-1]->out());
  }
  upper->forward(lower[Nlayers-1]->out(),lower[Nlayers-1]->out());
}

double
NN::backward(double* t, double weight)
{
  double err=upper->backward(t,weight);
  double BKD[1024];
  for (int i=0;i<NH[Nlayers-1];i++)
	BKD[i]=upper->back_out()[i];

for (int l=Nlayers-1; l>0; l--) {
  lower[l]->backward(BKD, weight);
  for (int i=0;i<NH[l-1];i++) BKD[i] = lower[l]->back_out()[i];
}
lower[0]->backward(BKD,weight);

  if (inp==1)
    for (int r=NItot;r<NItot+NIr;r++)
      backprop[r]=lower[0]->back_out()[r];
  else if (inp==2)
    for (int r=0;r<NItot+NIr;r++)
      backprop[r]=lower[0]->back_out()[r];
  return err;
}


  double NN::deepLearning(int* I, double* t, int lay) {
    int** Targets;
    Targets = new int*[10];
// Make random targets
    for (int i=0;i<10;i++) {
        Targets[i] = new int[NH[lay]];
        for (int j=0;j<NH[lay]; j++) {
            int y = rand()%2;
            if (y==0) Targets[i][j] = -1.0;
            else Targets[i][j] = 1.0;
        }
    }

    double bestErr = 100000.0;
    int bestJ = -1;
    double T[1024];
    for (int i=0;i<10;i++) {
        for (int j=0;j<NH[lay]; j++) {
            T[j] = Targets[i][j];
        }
            for (int l=lay+1;l<Nlayers;l++) {
                lower[l]->forward(T,T);
                for (int j=0;j<NH[l]; j++) {
                    T[j] = lower[l]->out()[j];
                }
            }
            upper->forward(T,T);
            double erJ = upper->f_cost(t);
            if (erJ<bestErr) {
                bestErr = erJ;
                bestJ = i;
            }
    }


  lower[0]->forward(I);
  for (int l=1;l<=lay;l++) {
	lower[l]->forward(lower[l-1]->out(),lower[l-1]->out());
  }

  double BKD[1024];
  for (int i=0;i<NH[lay];i++)
	BKD[i]=Targets[bestJ][i];

lower[lay]->set_output(1);
for (int l=lay; l>0; l--) {
  lower[l]->backward(BKD, 1);
  for (int i=0;i<NH[l-1];i++) BKD[i] = lower[l]->back_out()[i];
  lower[l]->gradient();
}
lower[0]->backward(BKD,1);
lower[0]->gradient(I);
lower[lay]->set_output(0);

    for (int i=0;i<10;i++) {
        delete[] Targets[i];
    }
    delete[] Targets;
    return bestErr;
  }




  double NN::deepLearning(double* I, double* t, int lay) {
    int** Targets;
    Targets = new int*[10];
// Make random targets
    for (int i=0;i<10;i++) {
        Targets[i] = new int[NH[lay]];
        for (int j=0;j<NH[lay]; j++) {
            int y = rand()%2;
            if (y==0) Targets[i][j] = -1.0;
            else Targets[i][j] = 1.0;
        }
    }

    double bestErr = 100000.0;
    int bestJ = -1;
    double T[1024];
    for (int i=0;i<10;i++) {
        for (int j=0;j<NH[lay]; j++) {
            T[j] = Targets[i][j];
        }
            for (int l=lay+1;l<Nlayers;l++) {
                lower[l]->forward(T,T);
                for (int j=0;j<NH[l]; j++) {
                    T[j] = lower[l]->out()[j];
                }
            }
            upper->forward(T,T);
            double erJ = upper->f_cost(t);
            if (erJ<bestErr) {
                bestErr = erJ;
                bestJ = i;
            }
    }


  lower[0]->forward(I);
  for (int l=1;l<=lay;l++) {
	lower[l]->forward(lower[l-1]->out(),lower[l-1]->out());
  }

  double BKD[1024];
  for (int i=0;i<NH[lay];i++)
	BKD[i]=Targets[bestJ][i];

lower[lay]->set_output(1);
for (int l=lay; l>0; l--) {
  lower[l]->backward(BKD, 1);
  for (int i=0;i<NH[l-1];i++) BKD[i] = lower[l]->back_out()[i];
  lower[l]->gradient();
}
lower[0]->backward(BKD,1);
lower[0]->gradient(I);
lower[lay]->set_output(0);

    for (int i=0;i<10;i++) {
        delete[] Targets[i];
    }
    delete[] Targets;
    return bestErr;
  }



  double NN::deepLearning(int* I1, double* I2, double* t, int lay) {
    int** Targets;
    Targets = new int*[10];
// Make random targets
    for (int i=0;i<10;i++) {
        Targets[i] = new int[NH[lay]];
        for (int j=0;j<NH[lay]; j++) {
            int y = rand()%2;
            if (y==0) Targets[i][j] = -1.0;
            else Targets[i][j] = 1.0;
        }
    }

    double bestErr = 100000.0;
    int bestJ = -1;
    double T[1024];
    for (int i=0;i<10;i++) {
        for (int j=0;j<NH[lay]; j++) {
            T[j] = Targets[i][j];
        }
            for (int l=lay+1;l<Nlayers;l++) {
                lower[l]->forward(T,T);
                for (int j=0;j<NH[l]; j++) {
                    T[j] = lower[l]->out()[j];
                }
            }
            upper->forward(T,T);
            double erJ = upper->f_cost(t);
            if (erJ<bestErr) {
                bestErr = erJ;
                bestJ = i;
            }
    }


  lower[0]->forward(I1,I2);
  for (int l=1;l<=lay;l++) {
	lower[l]->forward(lower[l-1]->out(),lower[l-1]->out());
  }

  double BKD[1024];
  for (int i=0;i<NH[lay];i++)
	BKD[i]=Targets[bestJ][i];

lower[lay]->set_output(1);
for (int l=lay; l>0; l--) {
  lower[l]->backward(BKD, 1);
  for (int i=0;i<NH[l-1];i++) BKD[i] = lower[l]->back_out()[i];
  lower[l]->gradient();
}
lower[0]->backward(BKD,1);
lower[0]->gradient(I1,I2);
lower[lay]->set_output(0);

    for (int i=0;i<10;i++) {
        delete[] Targets[i];
    }
    delete[] Targets;
    return bestErr;
  }



  double NN::deepLearning(double* I1, double* I2, double* t, int lay) {
    int** Targets;
    Targets = new int*[10];
// Make random targets
    for (int i=0;i<10;i++) {
        Targets[i] = new int[NH[lay]];
        for (int j=0;j<NH[lay]; j++) {
            int y = rand()%2;
            if (y==0) Targets[i][j] = -1.0;
            else Targets[i][j] = 1.0;
        }
    }

    double bestErr = 100000.0;
    int bestJ = -1;
    double T[1024];
    for (int i=0;i<10;i++) {
        for (int j=0;j<NH[lay]; j++) {
            T[j] = Targets[i][j];
        }
            for (int l=lay+1;l<Nlayers;l++) {
                lower[l]->forward(T,T);
                for (int j=0;j<NH[l]; j++) {
                    T[j] = lower[l]->out()[j];
                }
            }
            upper->forward(T,T);
            double erJ = upper->f_cost(t);
            if (erJ<bestErr) {
                bestErr = erJ;
                bestJ = i;
            }
    }


  lower[0]->forward(I1,I2);
  for (int l=1;l<=lay;l++) {
	lower[l]->forward((lower[l-1])->out(),(lower[l-1])->out());
  }

  double BKD[1024];
  for (int i=0;i<NH[lay];i++)
	BKD[i]=Targets[bestJ][i];

lower[lay]->set_output(1);
for (int l=lay; l>0; l--) {
  lower[l]->backward(BKD, 1);
  for (int i=0;i<NH[l-1];i++) BKD[i] = lower[l]->back_out()[i];
  lower[l]->gradient();
}
lower[0]->backward(BKD,1);
lower[0]->gradient(I1,I2);
lower[lay]->set_output(0);

    for (int i=0;i<10;i++) {
        delete[] Targets[i];
    }
    delete[] Targets;
    return bestErr;
  }




void
NN::gradient(int* I, double* t)
{
  upper->gradient();
  for (int l=1;l<Nlayers;l++) {
	lower[l]->gradient();
  }
  lower[0]->gradient(I);
}
void
NN::gradient(double* I, double* t)
{
  upper->gradient();
  for (int l=1;l<Nlayers;l++) {
	lower[l]->gradient();
  }
  lower[0]->gradient(I);
}
void
NN::gradient(int* I1,double* I2, double* t)
{
  upper->gradient();
  for (int l=1;l<Nlayers;l++) {
	lower[l]->gradient();
  }
  lower[0]->gradient(I1,I2);
}
void
NN::gradient(double* I1,double* I2, double* t)
{
  upper->gradient();
  for (int l=1;l<Nlayers;l++) {
	lower[l]->gradient();
  }
  lower[0]->gradient(I1,I2);
}


void
NN::resetGradient()
{
  for (int l=0;l<Nlayers;l++) {
  	lower[l]->resetGradient();
  }
  upper->resetGradient();
}

void
NN::updateWeights(double epsilon)
{
  for (int l=0;l<Nlayers;l++) {
  	lower[l]->updateWeights(epsilon);
  }
  upper->updateWeights(epsilon);
}

void
NN::updateWeightsL1(double epsilon)
{
  for (int l=0;l<Nlayers;l++) {
  	lower[l]->updateWeightsL1(epsilon);
  }
  upper->updateWeightsL1(epsilon);
}

void
NN::updateWeightsClipped(double epsilon)
{
  for (int l=0;l<Nlayers;l++) {
  	lower[l]->updateWeightsClipped(epsilon);
  }
  upper->updateWeightsClipped(epsilon);
}

void
NN::initWeights(int seed)
{
  for (int l=0;l<Nlayers;l++) {
  	lower[l]->initWeights(seed);
  }
  upper->initWeights(seed);
}


void
NN::write(ostream& os)
{
  os << NO << " " << Nlayers<< " " << NI<< " " << NIr <<" ";
  os << which << " " << outp << " " << inp << "\n";
for (int l=0;l<Nlayers;l++) {
	os << NH[l] << " ";
}
os << "\n";
  upper->write(os);
  for (int l=0;l<Nlayers;l++) {
    lower[l]->write(os);
  }
}



