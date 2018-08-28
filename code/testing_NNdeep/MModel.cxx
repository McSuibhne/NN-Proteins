



#include "MModel.h"




void
MModel::alloc() {

	counted = new int[NY[0]];
	nerrors_ = new int[NY[0]];
	dcycles = new double[cycles];

	Conf=new int*[NY[0]];
	for (int y=0;y<NY[0];y++)
		Conf[y]=new int[NY[0]];
}





MModel::MModel(istream& is) {
char fname[256];
filebuf inbuf;

is >> nModels;
Net=new NN*[nModels];

  NU=new int[nModels];
  NY=new int[nModels];
  NLayers=new int[nModels];
  CODE = new int*[nModels];

  context=new int[nModels];
  Moore=new int[nModels];

  Cseg=new int[nModels];
  Cwin=new int[nModels];
  shortcut=new int[nModels];
  Step=new int[nModels];

  Thresholds=new double*[nModels];

for (int n=0;n<nModels; n++) {
  is>>fname;
  inbuf.open(fname, ios::in);
  istream is2(&inbuf);

  is2 >> NU[n]>>NY[n]>>NLayers[n];
  CODE[n] = new int[NLayers[n]];
  for (int y=0;y<NLayers[n];y++) is2 >> CODE[n][y];

  is2>>context[n]>>Cseg[n]>>Cwin[n];
  is2>>Step[n]>>shortcut[n]>>Moore[n]>>cycles;
  Thresholds[n] = new double[NY[n]];
  for (int y=0;y<NY[n]-1;y++) {
        is2 >> Thresholds[n][y];
  }

  Net[n] = new NN(is2);
  Net[n]->resetGradient();

  inbuf.close();
}

alloc();

}







void
MModel::predict(Sequence* seq) {

int t,y;
int a,c,cycle;
double sum=0;
int* O=new int[seq->length+1];

	for (t=1; t<=seq->length; t++) {
		int close = 0;
		for (y=0;y<NY[0]-1;y++) {
			if (seq->y[t]>Thresholds[0][y]) {
				close =y+1;
			}
		}
		O[t]= close;
		seq->yc[t] = close;
	}

	double* app = new double[NY[0]*seq->length];
	memset(app,0,sizeof(double)*NY[0]*seq->length);

for (int n=0;n<nModels; n++) {
	double* If = new double[NU[n]*(2*context[n]+1)];

	for (t=1; t<=seq->length; t++) {
		int pos = t-1;
		for (int i=NU[n]*(pos-context[n]);i<NU[n]*(pos+context[n]+1);i++) {
			if (i>=0 && i<seq->length * seq->attributes) {
				If[i-NU[n]*(pos-context[n])] = seq->u[i];
			} else {
				If[i-NU[n]*(pos-context[n])] = 0;
			}
		}

		Net[n]->forward(If,If);
		for (int i = 0; i<NY[n]; i++) {
			app[NY[n]*pos+i] += Net[n]->out()[i]/nModels;
		}
	}
	delete[] If;
}



for (t=1; t<=seq->length; t++) {
	  int pos = t-1;
	  double pred=0.0;
	  int argp=-1;

	  for (int c=0; c<NY[0]; c++) {
		  if (app[NY[0]*pos+c]>pred) {
			  pred = app[NY[0]*pos+c];
//cout << app[NY*pos+c] << " " << flush;
			  argp = c;
		  }
	  }
	  seq->y_pred[t] = argp;
}


for (t=1; t<=seq->length; t++) {

	  if (seq->y_pred[t]!=seq->yc[t]) {
		    nerrors++;
		    nerrors_[seq->yc[t]]++;
	  }
	
	if (seq->yc[t] != -1 && seq->y_pred[t] != -1) {
	  Conf[seq->y_pred[t]][seq->yc[t]]++;
	  counted[seq->yc[t]]++;
	}
}

delete[] O;
delete[] app;
}
















void
MModel::predict(Sequence* seq, double thre) {

int t,y;
int a,c,cycle;
double sum=0;
int* O=new int[seq->length+1];

	for (t=1; t<=seq->length; t++) {
		int close = 0;
		for (y=0;y<NY[0]-1;y++) {
			if (seq->y[t]>Thresholds[0][y]) {
				close =y+1;
			}
		}
		O[t]= close;
		seq->yc[t] = close;
	}

	double* app = new double[NY[0]*seq->length];
	memset(app,0,sizeof(double)*NY[0]*seq->length);

for (int n=0;n<nModels; n++) {
	double* If = new double[NU[n]*(2*context[n]+1)];

	for (t=1; t<=seq->length; t++) {
		int pos = t-1;
		for (int i=NU[n]*(pos-context[n]);i<NU[n]*(pos+context[n]+1);i++) {
			if (i>=0 && i<seq->length * seq->attributes) {
				If[i-NU[n]*(pos-context[n])] = seq->u[i];
			} else {
				If[i-NU[n]*(pos-context[n])] = 0;
			}
		}

		Net[n]->forward(If,If);
		for (int i = 0; i<NY[n]; i++) {
			app[NY[n]*pos+i] += Net[n]->out()[i]/nModels;
		}
	}
	delete[] If;
}



for (t=1; t<=seq->length; t++) {
	  int pos = t-1;
	  double pred=0.0;
	  int argp=-1;

	  if (app[NY[0]*pos]>thre) {
		  pred = app[NY[0]*pos+1];
		  argp = 0;
	  } else {
		  pred = app[NY[0]*pos+1];
		  argp = 1;
	  }

	  seq->y_pred[t] = argp;
}


for (t=1; t<=seq->length; t++) {

	  if (seq->y_pred[t]!=seq->yc[t]) {
		    nerrors++;
		    nerrors_[seq->yc[t]]++;
	  }
	
	if (seq->yc[t] != -1 && seq->y_pred[t] != -1) {
	  Conf[seq->y_pred[t]][seq->yc[t]]++;
	  counted[seq->yc[t]]++;
	}
}

delete[] O;
delete[] app;
}

