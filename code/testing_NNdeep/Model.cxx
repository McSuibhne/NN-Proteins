



#include "Model.h"




void
Model::alloc() {

	counted = new int[NY];
	nerrors_ = new int[NY];
	dcycles = new double[cycles];

	Conf=new int*[NY];
	for (int y=0;y<NY;y++)
		Conf[y]=new int[NY];
}


Model::Model(int the_NU, int the_NY, int the_NLayers,  int* the_CODE, int the_context ,int the_Moore, 
	int the_Cseg, int the_Cwin, int the_Step, int the_shortcut, double* the_Thresholds, int the_cycles) :
NU(the_NU), NY(the_NY), NLayers(the_NLayers), context(the_context), Moore(the_Moore),
Cseg(the_Cseg), Cwin(the_Cwin), Step(the_Step), shortcut(the_shortcut), cycles(the_cycles)
{

Thresholds = new double[NY];
for (int y=0;y<NY-1;y++) 
	Thresholds[y] = the_Thresholds[y];
CODE = new int[NLayers];
for (int y=0;y<NLayers;y++) 
	CODE[y] = the_CODE[y];

Net = new NN(0,(2*context+1)*NU,CODE,NLayers,NY,CODE);
Net->resetGradient();
//NetF = new BRNN(NY*(2*Cseg+2),NY,(int)(0.5*NH),context,Moore,(int)(0.5*NF),(int)(0.5*NB),(int)(0.5*NH2),CoF,CoB,Step,shortcut,0);
//NetF->resetGradient();

alloc();
}

Model::Model(Model* from) {
	NU = from->NU;
	NY = from->NY;
	NLayers = from->NLayers;
	context = from->context;
	Moore = from->Moore;
	Cseg = from->Cseg;
	Cwin = from->Cwin;
	Step = from->Step;
	shortcut = from->shortcut;
	cycles = from->cycles;

	Thresholds = new double[NY];
	for (int y=0;y<NY-1;y++) 
		Thresholds[y] = from->Thresholds[y];
	CODE = new int[NLayers];
	for (int y=0;y<NLayers;y++) 
		CODE[y] = from->CODE[y];

	Net = new NN(from->Net);
	Net->resetGradient();

	counted = new int[NY];
	nerrors_ = new int[NY];
	dcycles = new double[cycles];

	Conf=new int*[NY];
	for (int y=0;y<NY;y++)
		Conf[y]=new int[NY];
	error = 0;

}

Model::~Model() {

	delete[] Thresholds;
	delete[] CODE;
	delete Net;
	delete[] counted;
	delete[] nerrors_;
	delete[] dcycles;
	for (int y=0;y<NY;y++) {
		delete[] Conf[y];
	}
	delete[] Conf;
}




Model::Model(istream& is) {
is >> NU >> NY >> NLayers;
CODE = new int[NLayers];
for (int i=0;i<NLayers;i++) is >> CODE[i];
is >> context >> Cseg >> Cwin >> Step >> shortcut >> Moore >> cycles;

Thresholds = new double[NY];
for (int y=0;y<NY-1;y++) 
	is >> Thresholds[y];

Net = new NN(is);
Net->resetGradient();

alloc();
}





void
Model::read(istream& is) {
is >> NU >> NY >> NLayers;
for (int i=0;i<NLayers;i++) is >> CODE[i];
is >> context >> Cseg >> Cwin >> Step >> shortcut >> Moore >> cycles;

for (int y=0;y<NY-1;y++) 
	is >> Thresholds[y];

Net->read(is);
Net->resetGradient();
}




void
Model::write(ostream& os) {
os << NU << " " << NY << " " << NLayers << "\n";
for (int i=0;i<NLayers;i++) os << CODE[i] <<" ";
os <<"\n";
os << context << " "<<Cseg<<" "<<Cwin<<" "<<Step<<" "<<shortcut<<" "<<Moore<<" "<<cycles<<"\n";

for (int y=0;y<NY-1;y++) 
	os << Thresholds[y] << " ";
os << "\n";

Net->write(os);
}



void
Model::randomize(int seed) {

Net->initWeights(seed);
}



void
Model::extimation(DataSet *D, int seqn, int pos) {

int t,y;

double sum=0;
double* If = new double[NU*(2*context+1)];

//cout << seqn << " " << pos << "\n" << flush;
for (int i=NU*(pos-context);i<NU*(pos+context+1);i++) {
	if (i>=0 && i<D->seq[seqn]->length * D->seq[seqn]->attributes) {
		If[i-NU*(pos-context)] = D->seq[seqn]->u[i];
	} else {
		If[i-NU*(pos-context)] = 0;
	}
//	cout << i-NU*(pos-context) << " " << If[i-NU*(pos-context)] << "\n";
}
		int close = 0;
		for (y=0;y<NY-1;y++) {
			if (D->seq[seqn]->y[pos+1]>Thresholds[y]) {
				close =y+1;
			}
		}
		D->seq[seqn]->yc[t] = close;

double target[256];
memset(target,0,256*sizeof(double));
target[close] = 1.0;

//cout << close << " " << flush;

Net->forward(If,If);
error += Net->backward(target,1.0);
Net->gradient(If,If,target);

delete[] If;
}



void
Model::deepExtimation(DataSet *D, int seqn, int pos) {

int t,y;

double sum=0;
double* If = new double[NU*(2*context+1)];

//cout << seqn << " " << pos << "\n" << flush;
for (int i=NU*(pos-context);i<NU*(pos+context+1);i++) {
	if (i>=0 && i<D->seq[seqn]->length * D->seq[seqn]->attributes) {
		If[i-NU*(pos-context)] = D->seq[seqn]->u[i];
	} else {
		If[i-NU*(pos-context)] = 0;
	}
//	cout << i-NU*(pos-context) << " " << If[i-NU*(pos-context)] << "\n";
}
		int close = 0;
		for (y=0;y<NY-1;y++) {
			if (D->seq[seqn]->y[pos+1]>Thresholds[y]) {
				close =y+1;
			}
		}
		D->seq[seqn]->yc[t] = close;

double target[256];
memset(target,0,256*sizeof(double));
target[close] = 1.0;

//cout << close << " " << flush;

int lay = rand()%NLayers;

Net->deepLearning(If,If,target,lay);
delete[] If;
}





void
Model::maximization() {
Net->updateWeights(epsilon);
Net->resetGradient();
}

void
Model::maximizationL1() {
Net->updateWeightsL1(epsilon);
Net->resetGradient();
}
void
Model::maximizationClipped() {
Net->updateWeightsClipped(epsilon);
Net->resetGradient();
}






void
Model::predict(Sequence* seq) {

int t,y;
int a,c,cycle;//,m,maxm;
double sum=0;
double* If = new double[NU*(2*context+1)];
double* app = new double[NU*seq->length];
int* O=new int[seq->length+1];

	sum=0;

	for (t=1; t<=seq->length; t++) {
		int close = 0;
		for (y=0;y<NY-1;y++) {
			if (seq->y[t]>Thresholds[y]) {
				close =y+1;
			}
		}
		O[t]= close;
		seq->yc[t] = close;
	}

	for (t=1; t<=seq->length; t++) {
		int pos = t-1;
		for (int i=NU*(pos-context);i<NU*(pos+context+1);i++) {
			if (i>=0 && i<seq->length * seq->attributes) {
				If[i-NU*(pos-context)] = seq->u[i];
			} else {
				If[i-NU*(pos-context)] = 0;
			}
		}

		Net->forward(If,If);
		for (int i = 0; i<NY; i++) {
			app[NY*pos+i] = Net->out()[i];
		}
	}



for (t=1; t<=seq->length; t++) {
	  int pos = t-1;
	  double pred=0.0;
	  int argp=-1;

	  for (int c=0; c<NY; c++) {
		  if (app[NY*pos+c]>pred) {
			  pred = app[NY*pos+c];
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

delete[] app;
delete[] If;
delete[] O;
}



void
Model::predict(Sequence* seq, int cy) {
int temp=cycles;
cycles=cy;
predict(seq);
cycles=temp;
}





