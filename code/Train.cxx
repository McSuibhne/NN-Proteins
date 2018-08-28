
#include <math.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdlib.h>

#include "Model.h"
#include "Sequence.h"

#include <omp.h>

#define NTH 6

using namespace std;


class Options
{
public:
  double epsilon;
  int NU;
  int NLayers;
  int* CODE;
  int context;
  int Moore; 
  int CoF;
  int CoB;

  int cycles;
  int Cseg;		//Semi-size of window of averages input
  int Cwin;		//Semi-size of window on which SS are averaged

  int Classes;
  double Thresholds[128];

  int Step;
  int shortcut;
  int seed;
  int shuffle;
  int batch_blocks;
  int readModel;
  int readEpoch;
  int nEpochs;

  int adaptive;
  int reload;
  double belief;
  double threshold;

  int DEEP;




  void write(ostream& os=cout) 
    {
    int i;
      cout << "epsilon " << epsilon << "\n";
      cout << "NU " << NU << "\n";
      cout << "NLayers " << NLayers << "\n";
	cout << "CODE ";
	  for (i=0;i<NLayers;i++) {
	      cout << CODE[i] << " ";
	  }
      cout << "\n";
      cout << "context " << context << "\n";
      cout << "Moore " << Moore << "\n";
      cout << "Cseg " << Cseg << "\n";
      cout << "Cwin " << Cwin << "\n";
      cout << "Classes " << Classes << "\n";
	  cout << "Thresholds ";
	  for (i=0;i<Classes-1;i++) {
	      cout << Thresholds[i] << " ";
	  }
      cout << "\n";
      cout << "Step " << Step << "\n";
      cout << "shortcut " << shortcut << "\n";
      cout << "seed " << seed << "\n";
      cout << "shuffle " << shuffle << "\n";
      cout << "batch_blocks " << batch_blocks << "\n";
      cout << "readModel " << readModel << "\n";
      cout << "readEpoch " << readEpoch << "\n";
      cout << "nEpochs " << nEpochs << "\n";
      cout << "adaptive " << adaptive << "\n";
      cout << "reload " << reload << "\n";
      cout << "belief " << belief << "\n";
      cout << "threshold " << threshold << "\n";
      cout << "cycles " << cycles << "\n";
      cout << "DEEP " << DEEP << "\n";
    };
  Options(istream& is)
    {
      int i;
      char str[1024];


      while (is) {
	is >> str;
	if (!strcmp(str, "epsilon")) is >> epsilon;
	else if (!strcmp(str, "NU")) is >> NU;
     else if (!strcmp(str, "NLayers")) is >> NLayers;
        else if (!strcmp(str, "CODE")) {
            CODE = new int[NLayers];
			for (int l=0;l<NLayers;l++) is >> CODE[l];
	   }
	else if (!strcmp(str, "context")) is >> context;
	else if (!strcmp(str, "Moore")) is >> Moore; 
	else if (!strcmp(str, "Cseg")) is >> Cseg;
	else if (!strcmp(str, "Cwin")) is >> Cwin;
	else if (!strcmp(str, "Step")) is >> Step;
	else if (!strcmp(str, "Classes")) is >> Classes;
	else if (!strcmp(str, "Thresholds")) {
		for (i=0;i<Classes-1;i++)
			is >> Thresholds[i];
	}
	else if (!strcmp(str, "shortcut")) is >> shortcut;
	else if (!strcmp(str, "seed")) is >> seed;
	else if (!strcmp(str, "shuffle")) is >> shuffle;
	else if (!strcmp(str, "batch_blocks")) is >> batch_blocks;
	else if (!strcmp(str, "readModel")) is >> readModel;
	else if (!strcmp(str, "readEpoch")) is >> readEpoch;
	else if (!strcmp(str, "nEpochs")) is >> nEpochs;
	else if (!strcmp(str, "adaptive")) is >> adaptive;
	else if (!strcmp(str, "reload")) is >> reload; 
	else if (!strcmp(str, "belief")) is >> belief; 
	else if (!strcmp(str, "threshold")) is >> threshold; 
	else if (!strcmp(str, "cycles")) is >> cycles; 
	else if (!strcmp(str, "DEEP")) is >> DEEP; 
      }
    };
};



int Errcomp=100000;
void
load(int epoch, Model* M)
{
  filebuf inbuf;
  char fname[1024];
  sprintf(fname, "trained-%d.model", epoch);
  if (inbuf.open(fname, ios::in) != 0) {
    istream is(&inbuf);
    M->read(is);
  } else {
    cout << "Failed to read file " << fname;
    exit;
  }
  inbuf.close();
}

void
save(int epoch, Model* M)
{
  filebuf outbuf;
  char fname[1024];
  sprintf(fname, "trained-%d.model", epoch);
  if (outbuf.open(fname, ios::out) != 0) {
    ostream os(&outbuf);
    M->write(os);
  } else {
    cout << "Failed to write to file " << fname;
    exit;
  }
  outbuf.close();
}




void
shuffle(DataSet& D, int* seqn, int* pos)
{
  // Shuffle training set positions
  for (int k=0; k<D.totSize; k++) {
    int p1= rand()%(D.totSize);
    int p2= rand()%(D.totSize);
    int tmp=pos[p1];
    pos[p1]=pos[p2];
    pos[p2]=tmp;
    tmp=seqn[p1];
    seqn[p1]=seqn[p2];
    seqn[p2]=tmp;
  }
}



void
save_map(int epoch, Model* M)
{
  filebuf outbuf;
  int i,j;
  char fname[1024];
  sprintf(fname, "trained-%d.map", epoch);
  if (outbuf.open(fname, ios::out) != 0) {
    ostream os(&outbuf);
    for (i=0;i<101;i++) {
		for (j=0;j<101;j++) {
			os << M->getConf()[i][j] << " ";
		}
		os << "\n";
	}

  } else {
    cout << "Failed to read file " << fname;
    exit;
  }
  outbuf.close();
}


void
evaluate(Model* M, DataSet& D, char* which, int cycles)
{
	int y;

  cout << "\n counting_" << which << "_errors" << flush;
  M->resetNErrors();
  int p;
#pragma omp parallel num_threads(NTH)
{
#pragma omp for
  for (p=0; p<D.length; p++) {
	Model* tempo = new Model(M);
  	tempo->resetNErrors();
	if (strncmp(which,"test",4)==0)
    		tempo->predict(D.seq[p], cycles);
	else
    		tempo->predict(D.seq[p], cycles);
    	if (p%200==0) cout << "." << flush;
#pragma omp critical
{
	M->copyerrors(tempo);
}
delete tempo;
  }
} //pragma


double a[128];
double all=0;

for (y=0;y<M->getClasses();y++) {
	a[y]=M->getCounted()[y];
	all += M->getCounted()[y];
}

  cout << "\n\n" << which << cycles<<"_NErrors= " << M->getNErrors() << "/" << all;
  cout << " " << (double)M->getNErrors()/(double)(all)*100.0;

//  cout << "\n\n" << which << cycles << "_SError= " << sqrt(M->get_error()/all);

for (y=0;y<M->getClasses();y++) {
  cout << "\nClass" << y << cycles<<"= " << M->getNErrors_(y) << "/" << a[y];
  cout << "\t" << (double)M->getNErrors_(y)/(double)a[y]*100.0;
}
//  save_map(0,M);

  if ((strncmp(which,"test",4)==0) && (M->getNErrors()<Errcomp))
	{
	save(-10,M);
	Errcomp = M->getNErrors();
	}

  cout<<"\n";
}






void
train(Model* M, DataSet& D, DataSet& T, Options& Opt)
{

  int Gui = Opt.adaptive;
	     //Number of steps at increasing error before
	     //rescaling the learning rate.
  int gui=0;
  int cy;

  double ep=Opt.epsilon;

  cout << "Actual lrate= " << ep << "\n";
  M->setEpsilon(ep);
  cout << "Start learning\n";
  int* wait=new int[D.totSize];
  for (int p=0; p<D.totSize; p++) {
    wait[p]=0;
  }
//  srand48(9199298);
  srand(Opt.seed);

  int* seqn=new int[D.totSize];
  int* pos=new int[D.totSize];
  int mark = 0;
  for (int se=0; se<D.length; se++) {
	for (int po=0;po<(D.seq[se])->length;po++) {
		seqn[mark] = se;
		pos[mark++] = po;
	}
  }
  double previous_squared_error=1e35;

  
  for (int epoch=Opt.readEpoch+1; epoch<=Opt.readEpoch + Opt.nEpochs; epoch++) {
    if (Opt.shuffle) shuffle(D, seqn, pos);


    M->reset_squared_error();
	double sofar=0;
    int batch_cnt=0;




int blocksize = D.totSize/Opt.batch_blocks;
int covered = Opt.batch_blocks * blocksize;
int remainder = D.totSize - covered;
//cout << "Block size, covered, remainder: " << blocksize << " , " << covered << " (" << D.totSize <<") " << remainder << "\n";

int* blocksizes=new int[Opt.batch_blocks+1];

for (int bl=0;bl<Opt.batch_blocks;bl++) {
	blocksizes[bl]=blocksize;
}
blocksizes[Opt.batch_blocks] = remainder;

const int chunk = 100;



for (int bl=0;bl<Opt.batch_blocks+1;bl++) {
	int om = blocksizes[bl]/chunk;
	int omremainder = blocksizes[bl]-om*chunk;

	int* oms=new int[om+1];
	for (int bl2=0;bl2<om;bl2++) {
		oms[bl2]=chunk;
	}
	oms[om] = omremainder;


#pragma omp parallel num_threads(NTH)
{
#pragma omp for
	for (int bl2=0;bl2<om+1;bl2++) {
		Model* tempo = new Model(M);
//		save(10000,tempo);
		for (int ch=0;ch<oms[bl2];ch++) {
//			cout << pp << " " << bl << " " << bl2 << " " << ch << "\n" << flush;
			int pp = bl*blocksize+bl2*chunk+ch;
//			cout << pp << " " << bl << " " << bl2 << " " << ch << "\n" << flush;
			int se = seqn[pp];
			int po = pos[pp];
			tempo -> extimation(&D,se,po);
//			M -> extimation(&D,se,po);

     			if (pp%100000==0) {
				cout << "." << flush;
	  		}

if (rand()%(10+epoch) == 0 && Opt.DEEP == 1) {
      M->deepExtimation(&D,se,po);
}
		}
//cout << M->get_squared_error() << "\n" << flush;
//		cout << "c" << flush;
#pragma omp critical
{
		M->copytempo(tempo);
}
		delete tempo;
	}
}
//	cout << "B"<<flush;
	M->maximization();
	delete[] oms;
}


    double current_squared_error = M->get_squared_error();
    cout << "\nEpoch " << epoch << " Error= " << current_squared_error << " ";

    if (current_squared_error < previous_squared_error) {
      gui=0;
      save(0,M);
      if (Gui>0) {
  //      ep += ep0*0.01;
  //      M->setEpsilon(ep);
        }
      previous_squared_error=current_squared_error;
    } else {
      gui++;
      if ((Gui) && (gui>=Gui)) {
        gui=0;
        ep *= 0.5;
        cout << "-newEpsilon(" << ep << ")"<<flush;
//		Opt.batch_blocks = (int)(0.5*Opt.batch_blocks);
//        cout << "-newBB(" << Opt.batch_blocks << ")"<<flush;

		if (Opt.reload)
        		load(0,M);
        	M->setEpsilon(ep);
        }
    }

    if (epoch && epoch%10==0) {
      save(epoch, M);
		  	for (cy=1;cy<=Opt.cycles;cy++) {
				evaluate(M, D, "train", cy);
				evaluate(M, T, "test", cy);
//				D.write("train-predictions");
//				T.write("test-predictions");
			}
    }
    cout << "\n"<<flush;
  delete[] blocksizes;
  }
}



int
main(int argc, char** argv)
{
  if (argc<2) {
    cerr << "Usage: " << argv[0] << " option-file\n";
    exit(1);
  }
  ifstream optstream(argv[1]);
  Options Opt(optstream);
  Opt.write();

  Model* M;
  if (Opt.readModel) {
    char tmp[1024];
    sprintf(tmp, "trained-%d.model", Opt.readEpoch);
    cout << "Reading model from " << tmp << "\n";
    ifstream mstream(tmp);
    M = new Model(mstream);
  } else {
    cout << "Creating model\n"<<flush;

    M = new Model(Opt.NU, Opt.Classes, Opt.NLayers, Opt.CODE, Opt.context, Opt.Moore, Opt.Cseg, Opt.Cwin, Opt.Step, Opt.shortcut, Opt.Thresholds, Opt.cycles);

    cout << "Generating random parameters\n"<<flush;
    M->randomize(Opt.seed);
    save(-1, M);
    Opt.readEpoch = 0;
  }

  cout << "Reading train dataset\n"<<flush;
  ifstream dstream("train.dataset");
  DataSet D(dstream);
//  D.set_belief(Opt.belief);
  cout << "Reading test dataset\n"<<flush;
  ifstream tstream("test.dataset");
  DataSet T(tstream);
//  T.set_belief(Opt.belief);
  
  train(M, D, T, Opt);

  return 0;
}


