#ifndef __CLANG_SAMPLE_H__
#define __CLANG_SAMPLE_H__

#define N           20//20 //10  // number of neurons (200) // if Nk landscape is used, the maximal value of N is 63!
#define P           10   // partition size; must be an integer that divides N (only if blockFitness is used)
#define T           2   // number of optima per block (only if blockFitness is used)
#define B       	  N/P // number of blocks per sequence (only if blockFitness is used)
#define NN       100      // population size of a deme (number of networks)
//#define DX       1       // 1st dimension of spatial deme lattice
//#define DY       1       // 2nd dimension of spatial deme lattice
#define DN       1 // total deme number
#define STOREMAX      1000    // maximal number of stored patterns per network; only used for tracking 
#define ENVNUM     2       // number of different environments to cycle through (use ENVNUM = 1 and ENVLEN > MAXT for stable environment; ENVNUM = 0 won't work; Note, that if ENVNUM > 1, global optimum is set accordingly!)
#define TYPE 4 // type of operator to turn input to output
#define STORETEST     1    // test storedness of output (only used if TYPE == 4; otherwise sometimes it is automatically on)

#define n_v 20//20 //10 //N
#define n_h 20//20 //10 //N
#define epsilon_rbm 0.1
#define epoch_rbm 200
#define T_rbm 5

typedef struct DEME {
  float w[NN], minW, maxW, avgW, input_t[NN][N], output[NN][N], test[NN][N], maxOutput[N], weight[NN][N][N];
	int   minP, maxP, storedQ[NN], storedP[NN];
	float storedD[NN], stored[NN][STOREMAX][N];
	float w1, w2, v1[N], v2[N];
	int recombination;
	float weight_rbm[NN][n_h][n_v], b_rbm[NN][n_h], a_rbm[NN][n_v];
} DEME;

class Darwinian_deme {
public:

// RBM
void train_rbm(float *v, int d, int n);
void encode_rbm(float *v, int d, int n);
void encode_decode_rbm(float *v, float *p_h, int d, int n);
void update_rbm(float *v_0, float *v, float *p_h_0, float *p_h, int d, int n);
float sigmoid_rbm(float x);


// General landscape variables
float globalOptimumS[N];
float globalOptimumW;
float globalOptimumAll[ENVNUM][N]; // store optimum for each environment (for Pearson or Hamming fitnesses)
// Block landscape
float blockOptimaSequence[B][T][P]; // store local optima sequences for each block (for block fitness only)
float blockOptimaFitness [B][T];    // store local optima fitnesses for each block (for block fitness only)

// Auxiliary
int matrixNeighbors[DN][8]; // spatial deme neighbors
int STORE = (TYPE == 5) || (TYPE == 6) || (TYPE == 7) || ((TYPE == 4) && STORETEST); // store learned patterns in `stored` array
int PN; // if `STORE`, store this many learned patterns
int learnCounter[DN][NN];
DEME deme[DN];
int test;


float pearsonCorrelation(float *u, float *v); // Pearson product-moment correlation coefficient of vectors `u` and `v`
float pearsonFitness(float *v);
int hammingDistanceN(float *v, float *u, int n); // standard HD up to length `n`
int hammingDistance(float *v, float *u); // standard HD of length N
float relativeHammingDistance(float *v, float *u); // HD/N
float hammingFitness(float *v); // 1 - (HD/N)
int randomMinPosition(float *v, int n); // selects the position of the smallest value in vector `v` up to length `n`; if there are multiple 
int randomMaxPosition(float *v, int n); // selects the position of the largest value in vector `v` up to length `n`; if there are multiple 
int firstMaxPosition(float *v, int n); // selects the position of the largest value in vector `v` up to length `n`; if there are multiple 
int samePatternQ(float *u, float *v); // boolean test of pattern identity
void copyPattern(float *to, float *from_t);
void copyPatternN(float *to, float *from_t, int n);
void copyPatternNN(float *to, float *from_t);
void invertPattern(float *v); // flip each bit in vector `v`
void printPattern(float *v);
const char *vectorToString(float *v, int n); // cannot be called multiple times in e.g. printf!
void mutatePattern(float *v, float mut); // mutate pattern per-digit mutation rate `mut`
void exactlyMutatePattern(float *v, float mut); // mutates exactly `mut*N` bits (maximally N)
void mutateSinglebitPattern(float *v, float mut);
void randomPattern(float *v, float mut); // generates a random pattern with `mut` probability that a bit is +1
void alternatingPattern(float *v, int l, int start); // alternating +1.0 and -1.0 with partition length `l`, starting with `start` (1 -> +1.0, 
void continuousPattern(float *v, int from_t, int to); // [---..+++..---] where + subpattern runs from `from` to `to-1`
void correlatePattern(float *v, float corr); // generate a partially uncorrelated version of vector `v` (saving back to `v`)
void recombinePattern(float * u, float * v); // two-point recombination
void setWeights();
void randomWeights();
void storePattern(float *v, int d, int n); // tests if `v` is stored in network `n` at deme `d` or not, and pushes/stores it to most recent p
void trainNetwork(float *v, int d, int n); // training network `n` in deme `d` with vector `v`, updating `weight`
void trainNetworksRandom(int n); // train each network on `n` random patterns (and store these patterns if `STORE == 1`)
void updateOutput(float *v, int d, int n, int i); // update of neuron `i` in network `n` in deme `d` with threshold neuron model
void updateNetwork(float *v, int d, int n); // updating network 'n' in deme `d` with input `v`; output is written into `v`
void shuffleOutputs(int d); // shuffle `output` population for deme `d`
float testAgainst(int e, int iter); // tests all networks of all demes `iter` times against the global optimum of environment `e`, returning average per-bit relative distance
int closestStored(float *v, int d, int n); // returns the index of the stored pattern that is closest to `v` in deme `d`, network `n`
void setBlockOptimaDefault(int env); // sets the same T optima (sequence and fitness) for the B blocks in environment `env` {1, 2, ...}
float blockGlobalOptimumFitness(); // calculates global optimum using `blockOptimaSequence` and `blockOptimaFitness`
void blockGlobalOptimumSequence(float *v); // calculates global optimum using `blockOptimaSequence` and `blockOptimaFitness`
float blockFitness(float *v); // building block fitness; rescales range (min, max) to (min/max, 1.0)
void setLandscape(int ls, int env); // sets `globalOptimumS` and `globalOptimumW`
void setNeighborDemes(int n, int m); // sets up a lookup table for neughboring positions for each matrix element; 8 neighbours, no periodic boundary


void evolution();
void evolution_initialization(int t);
void evolution_gen_environment(int t);
DEME *evolution_gen_update(int t);
void evolution_gen_calcfitness(int t);
float *evolution_gen_calcbestfitness(int t);
DEME *evolution_gen_selection(int t);
void evolution_gen_calcmutation(int t);
void evolution_gen_replacemutation(int t);
void return_fitness(float *w, int d);
void return_fitness_mutate(float *w, int d);

};

#endif // __CLANG_SAMPLE_H__
