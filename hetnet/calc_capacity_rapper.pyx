# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = darwinian_rbm_rap.cpp
"""
Cython code for using function in Clang.
These functions can be called by Python.
Last update: 2018/11/18
"""

#from libc.stdlib cimport malloc
import numpy as np
cimport numpy as np

#ctypedef np.float_t DTYPE_t

DEF N = 20#20#10  #// number of neurons (200) // if Nk landscape is used, the maximal value of N is 63!
DEF P = 10   #// partition size; must be an integer that divides N (only if blockFitness is used)
DEF T = 2   #// number of optima per block (only if blockFitness is used)
DEF B = 1#N/P #// number of blocks per sequence (only if blockFitness is used)
DEF NN = 100      #// population size of a deme (number of networks)
DEF DX = 1       #// 1st dimension of spatial deme lattice
DEF DY = 1       #// 2nd dimension of spatial deme lattice
DEF DN = 1 #// total deme number
DEF STOREMAX = 1000    #// maximal number of stored patterns per network; only used for tracking 
DEF ENVNUM = 2       #// number of different environments to cycle through (use ENVNUM = 1 and ENVLEN > MAXT for stable environment; ENVNUM = 0 won't work; Note, that if ENVNUM > 1, global optimum is set accordingly!)
DEF TYPE = 4 #// type of operator to turn input to output
DEF STORETEST = 1    #// test storedness of output (only used if TYPE == 4; otherwise sometimes it is automatically on)
DEF STORE = 1 #(TYPE == 5) || (TYPE == 6) || (TYPE == 7) || ((TYPE == 4) && STORETEST)

DEF n_v = 20#20#10 #N
DEF n_h = 20#20#10 #N
DEF epsilon_rbm = 0.1
DEF epoch_rbm = 200
DEF T_rbm = 5

# define function from Clang header file
cdef extern from "darwinian_rbm_rap.h":


    cdef struct DEME:
        float w[NN], minW, maxW, avgW, input_t[NN][N], output[NN][N], test[NN][N], maxOutput[N], weight[NN][N][N];
        int   minP, maxP, storedQ[NN], storedP[NN];
        float storedD[NN], stored[NN][STOREMAX][N];
        float w1, w2, v1[N], v2[N];
        int recombination;
        float weight_rbm[NN][n_h][n_v], b_rbm[NN][n_h], a_rbm[NN][n_v];
    ctypedef DEME DEME

        #float w[:] = np.empty(NN, dtype=np.float32)
        #float minW, maxW, avgW
        #float input_t[:,:] = np.empty((NN,N), dtype=np.float32)
        #float output[:,:] = np.empty((NN,N), dtype=np.float32)
        #float test[:,:] = np.empty((NN,N), dtype=np.float32)
        #float maxOutput[:] = np.empty(N, dtype=np.float32)
        #float weight[:,:,:] = np.empty((NN,N,N), dtype=np.float32)
        #int   minP, maxP
        #int   storedQ[:] = np.empty(NN, dtype=np.int32)
        #int   storedP[:] = np.empty(NN, dtype=np.int32)
        #float storedD[:] = np.empty(NN, dtype=np.float32)
        #float stored[:,:,:] = np.empty((NN, STOREMAX, N), dtype=np.float32)
        #float w1, w2
        #float v1[:] = np.empty(N, dtype=np.float32)
        #float v2[:] = np.empty(N, dtype=np.float32)
        #int   recombination

    cppclass Darwinian_deme:
    
        #RBM
        void train_rbm(float *v, int d, int n);
        void encode_rbm(float *v, int d, int n);
        void encode_decode_rbm(float *v, float *p_h, int d, int n);
        void update_rbm(float *v_0, float *v, float *p_h_0, float *p_h, int d, int n);
        float sigmoid_rbm(float x);

    
        # General landscape variables
        float globalOptimumS[N];
        float globalOptimumW;
        float globalOptimumAll[ENVNUM][N]
        # Block landscape
        float blockOptimaSequence[B][T][P]
        float blockOptimaFitness [B][T]

        # Auxiliary
        int matrixNeighbors[DN][8]
        int PN
        int learnCounter[DN][NN];
        DEME deme[DN];
        int test;

        float pearsonCorrelation(float *u, float *v)
        float pearsonFitness(float *v);
        int hammingDistanceN(float *v, float *u, int n)
        int hammingDistance(float *v, float *u)
        float relativeHammingDistance(float *v, float *u)
        float hammingFitness(float *v)
        int randomMinPosition(float *v, int n)
        int randomMaxPosition(float *v, int n)
        int firstMaxPosition(float *v, int n)
        int samePatternQ(float *u, float *v)
        void copyPattern(float *to, float *from_t)
        void copyPatternNN(float *to, float *from_t)
        void invertPattern(float *v)
        void printPattern(float *v)
        const char *vectorToString(float *v, int n)
        void mutatePattern(float *v, float mut)
        void exactlyMutatePattern(float *v, float mut)
        void mutateSinglebitPattern(float *v, float mut)
        void randomPattern(float *v, float mut)
        void alternatingPattern(float *v, int l, int start)
        void continuousPattern(float *v, int from_t, int to)
        void correlatePattern(float *v, float corr)
        void recombinePattern(float * u, float * v)
        void setWeights()
        void randomWeights()
        void storePattern(float *v, int d, int n)
        void trainNetwork(float *v, int d, int n)
        void trainNetworksRandom(int n)
        void updateOutput(float *v, int d, int n, int i)
        void updateNetwork(float *v, int d, int n)
        void shuffleOutputs(int d)
        float testAgainst(int e, int iter)
        int closestStored(float *v, int d, int n)
        void setBlockOptimaDefault(int env)
        float blockGlobalOptimumFitness()
        void blockGlobalOptimumSequence(float *v)
        float blockFitness(float *v)
        void setLandscape(int ls, int env)
        void setNeighborDemes(int n, int m)
    
        void evolution()
        void evolution_initialization(int t)
        void evolution_gen_environment(int t)
        DEME *evolution_gen_update(int t)
        void evolution_gen_calcfitness(int t)
        float * evolution_gen_calcbestfitness(int t)
        DEME *evolution_gen_selection(int t)
        void evolution_gen_calcmutation(int t)
        void evolution_gen_replacemutation(int t)
        void return_fitness(float *w, int d)
        void return_fitness_mutate(float *w, int d)
    
#cdef convert_to_python_float(float *ptr, int n):
#    cdef int i
#    lst=[]
#    for i in range(n):
#        lst.append(ptr[i])
#    return lst

#cdef convert_to_python(DEME *ptr, int n):
#    #print("C")
#    cdef int i
 #   lst=np.empty(0)
 #   #print("D")
 #   for i in range(n):
 #       lst.append(ptr[i])
 #   #print("E")
#    return lst
    
#cdef convert_to_C(list deme, int n):
#    cdef DEME *deme_c = <DEME *>malloc(n * sizeof(DEME*))
#    for i in range(n):
#        deme_c[i] = &(deme[i])
#    return deme_c

#cdef convert_to_float(np.ndarray[np.float, ndim=1] ptr, int n, int d):
#    #cdef float *ptr_c = <float *>malloc(n * sizeof(float *))
#    #for i in range(n):
#    #    ptr_c[i] = &(ptr[i])
#
#    cdef float *ptr_c = <float *> ptr.data
#    return_fitness(ptr_c, d)

#cdef convert_to_float_mutate(np.ndarray[np.float, ndim=1] ptr, int n, int d):
#    #cdef float *ptr_c = <float *>malloc(n * sizeof(float *))
#    #for i in range(n):
#    #    ptr_c[i] = &(ptr[i])
#    cdef float *ptr_c = <float *> ptr.data
#    return_fitness_mutate(ptr_c, d)

cdef class PyDeme:

    cdef Darwinian_deme *cDeme

    def __cinit__(self):
        self.cDeme = new Darwinian_deme()

    
    def py_evolution_initialization(self, t):
        self.cDeme.evolution_initialization(t)
    
    def py_evolution_gen_environment(self, t):
        self.cDeme.evolution_gen_environment(t)

    def py_evolution_gen_update(self, t):
        #print("A")
        deme_list = self.cDeme.evolution_gen_update(t)
        #print("F")
        cdef int i
        lst2 = []
        #print("G")
        #for i in range(1):
        for i in range(NN):
            lst = []
            #print("Here")
            for j in range(N):
                #print("Here2")
                lst.append(deme_list[0].output[i][j])
            lst2.append(lst)
        #print("I")
        #print(lst2)

        #lst.append(deme_list[0])
        #print("H")
        return lst2
    
    
        #deme_list = convert_to_python(deme_list, 1)
        #print("B")
        #return(deme_list)
    
    def py_evolution_gen_calcfitness(self, t):
        self.cDeme.evolution_gen_calcfitness(t)

    def py_evolution_gen_calcbestfitness(self, t):
        best_output = self.cDeme.evolution_gen_calcbestfitness(t)
        cdef int i
        lst=[]
        for i in range(N):
            lst.append(best_output[i])
        return lst
    
    def py_evolution_gen_selection(self, t):
        #print("A")
        deme_list = self.cDeme.evolution_gen_selection(t)
        #print("F")
        cdef int i
        lst2 = []
        #print("G")
        #for i in range(1):
        lst = []
        #print("Here")
        for j in range(N):
            #print("Here2")
            lst.append(deme_list[0].v1[j])
        lst2.append(lst)

        lst = []
        #print("Here")
        for j in range(N):
            #print("Here2")
            lst.append(deme_list[0].v2[j])
        lst2.append(lst)
        #print("I")
        #print(lst2)
        #lst.append(deme_list[0])
        #print("H")
        return lst2

        #deme_list = convert_to_python(evolution_gen_selection(t), DN)
        #return(deme_list)
    
    def py_evolution_gen_calcmutation(self, t):
        self.cDeme.evolution_gen_calcmutation(t)
    
    def py_evolution_gen_replacemutation(self, t):
        self.cDeme.evolution_gen_replacemutation(t)
    
    cpdef py_return_fitness(self, np.ndarray[np.float32_t, ndim=1] w, d):
    #def py_return_fitness(wdata, d):
        #cdef float *w_c = <float *> w.data
        cdef np.ndarray w_a = np.empty(0, dtype=np.float32)
        w_a = np.append(w_a, w)
        #print(w_a)
        #print(type(w_a))
        #print(w_a.data)
        cdef float *w_c = <float *>w_a.data
        #print(type(w_c))
        #cdef float *w_c = <float *>wdata
        #print("C")
        self.cDeme.return_fitness(w_c, d)
        #print("D")
        #convert_to_float(w, N, d)
    
    cpdef py_return_mutate_fitness(self, np.ndarray[np.float32_t, ndim=1] w, d):
        cdef np.ndarray w_a = np.empty(0, dtype=np.float32)
        w_a = np.append(w_a, w)
        #print(w_a)
        #print(type(w_a))
        #print(w_a.data)
        cdef float *w_c = <float *>w_a.data
        #print(type(w_c))
        #cdef float *w_c = <float *>wdata
        #print("C")
        self.cDeme.return_fitness_mutate(w_c, d)
        #print("D")
        #convert_to_float(w, N, d)

    
