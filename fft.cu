//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!

#include "fft.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// you may define other parameters here!
// you may define other macros here!
// you may define other functions here!

#define	R2	2		// 2-radix
#define	R4	4		// 4-radix
//#define	PI	3.141592653589793238	

//-----------------------------------------------------------------------------
//---------------------------- GPU Kernels ------------------------------------
//-----------------------------------------------------------------------------
__global__ void FFT_Iter_R2(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M, const unsigned int Ns) 
{	
	long long j = bx * blockDim.x + tx;	//Thread Id
	long long idxS = j;
	float v_r[R2]; float v_i[R2];		//2 inputs of butterfly
	//angle come from W = e^(-2*PI/(R2*Ns) * m)
	float angle = -2*PI*(j%Ns) / (Ns*R2);	//Ns=1 --> angle=0 , Ns=2 --> angle=0 & -2PI/4 , and so on...
	//Reading From Memory and multiplying in W
	float v_r_temp[R2]; float v_i_temp[R2];
	for (int r=0; r<R2; r++){
		v_r_temp[r] = x_r_d[idxS+r*N/R2];
		v_i_temp[r] = x_i_d[idxS+r*N/R2];		
		v_r[r] = (v_r_temp[r]*cos(r*angle) - v_i_temp[r]*sin(r*angle));	// (v_r + i v_i)*(cos + i sin) = (v_r*cos - v_i*sin) + i (v_r*sin + v_i*cos)
		v_i[r] = (v_r_temp[r]*sin(r*angle) + v_i_temp[r]*cos(r*angle));			
	}
	//Butterfly (R = 2)
	float v0_r = v_r[0]; float v0_i = v_i[0];
	v_r[0] = v0_r + v_r[1]; v_i[0] = v0_i + v_i[1];
	v_r[1] = v0_r - v_r[1]; v_i[1] = v0_i - v_i[1];
	//expand
	int idxD = (j/Ns)*Ns*R2 + (j%Ns);
	//Write in X_d
	for (int r=0; r<R2 ; r++){
		X_r_d[idxD + r*Ns] = v_r[r];
		X_i_d[idxD + r*Ns] = v_i[r];
	}
}

__global__ void FFT_Iter_R4(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M, const unsigned int Ns) 
{	
	long long j = bx * blockDim.x + tx;	//Thread Id
	long long idxS = j;
	float v_r[R4]; float v_i[R4];		//2 inputs of butterfly
	//angle come from W = e^(-2*PI/(R4*Ns) * m)
	float angle = -2*PI*(j%Ns) / (Ns*R4);	//Ns=1 --> angle=0 , Ns=2 --> angle=0 & -2PI/4 , and so on...
	//Reading From Memory and multiplying in W
	float v_r_temp[R4]; float v_i_temp[R4];
	for (int r=0; r<R4; r++){
		v_r_temp[r] = x_r_d[idxS+r*N/R4];
		v_i_temp[r] = x_i_d[idxS+r*N/R4];		
		v_r[r] = (v_r_temp[r]*cos(r*angle) - v_i_temp[r]*sin(r*angle));	// (v_r + i v_i)*(cos + i sin) = (v_r*cos - v_i*sin) + i (v_r*sin + v_i*cos)
		v_i[r] = (v_r_temp[r]*sin(r*angle) + v_i_temp[r]*cos(r*angle));			
	}
	//Butterfly (R = 4)
	float v0_r = v_r[0]; float v0_i = v_i[0];
	float v1_r = v_r[1]; float v1_i = v_i[1];
	float v2_r = v_r[2]; float v2_i = v_i[2];
	float v3_r = v_r[3]; 
	v_r[0] = v0_r + v1_r + v2_r + v_r[3]; 	v_i[0] = v0_i + v1_i + v2_i + v_i[3];
	v_r[1] = v0_r + v1_i - v2_r - v_i[3];	v_i[1] = v0_i - v1_r - v2_i + v_r[3];
	v_r[2] = v0_r - v1_r + v2_r - v_r[3];	v_i[2] = v0_i - v1_i + v2_i - v_i[3];
	v_r[3] = v0_r - v1_i - v2_r + v_i[3];	v_i[3] = v0_i + v1_r - v2_i - v3_r;	
	//expand
	int idxD = (j/Ns)*Ns*R4 + (j%Ns);
	//Write in X_d
	for (int r=0; r<R4 ; r++){
		X_r_d[idxD + r*Ns] = v_r[r];
		X_i_d[idxD + r*Ns] = v_i[r];
	}
}

__global__ void Copy_X_to_x (float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d){	
	long long t_Id = bx * blockDim.x + tx;	//Thread Id	
	x_r_d[t_Id] = X_r_d[t_Id];
	x_i_d[t_Id] = X_i_d[t_Id];	
}

/* __global__ void FftIteration_eff(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M, const unsigned int Ns) 
{	
	long long j = bx * blockDim.x + tx;	//Thread Id
	long long idxS = j;

	float v_r[R2]; float v_i[R2];		//2 inputs of butterfly
	//angle come from W = e^(-2*PI/(R2*Ns) * m)
	float angle = -2*PI*(j%Ns) / (Ns*R2);	//Ns=1 --> angle=0 , Ns=2 --> angle=0 & -2PI/4 , and so on...
	
	//Reading From Memory and multiplying in W
	float v_r_temp[R2]; float v_i_temp[R2];
	for (int r=0; r<R2; r++){
		v_r_temp[r] = x_r_d[idxS+r*N/R2];
		v_i_temp[r] = x_i_d[idxS+r*N/R2];
		
		v_r[r] = (v_r_temp[r]*cos(r*angle) - v_i_temp[r]*sin(r*angle));	// (v_r + i v_i)*(cos + i sin) = (v_r*cos - v_i*sin) + i (v_r*sin + v_i*cos)
		v_i[r] = (v_r_temp[r]*sin(r*angle) + v_i_temp[r]*cos(r*angle));	
		
	}
	
	//Butterfly (must changes for R2 that isnt equal 2)
	float v0_r = v_r[0]; float v0_i = v_i[0];
	v_r[0] = v0_r + v_r[1]; v_i[0] = v0_i + v_i[1];
	v_r[1] = v0_r - v_r[1]; v_i[1] = v0_i - v_i[1];
	
	//expand
	//int idxD = (j/Ns)*Ns*R2 + (j%Ns);
	
	//Write in X_d
	//for (int r=0; r<R2 ; r++){
	//	X_r_d[idxD + r*Ns] = v_r[r];
	//	X_i_d[idxD + r*Ns] = v_i[r];
	//}
	
	//--===Shared Memory for writing===--
	int idxD = (tx/Ns)^R2 + (tx%Ns);
	//Exchange
	
	//========
	idxD = bx* blockDim.x*R2 + tx;
	for (int r=0; r<R2; r++){
		X_r_d[idxD + r*blockDim.x] = v_r[r];
		X_i_d[idxD + r*blockDim.x] = v_i[r];
	}
} */
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//---------------------------- GPU Simple  ------------------------------------
//-----------------------------------------------------------------------------
void gpuKernel_simple(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M)
{	
	if ( M<11 ){	
		for (int Ns=1; Ns<N ; Ns*=R2){							
			FFT_Iter_R2 <<< 1, N/R2 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);	//blockDim.x = 2^10 , gridDim.x = 2^(M-11) , N/2 Threads
			Copy_X_to_x <<< 1, N >>>(x_r_d, x_i_d, X_r_d, X_i_d);
		}
	}
	else {
		for (int Ns=1; Ns<N ; Ns*=R2){							
			FFT_Iter_R2 <<< N/(1024*R2), 1024 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);	//blockDim.x = 2^10 , gridDim.x = 2^(M-11) , N/2 Threads		//if .1 ms tasir dare
			Copy_X_to_x <<< (1<<M-10), (1<<10) >>>(x_r_d, x_i_d, X_r_d, X_i_d);
		}
	}	
	
}




//-----------------------------------------------------------------------------
//---------------------------- GPU Efficient  ------------------------------------
//-----------------------------------------------------------------------------
void gpuKernel_efficient(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M)
{	
	int Ns;
	if (M>24) {
		//========= M=25===========
		printf("\nM=25\n");
		// 1st Iteration
		Ns = 1;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);
		// 2nd Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(X_r_d, X_i_d, x_r_d, x_i_d, N, M, Ns);
		// 3rd Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);
		// 4th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(X_r_d, X_i_d, x_r_d, x_i_d, N, M, Ns);
		// 5th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);
		// 6th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(X_r_d, X_i_d, x_r_d, x_i_d, N, M, Ns);
		// 7th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);
		// 8th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(X_r_d, X_i_d, x_r_d, x_i_d, N, M, Ns);
		// 9th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);
		// 10th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(X_r_d, X_i_d, x_r_d, x_i_d, N, M, Ns);
		// 11th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);
		// 12th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(X_r_d, X_i_d, x_r_d, x_i_d, N, M, Ns);
		// 13th Iteration (2-radix)
		Ns = N/2;
		FFT_Iter_R2 <<< N/(1024*R2), 1024 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);
		
	}
	else if (M>23){
		//======= M=24 =======
		printf("\nM=24\n");
		// 1st Iteration
		Ns = 1;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);
		// 2nd Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(X_r_d, X_i_d, x_r_d, x_i_d, N, M, Ns);
		// 3rd Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);
		// 4th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(X_r_d, X_i_d, x_r_d, x_i_d, N, M, Ns);
		// 5th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);
		// 6th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(X_r_d, X_i_d, x_r_d, x_i_d, N, M, Ns);
		// 7th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);
		// 8th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(X_r_d, X_i_d, x_r_d, x_i_d, N, M, Ns);
		// 9th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);
		// 10th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(X_r_d, X_i_d, x_r_d, x_i_d, N, M, Ns);
		// 11th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);
		// 12th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(X_r_d, X_i_d, x_r_d, x_i_d, N, M, Ns);
		// Copy x to X
		Copy_X_to_x <<< (1<<M-10), (1<<10) >>>(X_r_d, X_i_d, x_r_d, x_i_d);
	}
	else if (M>22){
		//======= M=23 =======
		printf("\nM=23\n");
		// 1st Iteration
		Ns = 1;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);
		// 2nd Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(X_r_d, X_i_d, x_r_d, x_i_d, N, M, Ns);
		// 3rd Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);
		// 4th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(X_r_d, X_i_d, x_r_d, x_i_d, N, M, Ns);
		// 5th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);
		// 6th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(X_r_d, X_i_d, x_r_d, x_i_d, N, M, Ns);
		// 7th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);
		// 8th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(X_r_d, X_i_d, x_r_d, x_i_d, N, M, Ns);
		// 9th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);
		// 10th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(X_r_d, X_i_d, x_r_d, x_i_d, N, M, Ns);
		// 11th Iteration
		Ns = Ns*R4;
		FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);
		// 12th Iteration
		Ns = N/2;
		FFT_Iter_R2 <<< N/(1024*R2), 1024 >>>(X_r_d, X_i_d, x_r_d, x_i_d, N, M, Ns);
		// copy x to X
		Copy_X_to_x <<< (1<<M-10), (1<<10) >>>(X_r_d, X_i_d, x_r_d, x_i_d);
	}
	
	//========== Other Ms ========================================================
	else if ( M<11 ){
		if (M%2){
			for (Ns=1; Ns<(N/2) ; Ns*=R4){							
				FFT_Iter_R4 <<< 1, N/R4 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);	//blockDim.x = 2^10 , gridDim.x = 2^(M-11) , N/2 Threads
				Copy_X_to_x <<< 1, N >>>(x_r_d, x_i_d, X_r_d, X_i_d);
			}
			Ns = N/2;
			FFT_Iter_R2 <<< 1, N/R2 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);	//blockDim.x = 2^10 , gridDim.x = 2^(M-11) , N/2 Threads
			Copy_X_to_x <<< 1, N >>>(x_r_d, x_i_d, X_r_d, X_i_d);
		}
		else{
			for (int Ns=1; Ns<N ; Ns*=R4){							
			FFT_Iter_R4 <<< 1, N/R4 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);	//blockDim.x = 2^10 , gridDim.x = 2^(M-11) , N/2 Threads
			Copy_X_to_x <<< 1, N >>>(x_r_d, x_i_d, X_r_d, X_i_d);
			}
		}		
	}
	else {
		if (M%2){
			for (Ns=1; Ns<(N/2) ; Ns*=R4){							
				FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);
				Copy_X_to_x <<< (1<<M-10), (1<<10) >>>(x_r_d, x_i_d, X_r_d, X_i_d);
			}
			Ns = N/2;
			FFT_Iter_R2 <<< N/(1024*R2), 1024 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);	//blockDim.x = 2^10 , gridDim.x = 2^(M-11) , N/2 Threads
			Copy_X_to_x <<< (1<<M-10), (1<<10) >>>(x_r_d, x_i_d, X_r_d, X_i_d);
		}
		else {
			for (Ns=1; Ns<N ; Ns*=R4){							
				FFT_Iter_R4 <<< N/(1024*R4), 1024 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, Ns);
				Copy_X_to_x <<< (1<<M-10), (1<<10) >>>(x_r_d, x_i_d, X_r_d, X_i_d);
			}
		}
	}
}
