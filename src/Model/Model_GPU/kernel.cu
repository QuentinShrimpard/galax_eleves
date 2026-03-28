#ifdef GALAX_MODEL_GPU
#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)
#define BLOCK_SIZE 512

__global__ void compute_acc(float4 * positionsGPU, float4 * accelerationsGPU, int n_particles)
{	
	 //__shared__ float posMass[BLOCK_SIZE];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

		if (i >= n_particles)
		{
			return;
		}
	
		float3 a;
		a.x = 0.0f;
		a.y = 0.0f;
		a.z = 0.0f;

		float3 pos;
		pos.x = positionsGPU[i].x;
		pos.y = positionsGPU[i].y;
		pos.z = positionsGPU[i].z;
		for (int j = 0; j < n_particles; j++)
		{		
				const float diffx = positionsGPU[j].x - pos.x;
				const float diffy = positionsGPU[j].y - pos.y;
				const float diffz = positionsGPU[j].z - pos.z;

				float dij = diffx * diffx + diffy * diffy + diffz * diffz;


				dij = fmax(dij, 1.0f);
				float jeremy = std::sqrt(dij);
				dij = 10.0f / (dij*jeremy);

				// dij = (dij < 1.0f) ? 10.0f : amogus;

				/*
				if (dij < 1.0)
				{
					dij = 10.0f;
				}
				else
				{
					dij = std::sqrt(dij);
					// dij = __fsqrt_rn(dij);
					dij = 10.0f / (dij * dij * dij); //yo !

					// dij = rsqrtf(dij);
					// dij = 10.0 * dij*dij*dij;

					// dij = 10.0 * powf(dij, 3.0);

					// dij = 10.0 / __powf(dij, 3.0f);
					// dij = 10.0 / (std::sqrt(dij) * dij); // pas de changement de fps mais crée de l'erreur, probablement à cause de la précision de float
				} */
				// accelerationsGPU[i].x += diffx * dij * massesGPU[j];
				// accelerationsGPU[i].y += diffy * dij * massesGPU[j];
				// accelerationsGPU[i].z += diffz * dij * massesGPU[j];
				// a.x += diffx * dij * massesGPU[j];
				// a.y += diffy * dij * massesGPU[j];
				// a.z += diffz * dij * massesGPU[j];

				a.x += diffx * dij * positionsGPU[j].w;
				a.y += diffy * dij * positionsGPU[j].w;
				a.z += diffz * dij * positionsGPU[j].w;

				// float dijx = diffx*dij;
				// float dijy = diffy*dij;
				// float dijz = diffz*dij;
				// a.x = fmaf(dijx, positionsGPU[j].w, a.x);
				// a.y = fmaf(dijy, positionsGPU[j].w, a.y);
				// a.z = fmaf(dijz, positionsGPU[j].w, a.z);
		}
		accelerationsGPU[i].x = a.x;
		accelerationsGPU[i].y = a.y;
		accelerationsGPU[i].z = a.z;
}

__global__ void compute_acc_old(float4 * positionsGPU, float4 * accelerationsGPU, int n_particles)
{
	__shared__ float4 posMass[BLOCK_SIZE]; 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		
		float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		float4 posi = make_float4(0.0f,0.0f,0.0f,0.0f);
		posi = (i < n_particles) ? positionsGPU[i] : posi;
		// posi = positionsGPU[i];

		int numTiles = (n_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;
	
		for (int tuilax = 0; tuilax < numTiles; tuilax++)
    	{       
        int idx = tuilax * BLOCK_SIZE + threadIdx.x;
		// posMass[threadIdx.x] = positionsGPU[idx];
		posMass[threadIdx.x] = (idx < n_particles) ? positionsGPU[idx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);


        __syncthreads();
		// #pragma unroll 512
    	for (int j = 0; j < BLOCK_SIZE; j++)
        {		
			float4 temp = posMass[j];
            const float diffx = temp.x - posi.x;
            const float diffy = temp.y - posi.y;
            const float diffz = temp.z - posi.z;

            // float dij = diffx * diffx + diffy * diffy + diffz * diffz;
			float test1 = diffz*diffz;
			float test2 = fmaf(diffy, diffy, test1);
			float dij = fmaf(diffx, diffx, test2);

			dij = std::sqrt(fmaxf(dij, 1.0f));
			dij = 10.0 / (dij * dij * dij);

			float fiona = dij * temp.w;
			a.x = fmaf(diffx, fiona, a.x);
			a.y = fmaf(diffy, fiona, a.y);
			a.z = fmaf(diffz, fiona, a.z);

            // a.x += diffx * dij * temp.w;
            // a.y += diffy * dij * temp.w;
            // a.z += diffz * dij * temp.w;
        }
		__syncthreads();
		}
	accelerationsGPU[i] = a;
}

// c'est celui là le bon
__global__ void compute_acc23t1oie(float4 * positionsGPU, float4 * accelerationsGPU, int n_particles)
{   
    __shared__ float4 posMass[BLOCK_SIZE]; 
    
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	float4 posi = make_float4(0.0f,0.0f,0.0f,0.0f);

	posi = (i < n_particles) ? positionsGPU[i] : posi;

    // Calcul du nombre de tuiles (arrondi au supérieur)
    int numTiles = (n_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;

	// #pragma unroll
    for (int tuilax = 0; tuilax < numTiles; tuilax++)
    {       
        int idx = tuilax * BLOCK_SIZE + threadIdx.x;
		posMass[threadIdx.x] = (idx < n_particles) ? positionsGPU[idx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		// posMass[threadIdx.x] = positionsGPU[idx];

        __syncthreads();

            // #pragma unroll 512
            for (int j = 0; j < BLOCK_SIZE; j++) {
                float4 shrek = posMass[j];
                const float diffx = shrek.x - posi.x;
                const float diffy = shrek.y - posi.y;
                const float diffz = shrek.z - posi.z;

                // float dij = diffx * diffx + diffy * diffy + diffz * diffz;
				float isabelle = diffz*diffz;
				float neymar = fmaf(diffy, diffy, isabelle);
				float dij = fmaf(diffx, diffx, neymar);

				dij = fmaxf(dij, 1.0f);
				dij = 10.0f * rsqrtf(dij * dij * dij);


				float fiona = dij * shrek.w;
                // a.x += diffx * dijmass;
                // a.y += diffy * dijmass;
                // a.z += diffz * dijmass;
				// a.x += diffx * dij * shrek.w;
				// a.y += diffy * dij * shrek.w;
				// a.z += diffz * dij * shrek.w;
				a.x = fmaf(diffx, fiona, a.x);
				a.y = fmaf(diffy, fiona, a.y);
				a.z = fmaf(diffz, fiona, a.z);

            
        }
        __syncthreads();
    }
    if (i<n_particles){
		accelerationsGPU[i] = a;
	}
}

__global__ void maj_pos(float4 * positionsGPU, float4 * velocitiesGPU, float4 * accelerationsGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n_particles)
		{
			return;
		}
		velocitiesGPU[i].x += accelerationsGPU[i].x * 2.0f;
		velocitiesGPU[i].y += accelerationsGPU[i].y * 2.0f;
		velocitiesGPU[i].z += accelerationsGPU[i].z * 2.0f;
		positionsGPU[i].x += velocitiesGPU[i].x * 0.1f;
		positionsGPU[i].y += velocitiesGPU[i].y * 0.1f;
		positionsGPU[i].z += velocitiesGPU[i].z * 0.1f;

		// accelerationsGPU[i].x = 0.0f;
		// accelerationsGPU[i].y = 0.0f;
		// accelerationsGPU[i].z = 0.0f;
}

void update_position_cu(float4* positionsGPU, float4* velocitiesGPU, float4* accelerationsGPU, int n_particles)
{
	int nthreads = BLOCK_SIZE;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc23t1oie<<<nblocks, nthreads>>>(positionsGPU, accelerationsGPU, n_particles);
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
	// int nthreads = BLOCK_SIZE;
	// int nblocks =  (n_particles + (nthreads -1)) / nthreads;

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudaEventRecord(start); // Début du chrono GPU
	// compute_acc23t1oie<<<nblocks, nthreads>>>(positionsGPU, accelerationsGPU, n_particles);
	// maj_pos<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
    // cudaEventRecord(stop); // Fin du chrono GPU
    
    // cudaEventSynchronize(stop);
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    
	// printf("FPS GPU PUR %f\n", 1000.0f/milliseconds);

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
}


#endif // GALAX_MODEL_GPU
