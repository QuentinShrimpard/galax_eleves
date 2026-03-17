#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)
#define BLOCK_SIZE 32

__global__ void compute_acc(float3 * positionsGPU, float3 * accelerationsGPU, float* massesGPU, int n_particles)
{	
	 //__shared__ float posMass[BLOCK_SIZE];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

		if (i >= n_particles)
		{
			return;
		}

		//cétéici, on gagne une dizaine de fps à le faire ici
		// accelerationsGPU[i].x = 0.0f;
		// accelerationsGPU[i].y = 0.0f; 
		// accelerationsGPU[i].z = 0.0f;
	
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


				// dij = fmax(dij, 1.0f);
				float jeremy = std::sqrt(dij);
				float amogus = 10.0 / (jeremy * jeremy * jeremy);
				dij = (dij < 1.0f) ? 10.0f : amogus;

				/*
				if (dij < 1.0)
				{
					dij = 10.0;
				}
				else
				{
					dij = std::sqrt(dij);
					// dij = __fsqrt_rn(dij);
					dij = 10.0 / (dij * dij * dij); //yo !

					// dij = rsqrtf(dij);
					// dij = 10.0 * dij*dij*dij;

					// dij = 10.0 * powf(dij, 3.0);

					// dij = 10.0 / __powf(dij, 3.0f);
					// dij = 10.0 / (std::sqrt(dij) * dij); // pas de changement de fps mais crée de l'erreur, probablement à cause de la précision de float
				} */
				// accelerationsGPU[i].x += diffx * dij * massesGPU[j];
				// accelerationsGPU[i].y += diffy * dij * massesGPU[j];
				// accelerationsGPU[i].z += diffz * dij * massesGPU[j];
				a.x += diffx * dij * massesGPU[j];
				a.y += diffy * dij * massesGPU[j];
				a.z += diffz * dij * massesGPU[j];

				// float dijx = diffx*dij;
				// float dijy = diffy*dij;
				// float dijz = diffz*dij;
				// ax = __fmaf_ieee_rz(dijx, massesGPU[j], ax);
				// ay = __fmaf_ieee_rz(dijy, massesGPU[j], ay);
				// ax = __fmaf_ieee_rz(dijz, massesGPU[j], az);
		}
		accelerationsGPU[i].x = a.x;
		accelerationsGPU[i].y = a.y;
		accelerationsGPU[i].z = a.z;

	
}

__global__ void compute_acc23t1oie(float3 * positionsGPU, float3 * accelerationsGPU, float* massesGPU, int n_particles)
{   
    __shared__ float4 posMass[BLOCK_SIZE]; 
    
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float3 a;
	a.x = 0.0f;
	a.y = 0.0f;
	a.z = 0.0f;

	float3 posipi;
	posipi.x = 0.0f;
	posipi.y = 0.0f;
	posipi.z = 0.0f;
    if (i < n_particles) {
        posipi.x = positionsGPU[i].x;
        posipi.y = positionsGPU[i].y;
        posipi.z = positionsGPU[i].z;
    }

    // Calcul du nombre de tuiles (arrondi au supérieur)
    int numTiles = (n_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tuilax = 0; tuilax < numTiles; tuilax++)
    {       
        int idx = tuilax * BLOCK_SIZE + threadIdx.x;

        if (idx < n_particles) {
            posMass[threadIdx.x] = make_float4(positionsGPU[idx].x, 
                                               positionsGPU[idx].y, 
                                               positionsGPU[idx].z, 
                                               massesGPU[idx]);
        } else {
            posMass[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        __syncthreads();

        if (i < n_particles) {
            #pragma unroll
            for (int j = 0; j < BLOCK_SIZE; j++) {
                float4 shrek = posMass[j];
                const float diffx = shrek.x - posipi.x;
                const float diffy = shrek.y - posipi.y;
                const float diffz = shrek.z - posipi.z;

                float dij = diffx * diffx + diffy * diffy + diffz * diffz;
				// float isabelle = diffz*diffz;
				// float neymar = fmaf(diffy, diffy, isabelle);
				// float dij = fmaf(diffx, diffx, neymar);

				
				// float jeremy = __frsqrt_rn(dij);
				// float amogus = 10.0f * jeremy*jeremy*jeremy;

                float jeremy = std::sqrt(dij);
                float amogus = 10.0f / (jeremy * jeremy * jeremy);
                dij = (dij < 1.0f) ? 10.0f : amogus;
				float fiona = dij * shrek.w;

                a.x += diffx * fiona;
                a.y += diffy * fiona;
                a.z += diffz * fiona;
				// a.x = fmaf(diffx, fiona, a.x);
				// a.y = fmaf(diffy, fiona, a.y);
				// a.z = fmaf(diffz, fiona, a.z);

            }
        }
        __syncthreads();
    }
    
    if (i < n_particles) {
        accelerationsGPU[i].x = a.x;
        accelerationsGPU[i].y = a.y;
        accelerationsGPU[i].z = a.z;
    }
}

__global__ void maj_pos(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, int n_particles)
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

void update_position_cu(float3* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, float* massesGPU, int n_particles)
{
	// int nthreads = 128;
	int nthreads = BLOCK_SIZE;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc23t1oie<<<nblocks, nthreads>>>(positionsGPU, accelerationsGPU, massesGPU, n_particles);
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}


#endif // GALAX_MODEL_GPU
