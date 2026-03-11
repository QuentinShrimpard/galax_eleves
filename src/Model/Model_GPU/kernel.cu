#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)
//#define BLOCK_SIZE 64

__global__ void compute_acc(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, float* massesGPU, int n_particles)
{	
	 //__shared__ float posMass[BLOCK_SIZE];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

		if (i >= n_particles)
		{
			return;
		}

		//cétéici, on gagne une dizaine de fps à le faire ici
		accelerationsGPU[i].x = 0.0f;
		accelerationsGPU[i].y = 0.0f; 
		accelerationsGPU[i].z = 0.0f;
	
	float ax, ay, az;
	ax = 0.0f;
	ay = 0.0f;
	az = 0.0f;
	for (int j = 0; j < n_particles; j++)
		{		
				const float diffx = positionsGPU[j].x - positionsGPU[i].x;
				const float diffy = positionsGPU[j].y - positionsGPU[i].y;
				const float diffz = positionsGPU[j].z - positionsGPU[i].z;

				float dij = diffx * diffx + diffy * diffy + diffz * diffz;

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
				}

				// accelerationsGPU[i].x += diffx * dij * massesGPU[j];
				// accelerationsGPU[i].y += diffy * dij * massesGPU[j];
				// accelerationsGPU[i].z += diffz * dij * massesGPU[j];
				ax += diffx * dij * massesGPU[j];
				ay += diffy * dij * massesGPU[j];
				az += diffz * dij * massesGPU[j];

				float dijx = diffx*dij;
				float dijy = diffy*dij;
				float dijz = diffz*dij;
				ax = __fmaf_ieee_ru(dijx, massesGPU[j], ax);
				ay = __fmaf_ieee_ru(dijy, massesGPU[j], ay);
				ax = __fmaf_ieee_ru(dijz, massesGPU[j], az);

		}
		accelerationsGPU[i].x = ax;
		accelerationsGPU[i].y = ay;
		accelerationsGPU[i].z = az;

	
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
	int nthreads = 128;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, massesGPU, n_particles);
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}


#endif // GALAX_MODEL_GPU
