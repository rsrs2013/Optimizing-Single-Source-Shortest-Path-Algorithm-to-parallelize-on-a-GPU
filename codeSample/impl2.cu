#include <vector>
#include <iostream>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

#define SSSP_INF 1073741824

/*Counts the number of edges whose source vertex distance has been changed for each warp*/
__global__ void filter_stage_1(edge_list *edgeList, uint* warp_count, uint number_of_edges, uint* is_src_changed) {

  int global_thread_id = threadIdx.x + (blockIdx.x * blockDim.x);
  int totalThreads = (blockDim.x*gridDim.x);

 //total number of warps;
  unsigned int warp_num = totalThreads/32;
  unsigned int warpid = threadIdx.x>>5;
  const unsigned int lane = threadIdx.x%32;
  unsigned int warp_first_threadId = warpid<<5;
  //unsigned int warp_last_threadId = warp_first_threadId+31;
  unsigned int global_warp_id = global_thread_id/32;

  unsigned int load = number_of_edges % warp_num == 0 ? number_of_edges/warp_num: number_of_edges/warp_num+1;
  unsigned int beg = load * global_warp_id;
  unsigned int end = (beg+load) > number_of_edges ? number_of_edges:beg+load;
  beg = beg+lane;

	uint mask = 0;
	uint maskCount = 0;
	int predicate = 0;
	edge_list *edge;

  //included i<number_of_edges, so that, it may not lead to any segmentation fault
	for (int i = beg; i< end && i< number_of_edges; i += 32) {
		edge = edgeList+i;
		predicate = (is_src_changed[edge->srcIndex] == 1);
		mask = __ballot(predicate);
		maskCount = __popc(mask);
    if(lane == 0){

       atomicAdd(warp_count+global_warp_id, maskCount);

    }
		//warp_count[global_warp_id] += maskCount;
	}
}

__global__ void filter_stage_2(uint* warp_count, uint total_warps, uint *number_of_edges) {

 /*This implementation of prefix sum is inspired from the CUDA Gems;
 http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html*/

 /*The reason it is 2048, because in this project we assume the total number of totalThreads
 are maximum 2048*/

	__shared__ uint shared_mem[2048];

	//int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  int global_thread_id = threadIdx.x + (blockIdx.x * blockDim.x);
  int totalThreads = (blockDim.x*gridDim.x);
	//int threadCount = blockDim.x * gridDim.x;
	if (global_thread_id == 0) {
		*number_of_edges = warp_count[total_warps - 1];
	}

 /*if the global thread id is less than the total warp num,
 only then participate*/
	if (global_thread_id < total_warps)
		shared_mem[global_thread_id] = warp_count[global_thread_id];

	__syncthreads();

	for (int offset = 1; offset < total_warps; offset *= 2) {
		for (int idx = global_thread_id; idx < total_warps; idx += totalThreads) {
			if (idx >= offset) {
				warp_count[idx] += warp_count[idx - offset];
			}
		}

		__syncthreads();
	}

 /*Update the number of edges, because in the next iteration,
   we need to process, these many number of edges;
 */
	if (global_thread_id == 0) {
		*number_of_edges += warp_count[total_warps - 1];
	}

 /*Since we want an exclusive prefix sum, subtract from the shared mem*/
	if (global_thread_id < total_warps)
		warp_count[global_thread_id] -= shared_mem[global_thread_id];

}

__global__ void filter_stage_3(edge_list* edgeList, uint *edge_indices, uint* warp_count, uint number_of_edges, uint* is_src_changed) {

  int global_thread_id = threadIdx.x + (blockIdx.x * blockDim.x);
  int totalThreads = (blockDim.x*gridDim.x);

 //total number of warps;
  unsigned int warp_num = totalThreads/32;
  unsigned int warpid = threadIdx.x>>5;
  const unsigned int lane = threadIdx.x%32;
  unsigned int warp_first_threadId = warpid<<5;
  unsigned int warp_last_threadId = warp_first_threadId+31;
  unsigned int global_warp_id = global_thread_id/32;

  unsigned int load = number_of_edges % warp_num == 0 ? number_of_edges/warp_num: number_of_edges/warp_num+1;
  unsigned int beg = load * global_warp_id;
  unsigned int end = (beg+load) > number_of_edges ? number_of_edges:beg+load;
  beg = beg+lane;

	uint mask = 0;
	uint localId = 0;
	int predicate = 0;
	edge_list *edge;
	uint offset =  warp_count[global_warp_id];
  //included i<number_of_edges, so that, it may not lead to any segmentation fault
	for (int i = beg; (i< end && i< number_of_edges); i += 32) {
		edge = edgeList+i;

		predicate = (is_src_changed[edge->srcIndex] == 1);
		mask = __ballot(predicate);
		localId = __popc(mask << (32-lane));
		if (predicate == 1) {
			edge_indices[offset + localId ] = i;
		}

		offset += __popc(mask);
	}
}

//neighbourHandling_kernel<<<blockNum, blockSize>>>(device_edgeList,device_edge_indices,to_process_count, device_distance_current, device_distance_prev, device_isChange, device_src_change);
__global__ void neighbourHandling_kernel(edge_list *edgeList, uint *edge_indices, uint number_of_edges, uint* distance_current, uint* distance_prev, int *isChange, uint* is_src_changed) {


    int global_thread_id = threadIdx.x + (blockIdx.x * blockDim.x);
    int totalThreads = (blockDim.x*gridDim.x);

   //total number of warps;
    unsigned int warp_num = totalThreads/32;
    unsigned int warpid = threadIdx.x>>5;
    const unsigned int lane = threadIdx.x%32;
    unsigned int warp_first_threadId = warpid<<5;
    unsigned int warp_last_threadId = warp_first_threadId+31;
    unsigned int global_warp_id = global_thread_id/32;

    unsigned int load = number_of_edges % warp_num == 0 ? number_of_edges/warp_num: number_of_edges/warp_num+1;
    unsigned int beg = load * global_warp_id;
    unsigned int end = (beg+load) > number_of_edges ? number_of_edges:beg+load;
    beg = beg+lane;
    edge_list *edge;

        for( int i = beg ; (i < end && i < number_of_edges) ; i += 32 ) {
            edge = edgeList + edge_indices[i];
            int u = edge->srcIndex;
            int v = edge->destIndex;
            int weight = edge->weight;
            int temp = distance_prev[u] + weight;
            if( temp < distance_current[v]) {
                atomicMin(&distance_current[v],temp);
                *isChange = 1;
                is_src_changed[v] = 1;
            }
        }
}

void neighborHandler(std::vector<initial_vertex> * peeps, int blockSize, int blockNum,std::vector<edge_list> *edgeList,int syncMethod, int smemMethod,int edge_list_mode) {


  edge_list *device_edgeList;
	uint* device_edge_indices;
	uint* device_distance_current;
	uint* device_distance_prev;
	uint* device_warp_count;
	uint* device_number_of_edges;
	uint* device_isChange;
	int *device_src_changed;
	int isChange;
  uint *distance_current;
  uint *distance_prev;

  int number_of_vertices = peeps->size();
  int number_of_edges = edgeList->size();


	uint to_process_count = number_of_edges;
	uint* edge_indices = new uint[number_of_edges];
	int number_of_iterations = 0;
	int total_warps = blockSize * blockNum % 32 == 0 ? blockSize * blockNum / 32 : (blockSize * blockNum / 32) + 1 ;

	cudaMalloc((void**)&device_edgeList, sizeof(edge_list)*number_of_edges);
	cudaMalloc((void**)&device_edge_indices, sizeof(uint)*number_of_edges);
	cudaMalloc((void**)&device_number_of_edges, sizeof(uint));
	cudaMalloc((void**)&device_distance_current, sizeof(uint)*number_of_vertices);
	cudaMalloc((void**)&device_distance_prev, sizeof(uint)*number_of_vertices);
	cudaMalloc((void**)&device_isChange, sizeof(uint)*number_of_vertices);
	cudaMalloc((void**)&device_warp_count, sizeof(uint)*total_warps);
	cudaMalloc((void**)&device_src_changed, sizeof(int));

  distance_current = (uint*)malloc(sizeof(uint)*number_of_vertices);
  distance_prev = (uint*)malloc(sizeof(uint)*number_of_vertices);

  /* Setting all vertices distances to infinite except the source vertex */
  std::fill_n(distance_current,number_of_vertices,SSSP_INF);
  distance_current[0] = 0;
  std::fill_n(distance_prev,number_of_vertices,SSSP_INF);
  distance_prev[0] = 0;

  //create a temporary array, because we can't use vector realted stuff on cuda;
  edge_list *temp_array = (edge_list*)malloc(sizeof(edge_list)*number_of_edges);
  std::copy(edgeList->begin(),edgeList->end(),temp_array);


	cudaMemcpy(device_edgeList, temp_array, sizeof(edge_list)*number_of_edges, cudaMemcpyHostToDevice);
	cudaMemcpy(device_number_of_edges, &number_of_edges, sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemcpy(device_distance_current, distance_current, sizeof(uint)*number_of_vertices, cudaMemcpyHostToDevice);
	cudaMemcpy(device_distance_prev, distance_prev, sizeof(uint)*number_of_vertices, cudaMemcpyHostToDevice);

	for (int i = 0; i < number_of_edges; ++i) {
		edge_indices[i] = i;
	}

	cudaMemcpy(device_edge_indices, edge_indices, sizeof(uint)*number_of_edges, cudaMemcpyHostToDevice);

	double filter_time = 0.0;
	double processing_time = 0.0;

	for (int i = 0; i < number_of_vertices-1; ++i) {
		setTime();

		cudaMemset(device_src_changed, 0, sizeof(int));
		cudaMemset(device_isChange, 0, sizeof(uint)*number_of_vertices);
		cudaMemset(device_warp_count, 0, sizeof(uint)*total_warps);


		if (syncMethod == 0)
			neighbourHandling_kernel<<<blockNum, blockSize>>>(device_edgeList, device_edge_indices, to_process_count, device_distance_current, device_distance_current, device_src_changed, device_isChange);
		else
			neighbourHandling_kernel<<<blockNum, blockSize>>>(device_edgeList, device_edge_indices, to_process_count, device_distance_current, device_distance_prev, device_src_changed, device_isChange);

		cudaDeviceSynchronize();

		if (syncMethod == 1)
			cudaMemcpy(device_distance_prev, device_distance_current, sizeof(uint)*number_of_vertices, cudaMemcpyDeviceToDevice);

		number_of_iterations++;

		cudaMemcpy(&isChange, device_src_changed, sizeof(int), cudaMemcpyDeviceToHost);

		processing_time += getTime();

		if (isChange == 0) {
			break;
		}

		setTime();

		filter_stage_1<<<blockNum, blockSize>>>(device_edgeList, device_warp_count, number_of_edges, device_isChange);

		cudaDeviceSynchronize();

		filter_stage_2<<<blockNum, blockSize>>>(device_warp_count, total_warps, device_number_of_edges);

		cudaDeviceSynchronize();

		filter_stage_3<<<blockNum, blockSize>>>(device_edgeList, device_edge_indices, device_warp_count, number_of_edges, device_isChange);

		cudaDeviceSynchronize();

		cudaMemcpy(&to_process_count, device_number_of_edges, sizeof(uint), cudaMemcpyDeviceToHost);
		cudaMemcpy(edge_indices, device_edge_indices, sizeof(uint)*number_of_edges, cudaMemcpyDeviceToHost);
		filter_time += getTime();
	}

	std::cout << "Took "<<number_of_iterations << " iterations " << processing_time + filter_time << "ms.(filter - "<<filter_time<<"ms processing - "<<processing_time<<"ms)\n";

	//cudaMemcpy(distance, device_distance_current, sizeof(uint)*number_of_vertices, cudaMemcpyDeviceToHost);
  cudaMemcpy(distance_current,device_distance_current,sizeof(int)*number_of_vertices,cudaMemcpyDeviceToHost);
  for( int i = 0 ; i < number_of_vertices ; i++ ) {
      peeps->at(i).vertexValue.distance = distance_current[i];
  }

  cudaFree(device_isChange);
	cudaFree(device_distance_current);
	cudaFree(device_distance_prev);
	cudaFree(device_warp_count);
	cudaFree(device_src_changed);
	cudaFree(device_edgeList);
	cudaFree(device_edge_indices);
	cudaFree(device_number_of_edges);
  delete[] edge_indices;

}
