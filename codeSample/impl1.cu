#include <vector>
#include <iostream>
#include <algorithm>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

#define SSSP_INF 1073741824

#define WARP_SZ 32
__device__
inline int lane_id(void) { return threadIdx.x % WARP_SZ; }

__global__ void pulling_kernel(std::vector<initial_vertex> * peeps, int offset, int * isChange){

    //update me based on my neighbors. Toggle isChange as needed.
    //offset will tell you who I am.

}

__device__ uint segmented_scan(const int lane,const uint* dest_vertex,uint* ptrs){


     //printf("Doing a segmented scan\n");
     if(lane>=1 && dest_vertex[threadIdx.x] ==  dest_vertex[threadIdx.x-1]){

           if(ptrs[threadIdx.x-1] < ptrs[threadIdx.x]){
                ptrs[threadIdx.x] = ptrs[threadIdx.x-1];
           }
     }

     if(lane>=2 && dest_vertex[threadIdx.x] ==  dest_vertex[threadIdx.x-2]){

         if(ptrs[threadIdx.x-2] < ptrs[threadIdx.x]){
              ptrs[threadIdx.x] = ptrs[threadIdx.x-2];
         }
     }

     if(lane>=4 && dest_vertex[threadIdx.x] ==  dest_vertex[threadIdx.x-4]){

         if(ptrs[threadIdx.x-4] < ptrs[threadIdx.x]){
              ptrs[threadIdx.x] = ptrs[threadIdx.x-4];
         }
     }

     if(lane>=8 && dest_vertex[threadIdx.x] ==  dest_vertex[threadIdx.x-8]){

        if(ptrs[threadIdx.x-8] < ptrs[threadIdx.x]){
            ptrs[threadIdx.x] = ptrs[threadIdx.x-8];
        }

     }

     if(lane>=16 && dest_vertex[threadIdx.x] ==  dest_vertex[threadIdx.x-16]){

       if(ptrs[threadIdx.x-16] < ptrs[threadIdx.x]){
           ptrs[threadIdx.x] = ptrs[threadIdx.x-16];
       }
     }

    return ptrs[threadIdx.x];

}

__global__ void incore(edge_list *edgeList, int number_of_edges , int *distance, int *isChange){


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

      for( int i = beg ; (i < end && i < number_of_edges) ; i += 32 ) {
          int u = edgeList[i].srcIndex;
          int v = edgeList[i].destIndex;
          int weight = edgeList[i].weight;
          int temp = distance[u] + weight;
          if( temp < distance[v]) {
              atomicMin(&distance[v],temp);
              *isChange = 1;
          }
      }
}


__global__ void sharedMemImpl2(edge_list *edgeList, int number_of_edges, int *distance_prev, int *distance_current, int *anyChange) {

    __shared__ unsigned int ptrs[1024];
    __shared__ unsigned int shared_dest_vertices[1024];

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
    beg  = beg + lane;

    for( int i = beg ; (i < end  && i <number_of_edges) ; i += 32 ) {


        int u = edgeList[i].srcIndex;
        int v = edgeList[i].destIndex;
        int weight = edgeList[i].weight;
        int calDistance = distance_prev[u] + weight;


        ptrs[threadIdx.x] = calDistance;
        shared_dest_vertices[threadIdx.x] = v;

        __syncthreads();

        uint minVal = segmented_scan(lane,shared_dest_vertices,ptrs);

        __syncthreads();

        if( i == end-1 ) {
            if( distance_current[v] > minVal)
                *anyChange = 1;
            atomicMin(&distance_current[v],minVal);
        }
        else if ( threadIdx.x != warp_last_threadId ) {
            if(shared_dest_vertices[threadIdx.x] != shared_dest_vertices[threadIdx.x+1] ) {
                if( distance_current[v] > minVal)
                    *anyChange = 1;
                atomicMin(&distance_current[v],minVal);
            }
        }
        else {
            if( distance_current[v] > minVal)
                *anyChange = 1;
            atomicMin(&distance_current[v],minVal);
        }
    }
}



__global__ void outcore(edge_list *edgeList, int number_of_edges , int number_of_vertices, int *distance_prev, int *distance_current, int *isChange){


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

  for( int i = beg ; (i < end && i < number_of_edges) ; i += 32 ) {
      int u = edgeList[i].srcIndex;
      int v = edgeList[i].destIndex;
      int weight = edgeList[i].weight;
      int temp = distance_prev[u] + weight;
      if( temp < distance_current[v]) {
          atomicMin(&distance_current[v],temp);
          *isChange = 1;
      }
  }


}




__global__ void outcore_earlier(edge_list *edgeList, int number_of_edges , int number_of_vertices, int *distance_prev, int *distance_current, int *isChange){


    //printf("Outcore implementation\n");
    int thread_id = threadIdx.x + (blockIdx.x * blockDim.x);
    int totalThreads = (blockDim.x*gridDim.x);

    int iterations = (number_of_edges % totalThreads == 0 )? number_of_edges / totalThreads : number_of_edges/ totalThreads + 1;

    for( int i = 0 ; i < iterations ; i++ ) {

        int dataId = thread_id + i*totalThreads;
         //printf("The thread_id is %d\n",thread_id);
         //printf("The data id is %d\n",dataId);

        if( dataId < number_of_edges ) {

            int u = edgeList[dataId].srcIndex;
            int v = edgeList[dataId].destIndex;
            int weight = edgeList[dataId].weight;
            if( distance_prev[u] + weight < distance_current[v]) {
                atomicMin(&distance_current[v],distance_prev[u]+weight);
                *isChange = 1;
            }
        }
    }



}



void puller(std::vector<initial_vertex> * peeps, int blockSize, int blockNum,std::vector<edge_list> *edgeList,int syncMethod, int smemMethod,int edge_list_mode){

    /*
     * Do all the things here!
     **/

    //if the syncMethod is outcore (1) then call the kernel for outcore
         int number_of_iterations = 0;

        if(smemMethod == 1 && syncMethod == 1){
         //Pure Outcore;
          setTime();
          /* Arrays on Host*/
          int *distance_current;
          int *distance_prev;
          int *isChange;

          /*Array on Device*/
          int *device_distance_prev;
          int *device_distance_current;
          int *device_isChange;
          edge_list *device_edgeList;


          double kernel_start_time;
          double kernel_end_time;
          double kernelTimeTaken = 0;

          /* get the number of vertices and edges of the graph */
          uint vertices = peeps->size();
          uint edges = edgeList->size();

          distance_current = (int*)malloc(sizeof(int)*vertices);
          distance_prev = (int*)malloc(sizeof(int)*vertices);
          isChange = (int *)malloc(sizeof(int));

          /* Setting all vertices distances to infinite except the source vertex */
          std::fill_n(distance_current,vertices,SSSP_INF);
          distance_current[0] = 0;
          std::fill_n(distance_prev,vertices,SSSP_INF);
          distance_prev[0] = 0;

          //create a temporary array, because we can't use vector realted stuff on cuda;
          edge_list *temp_array = (edge_list*)malloc(sizeof(edge_list)*edges);
          std::copy(edgeList->begin(),edgeList->end(),temp_array);

          //Allocate the memory on the device;
          cudaMalloc((void **)&device_edgeList,sizeof(edge_list)*edges);
          cudaMalloc((void **)&device_distance_prev, sizeof(int)*vertices );
          cudaMalloc((void **)&device_distance_current, sizeof(int)*vertices);
          cudaMalloc((void **)&device_isChange,sizeof(int));

          //Copy all the vertices to the device;
          cudaMemcpy(device_distance_current,distance_current,sizeof(int)*vertices,cudaMemcpyHostToDevice);
          cudaMemcpy(device_distance_prev,distance_prev,sizeof(int)*vertices,cudaMemcpyHostToDevice);
          cudaMemcpy(device_edgeList,temp_array,sizeof(edge_list)*edges,cudaMemcpyHostToDevice);

          //Method for outcore implementation;

              for(uint i = 0 ; i < vertices ; i++) {

                  *isChange = 0;
                  cudaMemcpy(device_isChange,isChange,sizeof(int),cudaMemcpyHostToDevice);
                  kernel_start_time = getTime();
                  outcore<<< blockNum, blockSize >>>(device_edgeList,edges,vertices,device_distance_prev,device_distance_current,device_isChange);
                  cudaDeviceSynchronize();
                  kernel_end_time = getTime();
                  kernelTimeTaken = kernelTimeTaken + (kernel_end_time - kernel_start_time);
                  cudaMemcpy(isChange,device_isChange,sizeof(int),cudaMemcpyDeviceToHost);
                  number_of_iterations++;

                  if( *isChange == 0 ) {
                      break;
                  }
                  int *temp = device_distance_current;
                  device_distance_current = device_distance_prev;
                  device_distance_prev = temp;



              }

              //kernel_end_time = getTime();

              cudaMemcpy(distance_current,device_distance_current,sizeof(int)*vertices,cudaMemcpyDeviceToHost);
              for( int i = 0 ; i < vertices ; i++ ) {
                  peeps->at(i).vertexValue.distance = distance_current[i];
              }


              //Free the memory on the device;
              cudaFree(device_edgeList);
              cudaFree(device_distance_current);
              cudaFree(device_distance_prev);
              cudaFree(device_isChange);

              std::cout << "Total time taken :" << getTime() << "ms.\n";
              std::cout << "Time taken by Kernel : " << kernelTimeTaken<< "ms.\n";
              std::cout << "Total number of iterations are: "<<number_of_iterations<<"\n";
              std::cout << "Took " << getTime() << "ms.\n";


        }else if(syncMethod == 0 && smemMethod == 1){

           setTime();
          /* get the number of vertices and edges of the graph */
          uint vertices = peeps->size();
          uint edges = edgeList->size();

          /* Arrays on Host*/
          int *distance;
          int *isChange;

          /*Array on Device*/
          int *device_distance;
          int *device_isChange;
          edge_list *device_edgeList;

          double kernel_start_time;
          double kernel_end_time;
          double kernelTimeTaken = 0;

          distance = (int*)malloc(sizeof(int)*vertices);;
          isChange = (int *)malloc(sizeof(int));;
          //create a temporary array, because we can't use vector realted stuff on cuda;
          edge_list *temp_array = (edge_list*)malloc(sizeof(edge_list)*edges);
          std::copy(edgeList->begin(),edgeList->end(),temp_array);

          std::fill_n(distance,vertices,SSSP_INF);
          distance[0] = 0;

          //Allocate the memory on the device;
          cudaMalloc((void **)&device_edgeList,sizeof(edge_list)*edges);
          cudaMalloc((void **)&device_distance, sizeof(int)*vertices );
          cudaMalloc((void **)&device_isChange,sizeof(int));

          cudaMemcpy(device_distance,distance,sizeof(int)*vertices,cudaMemcpyHostToDevice);
          cudaMemcpy(device_edgeList,temp_array,sizeof(edge_list)*edges,cudaMemcpyHostToDevice);

          for(uint i = 0 ; i < vertices ; i++) {

              *isChange = 0;
              cudaMemcpy(device_isChange,isChange,sizeof(int),cudaMemcpyHostToDevice);

              /* Kernel call and measure the time taken by the kernel */
              kernel_start_time = getTime();
              incore<<< blockNum,blockSize >>>(device_edgeList,edges,device_distance,device_isChange);
              cudaDeviceSynchronize();
              kernel_end_time = getTime();
              kernelTimeTaken = kernelTimeTaken+(kernel_end_time - kernel_start_time);

              cudaMemcpy(isChange,device_isChange,sizeof(int),cudaMemcpyDeviceToHost);

              number_of_iterations++;
              if( *isChange == 0 ) {
                  std::cout << "The loop stops at : " << i << "\n";
                  break;
              }

          }

          cudaMemcpy(distance,device_distance,sizeof(int)*vertices,cudaMemcpyDeviceToHost);

          for( int i = 0 ; i < vertices ; i++ ) {
              peeps->at(i).vertexValue.distance = distance[i];
          }

          /* Free the memory allocated */
          cudaFree(device_edgeList);
          cudaFree(device_distance);
          cudaFree(device_isChange);
          std::cout << "Total time taken :" << getTime() << "ms.\n";
          std::cout << "Time taken by Kernel : " << kernelTimeTaken << "ms.\n";
          std::cout << "Total number of iterations are: "<<number_of_iterations<<"\n";
          std::cout << "Took " << getTime() << "ms.\n";

      }else if(syncMethod == 1 && smemMethod == 0 && edge_list_mode == 2 ){

          //shared memory with destination order;

          setTime();
          /* Arrays on Host*/
          int *distance_current;
          int *distance_prev;
          int *isChange;


          /*Array on Device*/
          int *device_distance_prev;
          int *device_distance_current;
          int *device_isChange;
          edge_list *device_edgeList;

          double kernel_start_time;
          double kernel_end_time;
          double kernelTimeTaken = 0;

          /* get the number of vertices and edges of the graph */
          uint vertices = peeps->size();
          uint edges = edgeList->size();

          distance_current = (int*)malloc(sizeof(int)*vertices);
          distance_prev = (int*)malloc(sizeof(int)*vertices);
          isChange = (int *)malloc(sizeof(int));

          /* Setting all vertices distances to infinite except the source vertex */
          std::fill_n(distance_current,vertices,SSSP_INF);
          distance_current[0] = 0;
          std::fill_n(distance_prev,vertices,SSSP_INF);
          distance_prev[0] = 0;

          //create a temporary array, because we can't use vector realted stuff on cuda;
          edge_list *temp_array = (edge_list*)malloc(sizeof(edge_list)*edges);
          std::copy(edgeList->begin(),edgeList->end(),temp_array);

          //Allocate the memory on the device;
          cudaMalloc((void **)&device_edgeList,sizeof(edge_list)*edges);
          cudaMalloc((void **)&device_distance_prev, sizeof(int)*vertices );
          cudaMalloc((void **)&device_distance_current, sizeof(int)*vertices);
          cudaMalloc((void **)&device_isChange,sizeof(int));

          //Copy all the vertices to the device;
          cudaMemcpy(device_distance_current,distance_current,sizeof(int)*vertices,cudaMemcpyHostToDevice);
          cudaMemcpy(device_distance_prev,distance_prev,sizeof(int)*vertices,cudaMemcpyHostToDevice);
          cudaMemcpy(device_edgeList,temp_array,sizeof(edge_list)*edges,cudaMemcpyHostToDevice);

          for(uint i = 0 ; i < vertices ; i++) {

              *isChange = 0;
              cudaMemcpy(device_isChange,isChange,sizeof(int),cudaMemcpyHostToDevice);
              kernel_start_time = getTime();
              sharedMemImpl2<<<blockNum,blockSize>>>(device_edgeList,edges,device_distance_prev,device_distance_current,device_isChange);
              cudaDeviceSynchronize();
              kernel_end_time = getTime();
              kernelTimeTaken = kernelTimeTaken + (kernel_end_time- kernel_start_time);
              cudaMemcpy(isChange,device_isChange,sizeof(int),cudaMemcpyDeviceToHost);
              number_of_iterations++;

              if( *isChange == 0 ) {
                  break;
              }
              int *temp = device_distance_current;
              device_distance_current = device_distance_prev;
              device_distance_prev = temp;

          }


          cudaMemcpy(distance_current,device_distance_current,sizeof(int)*vertices,cudaMemcpyDeviceToHost);
          for( int i = 0 ; i < vertices ; i++ ) {
              peeps->at(i).vertexValue.distance = distance_current[i];
          }

          //Free the memory on the device;
          cudaFree(device_edgeList);
          cudaFree(device_distance_current);
          cudaFree(device_distance_prev);
          cudaFree(device_isChange);

          std::cout << "Total time taken :" << getTime() << "ms.\n";
          std::cout << "Time taken by Kernel : " << kernelTimeTaken<< "ms.\n";
          std::cout << "Total number of iterations are: "<<number_of_iterations<<"\n";
          std::cout << "Took " << getTime() << "ms.\n";

      }else{

          return;

      }


        //printf("The address of device edge List is %u\n",device_edgeList);
        //printf("The edgeList size is %u",sizeof(edge_list));
        //printf("The address of next device edge list is %u\n",device_edgeList+1);

       /*for(std::vector<edge_list>::iterator it = edgeList->begin();it!=edgeList->end();++it){

             size_t sz = sizeof(edge_list);
             void *src = &(*it);
             cudaMemcpy(device_edgeList,src,sz,cudaMemcpyHostToDevice);
             device_edgeList += 1;

        }
        device_edgeList = device_edgeList - edges;*/

    return;
}
