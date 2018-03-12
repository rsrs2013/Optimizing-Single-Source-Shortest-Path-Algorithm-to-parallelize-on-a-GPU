#include <cstring>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

#include "opt.cu"
#include "impl2.cu"
#include "impl1.cu"

#define SSSP_INF 1073741824

enum class ProcessingType {Push, Neighbor, Own, Unknown};
enum SyncMode {InCore, OutOfCore};
enum SyncMode syncMethod;
enum SmemMode {UseSmem, UseNoSmem};
enum SmemMode smemMethod;
enum EdgeListMode {input=0,source=1,destination=2};
enum EdgeListMode edge_list_mode;

// Open files safely.
template <typename T_file>
void openFileToAccess( T_file& input_file, std::string file_name ) {
	input_file.open( file_name.c_str() );
	if( !input_file )
		throw std::runtime_error( "Failed to open specified file: " + file_name + "\n" );
}


int source_order_compare( edge_list a , edge_list b ) {
		return a.srcIndex < b.srcIndex;
}

int destination_order_compare( edge_list a , edge_list b ) {
	return a.destIndex < b.destIndex;
}

// Execution entry point.
int main( int argc, char** argv )
{

	std::string usage =
		"\tRequired command line arguments:\n\
			Input file: E.g., --input in.txt\n\
                        Block size: E.g., --bsize 512\n\
                        Block count: E.g., --bcount 192\n\
                        Output path: E.g., --output output.txt\n\
			Processing method: E.g., --method bmf (bellman-ford), or tpe (to-process-edge), or opt (one further optimizations)\n\
			Shared memory usage: E.g., --usesmem yes, or no \n\
			Sync method: E.g., --sync incore, or outcore\n\
			Edge List Order Mode : E.g., --edgelist input,source,destination\n";

	try {

		std::ifstream inputFile;
		std::ofstream outputFile;
		int selectedDevice = 0;
		int bsize = 0, bcount = 0;
		int vwsize = 32;
		int threads = 1;
		long long arbparam = 0;
		bool nonDirectedGraph = false;		// By default, the graph is directed.
		ProcessingType processingMethod = ProcessingType::Unknown;
		syncMethod = OutOfCore;


		/********************************
		 * GETTING INPUT PARAMETERS.
		 ********************************/

		for( int iii = 1; iii < argc; ++iii )
			if ( !strcmp(argv[iii], "--method") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "bmf") )
				        processingMethod = ProcessingType::Push;
				else if ( !strcmp(argv[iii+1], "tpe") )
    				        processingMethod = ProcessingType::Neighbor;
				else if ( !strcmp(argv[iii+1], "opt") )
				    processingMethod = ProcessingType::Own;
				else{
           std::cerr << "\n Un-recognized method parameter value \n\n";
           exit;
         }
			}
			else if ( !strcmp(argv[iii], "--sync") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "incore") )
				        syncMethod = InCore;
				else if ( !strcmp(argv[iii+1], "outcore") )
    				        syncMethod = OutOfCore;
				else{
           std::cerr << "\n Un-recognized sync parameter value \n\n";
           exit;
         }

			}
			else if ( !strcmp(argv[iii], "--usesmem") && iii != argc-1 ) {
				printf("The value of the argv[iii+1] is %s\n",argv[iii+1]);
				if ( !strcmp(argv[iii+1], "yes") )
				        smemMethod = UseSmem;
				else if (!strcmp(argv[iii+1], "no") )
    				        smemMethod = UseNoSmem;
        else{
           std::cerr << "\n Un-recognized usesmem parameter value \n\n";
           exit;
         }
			}
			else if ( !strcmp(argv[iii], "--edgelist") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "input") )
				        edge_list_mode = input;
				else if ( !strcmp(argv[iii+1], "source") )
    				        edge_list_mode = source;
				else if ( !strcmp(argv[iii+1], "destination") )
						    edge_list_mode = destination;
        else{
           std::cerr << "\n Un-recognized usesmem parameter value \n\n";
           exit;
         }
			}
			else if( !strcmp( argv[iii], "--input" ) && iii != argc-1 /*is not the last one*/)
				openFileToAccess< std::ifstream >( inputFile, std::string( argv[iii+1] ) );
			else if( !strcmp( argv[iii], "--output" ) && iii != argc-1 /*is not the last one*/)
				openFileToAccess< std::ofstream >( outputFile, std::string( argv[iii+1] ) );
			else if( !strcmp( argv[iii], "--bsize" ) && iii != argc-1 /*is not the last one*/)
				bsize = std::atoi( argv[iii+1] );
			else if( !strcmp( argv[iii], "--bcount" ) && iii != argc-1 /*is not the last one*/)
				bcount = std::atoi( argv[iii+1] );

		if(bsize <= 0 || bcount <= 0){
			std::cerr << "Usage: " << usage;
      exit;
			throw std::runtime_error("\nAn initialization error happened.\nExiting.");
		}
		if( !inputFile.is_open() || processingMethod == ProcessingType::Unknown ) {
			std::cerr << "Usage: " << usage;
			throw std::runtime_error( "\nAn initialization error happened.\nExiting." );
		}
		if( !outputFile.is_open() )
			openFileToAccess< std::ofstream >( outputFile, "out.txt" );
		CUDAErrorCheck( cudaSetDevice( selectedDevice ) );
		std::cout << "Device with ID " << selectedDevice << " is selected to process the graph.\n";
		std::cout << "Executing entry_point.cu \n";


		/********************************
		 * Read the input graph file.
		 ********************************/

		std::cout << "Collecting the input graph ...\n";
		std::vector<initial_vertex> parsedGraph( 0 );
		std::vector<edge_list> edgeList( 0 );

		uint nEdges = parse_graph::parse(inputFile,parsedGraph,edgeList,arbparam,nonDirectedGraph);
		std::cout << "Input graph collected with " << parsedGraph.size() << " vertices and " << nEdges << " edges.\n";



         /********************************
		 * Sort the edge list.
		 ********************************/

          switch(edge_list_mode) {
		    case 0:
			    break;
		    case 1:
			    std::sort(edgeList.begin(),edgeList.end(),source_order_compare);
			    break;
		    case 2:
			    std::sort(edgeList.begin(),edgeList.end(),destination_order_compare);
			    break;
	    }


      /*  for( int i = 0 ; i < std::min(nEdges,(uint)20) ; i++ ) {
		    std::cout << "Edge i : " << i << " ( " << edgeList[i].srcIndex << " , " << edgeList[i].destIndex << ") , weight : " << edgeList[i].weight << " \n";
	    }*/


        std::cout<<"Arranging for the Bellman-Ford\n";


		uint vertices = parsedGraph.size();
		uint edges = nEdges;
		uint distance[vertices];

		setTime();
		std::cout<<"The number of vertices are:"<< vertices<<std::endl;
		std::fill_n(distance,vertices,SSSP_INF);
		int start_vertex = 0;
		distance[start_vertex] = 0;


        std::cout<<"Starting the bellman-ford\n";

		for(uint i = 0 ; i < vertices ; i++ ) {
				bool change = false;
				for(uint j = 0 ; j < edges ; j++ ){
						int source = edgeList[j].srcIndex;
						int destination = edgeList[j].destIndex;
						int weight = edgeList[j].weight;
						if(distance[source] + weight < distance[destination]) {
								distance[destination] = distance[source] + weight;
								change = true;
						}
				}
				if( !change )
						break;
		}
		std::cout << "Sequential Bellman-Ford Takes " << getTime() << "ms.\n";



		 /* Process the graph.
		 ********************************/


		switch(processingMethod){
		case ProcessingType::Push:
		    puller(&parsedGraph, bsize, bcount, &edgeList,syncMethod,smemMethod,edge_list_mode);
		    break;
		case ProcessingType::Neighbor:
		    neighborHandler(&parsedGraph, bsize, bcount, &edgeList,syncMethod,smemMethod,edge_list_mode);
				//(std::vector<initial_vertex> * peeps, int blockSize, int blockNum,std::vector<edge_list> *edgeList,int syncMethod, int smemMethod,int edge_list_mode);
		    break;
		default:
		    own(&parsedGraph, bsize, bcount);
		}

		/********************************
		 * Do the comparision b/w parallel and sequential
		 ********************************/

		 int num_of_diff = 0;
		for(int i = 0 ; i < vertices ; i++ ) {
			if( parsedGraph[i].vertexValue.distance != distance[i]) {
				//std::cout << "Sequential Distance for v " << i << " is : " << distance[i] << " and parallel gave : " << parsedGraph[i].vertexValue.distance << " \n.";
				num_of_diff++;
			}
		}

		if( num_of_diff == 0 ) {
			std::cout << "Good Job! Serial and Parallel versions match.\n";
		}
		else {
			std::cout << "Warning!!! Serial and Parallel mismatch " << num_of_diff << " distance values.\n";
		}


		/********************************
		 * It's done here.
		 ********************************/

		CUDAErrorCheck( cudaDeviceReset() );
		std::cout << "Done.\n";
		return( EXIT_SUCCESS );

	}
	catch( const std::exception& strException ) {
		std::cerr << strException.what() << "\n";
		return( EXIT_FAILURE );
	}
	catch(...) {
		std::cerr << "An exception has occurred." << std::endl;
		return( EXIT_FAILURE );
	}

}
