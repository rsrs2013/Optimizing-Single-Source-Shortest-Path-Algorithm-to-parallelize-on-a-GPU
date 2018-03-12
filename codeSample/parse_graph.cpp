#include <string>
#include <cstring>
#include <cstdlib>
#include <stdio.h>
#include <iostream>

#include "parse_graph.hpp"

#define SSSP_INF 1073741824



uint parse_graph::parse(
		std::ifstream& inFile,
		std::vector<initial_vertex>& initGraph,
		std::vector<edge_list>& edgeList,
		const long long arbparam,
		const bool nondirected) {

	const bool firstColumnSourceIndex = true;

	std::string line;
	char delim[3] = " \t";	//In most benchmarks, the delimiter is usually the space character or the tab character.
	char* pch;
	uint nEdges = 0;

	unsigned int Additionalargc=0;
	char* Additionalargv[ 61 ];

	// Read the input graph line-by-line.
	while( std::getline( inFile, line ) ) {
		if( line[0] < '0' || line[0] > '9' )	// Skipping any line blank or starting with a character rather than a number.
			continue;
		char cstrLine[256];
		std::strcpy( cstrLine, line.c_str() );
		uint firstIndex, secondIndex;

		pch = strtok(cstrLine, delim);
		if( pch != NULL )
			firstIndex = atoi( pch );
		else
			continue;
		pch = strtok( NULL, delim );
		if( pch != NULL )
			secondIndex = atoi( pch );
		else
			continue;

		uint theMax = std::max( firstIndex, secondIndex );
		uint srcVertexIndex = firstColumnSourceIndex ? firstIndex : secondIndex;
		uint dstVertexIndex = firstColumnSourceIndex ? secondIndex : firstIndex;

        edge_list edge;
		edge.srcIndex = srcVertexIndex;
		edge.destIndex = dstVertexIndex;




		if( initGraph.size() <= theMax )
			initGraph.resize(theMax+1);
		{ //This is just a block
		        // Add the neighbor. A neighbor wraps edges
			neighbor nbrToAdd;
			nbrToAdd.srcIndex = srcVertexIndex;

			Additionalargc=0;
			Additionalargv[ Additionalargc ] = strtok( NULL, delim );
			while( Additionalargv[ Additionalargc ] != NULL ){
				Additionalargc++;
				Additionalargv[ Additionalargc ] = strtok( NULL, delim );
			}

			//Initializing the source vertex distance;
			initGraph.at(srcVertexIndex).vertexValue.distance = ( srcVertexIndex != arbparam ) ? SSSP_INF : 0;

			//Initializing the destination vertex distance
			initGraph.at(dstVertexIndex).vertexValue.distance = ( dstVertexIndex != arbparam ) ? SSSP_INF : 0;

			//Adding the edge weight of the edge
			nbrToAdd.edgeValue.weight = ( Additionalargc > 0 ) ? atoi(Additionalargv[0]) : 1;

            //Adding to the neighbors of the destination vertex;
			initGraph.at(dstVertexIndex).nbrs.push_back( nbrToAdd );

            edge.weight = ( Additionalargc > 0 ) ? atoi(Additionalargv[0]) : 1;
			edgeList.push_back( edge );


			nEdges++;
		}
		if( nondirected ) {
		        // Add the edge going the other way
			uint tmp = srcVertexIndex;
			srcVertexIndex = dstVertexIndex;
			dstVertexIndex = tmp;
			//swap src and dest and add as before

			edge_list undirected_edge;
			undirected_edge.srcIndex = srcVertexIndex;
			undirected_edge.destIndex = dstVertexIndex;

			neighbor nbrToAdd;
			nbrToAdd.srcIndex = srcVertexIndex;

			Additionalargc=0;
			Additionalargv[ Additionalargc ] = strtok( NULL, delim );
			while( Additionalargv[ Additionalargc ] != NULL ){
				Additionalargc++;
				Additionalargv[ Additionalargc ] = strtok( NULL, delim );
			}

			initGraph.at(srcVertexIndex).vertexValue.distance = ( srcVertexIndex != arbparam ) ? SSSP_INF : 0;
			initGraph.at(dstVertexIndex).vertexValue.distance = ( dstVertexIndex != arbparam ) ? SSSP_INF : 0;
			nbrToAdd.edgeValue.weight = ( Additionalargc > 0 ) ? atoi(Additionalargv[0]) : 1;
			initGraph.at(dstVertexIndex).nbrs.push_back( nbrToAdd );

            undirected_edge.weight = ( Additionalargc > 0 ) ? atoi(Additionalargv[0]) : 1;
			edgeList.push_back(undirected_edge);



			nEdges++;
		}
	}

	return nEdges;

}
