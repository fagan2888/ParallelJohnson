/*
 *@description - this file implements the the parallel Johnson's algorithm for
                 single source shortest path on sparse graphs using MPI. It
                 assumes the input graph file is in the format like the graph
                 http://snap.stanford.edu/data/web-Google.html
                 The node numbers are continuous intergers starting from 0.
                 The implementation depends on the Boost library.
 *@author: Yao Zhu (yzhucs@gmail.com).
 */

#include <algorithm>
#include <boost/heap/fibonacci_heap.hpp>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>

using namespace std;
using namespace boost::heap;

// some constants definition.
#define INT_INF 			100000//INT_MAX	// indicate an inf value.

// output a vector in column format to a stream.
void output_vector(ostream &out, const int* const z, const int len) {
	if (z == NULL) {
		cerr << "z cannot be NULL in output_vector()." << endl;
		exit(-1);
	}

	out << "the vector is:" << endl;
	for (int i = 0; i < len; i++) {
		if (z[i] != INT_INF) {
			out << z[i] << endl;
		} else {
			out << "inf" << endl;
		}
	}
}

// let process 0 print out some message for showing the progress.
void print_msg(string msg, int rank) {
	if (rank == 0) {
		cerr << msg << endl;
	}
}

/*
 *@description - read in the webgraph data. (src[m], dest[m]) is a directed edge
 	 	 	 	 in the webgraph. This function also returns the maximum node
 	 	 	 	 number, which is number of nodes - 1 (we assume the node
 	 	 	 	 number starts from 0).
 *@param - webgraph_file, name of the webgraph file.
 *@param[out] - src, the array of source nodes.
 *@param[out] - dest, the array of the destination nodes.
 *@param[out] - nnz, number of edges.
 *@return - the maximum node number.
 */
int read_webgraph(const char *webgraph_file, int **src, int **dest, int *nnz) {
	std::ifstream wbfile(webgraph_file);
	string line;
	*nnz = 0;
	// count the number of edges.
	while (std::getline(wbfile, line)) {
		// check the 1st character of this line.
		if (line[0] >= '0' && line[0] <= '9') {
			(*nnz)++;
		}
	}
	// allocate storage in coordinate format.
	*src = new int[*nnz];
	*dest = new int[*nnz];
	if (*src == NULL || *dest == NULL) {
		cerr << "cannot allocate src or dest in read_webgraph()." << endl;
		exit(-1);
	}
	// return to the beginning of the input file.
	wbfile.clear();
	wbfile.seekg(0, ios::beg);
	int i = 0;
	int max_nn = -1;	// maximum node number.
	while (std::getline(wbfile, line)) {
		// check the 1st character of this line.
		if (line[0] >= '0' && line[0] <= '9') {
			std::istringstream iss(line);
			iss >> (*src)[i] >> (*dest)[i];
			if ((*src)[i] > max_nn) {
				max_nn = (*src)[i];
			}
			if ((*dest)[i] > max_nn) {
				max_nn = (*dest)[i];
			}
			i++;
		}
	}
	wbfile.close();
	return max_nn;
}

/*
 *@description - construct the CSR format of the sparse digraph matrix
 	 	 	 	 (src, dest) from the coordinate format given by (src, dest).
 *@param - (src, dest) defines the web graph edges.
 *@param - nnz, number of nonzero elements. it's the length of
 	 	   src, dest, val, and col_ind.
 *@param - N, number of nodes, it's the length of row_ptr.
 *@param - (val, col_ind, row_ptr) represents the CSR format of the sparse
 	 	   digraph matrix (src, dest).
 */
void coord2csr(const int* const src, const int* const dest,
			   const int nnz, const int N,
			   int *val, int *col_ind, int *row_ptr) {
	if (src == NULL || dest == NULL) {
		cerr << "none of src and dest can be NULL in coord2csr()." << endl;
		exit(-1);
	}
	if (val == NULL || col_ind == NULL || row_ptr == NULL) {
		cerr << "none of val, col_ind, and row_ptr can be NULL in "
				"coord2csr()." << endl;
		exit(-1);
	}
	// we need the out_degree to construct the CSR format.
	int *out_degree = new int[N];
	std::fill(out_degree, out_degree + N, 0);
	for (int l = 0; l < nnz; l++) {
		int i = src[l];
		out_degree[i]++;
	}
	// compute row_ptr as the cumsum of out_degree.
	// note row_ptr[N] = nnz. the node numbers are in [0..N-1].
	row_ptr[0] = 0;
	for (int i = 1; i < N+1; i++) {
		row_ptr[i] = row_ptr[i-1] + out_degree[i-1];
	}
	// construct val and col_ind according to row_ptr.
	for (int l = 0; l < nnz; l++) {
		int i = src[l];
		int j = dest[l];
		col_ind[row_ptr[i]] = j;
		val[row_ptr[i]] = 1;
		row_ptr[i]++;
	}
	// recompute row_ptr as the cumsum of out_degree.
	row_ptr[0] = 0;
	for (int i = 1; i < N+1; i++) {
		row_ptr[i] = row_ptr[i-1] + out_degree[i-1];
	}
	// deallocate out_degree.
	if (out_degree != NULL) {
		delete [] out_degree;
	}
}

/*
 *@description - get the rank of the process that has the node with the given
 	 	 	 	 node number. This partition assumes all the remnant elements
 	 	 	 	 are given to the last process.
 *@param - nn, node number.
 *@param - N, total number of nodes.
 *@param - nproc, number of processes.
 */
int nn2rank(const int nn, const int N, const int nproc) {
	int quota = int(N / nproc);
	if (nn >= quota * nproc) {
		return (nproc - 1);
	} else {
		return int(nn / quota);
	}
}

/*
 *@description - get the number of rows a process p has from the row partition
 				 scheme.
 *@param - N, total number of rows (one row correspond to one node).
 *@param - nproc, total number of processes.
 *@param - rank, of the process.
 */
int get_Nlocal(const int N, const int nproc, const int rank) {
	int quota = int(N / nproc);
	if (rank == nproc - 1) {	// the last process.
		return (N - (nproc - 1) * quota);
	} else {
		return quota;
	}
}

/*
 *@description - get the first node number that belongs to a process p.
 */
int get_start_nn(const int N, const int nproc, const int rank) {
	int quota = int(N / nproc);
	return (quota * rank);
}

/*
 *@description - definition of a data item stored in the priority queue.
 */
struct pq_data
{
    int node_number;		//the global node number.
    int dist;				//current value of distance from the source node.

    pq_data(int nn, int d): node_number(nn), dist(d) {}

    bool operator< (pq_data const & data) const
    {
    	// we define this way because we want our priority queue to be min-heap.
        return dist > data.dist;
    }
};

typedef fibonacci_heap<pq_data>::handle_type handle_t;

/*
 *@description - this function is to encapsulate the operation to the local
 	 	 	 	 priority queue.
 *@param - total_out_num will index into node_buffer and dist_buffer.
 */
void extract_local_pq(int *sp, fibonacci_heap<pq_data> &local_pq,
					  handle_t *handle_local_pq, const int N, const int nproc,
					  const int rank, const int start_nn,
					  const int* const local_row_ptr,
					  const int* const local_col_ind,
					  const int* const local_val, int *sendcnts,
					  int *node_buffer, int *dist_buffer, int &total_out_num) {
	if (!local_pq.empty()) {
		// get the top data item.
		pq_data data_item = local_pq.top();
		if (data_item.dist != INT_INF) {
			// extract it from the queue.
			local_pq.pop();
			int u = data_item.node_number;
			sp[u] = data_item.dist;
			// scan the adjacency list of this node.
			for (int l = local_row_ptr[u-start_nn];
				 l < local_row_ptr[u-start_nn+1]; l++) {
				// v is the neighbor node number (in global value).
				int v = local_col_ind[l];
				int edge_weight = local_val[l];
				// check whether v is a node belong itself or not.
				if (nn2rank(v, N, nproc) == rank) {
					if (sp[v] == INT_INF) { // v still in the queue.
						if ((*handle_local_pq[v-start_nn]).dist >
				 	 	 	sp[u] + edge_weight) {
							// decrease dist value to v.
							pq_data temp_data_item(v, sp[u] + edge_weight);
							local_pq.decrease(handle_local_pq[v-start_nn],
											  temp_data_item);
						}
						} else if (sp[v] > sp[u] + edge_weight) {
							// v is already extracted.
						// insert v back to queue.
						pq_data temp_data_item(v, sp[u] + edge_weight);
						handle_local_pq[v-start_nn] =
								local_pq.push(temp_data_item);
						sp[v] = INT_INF;	// mark it in the queue again.
					}
				} else { // v belong to another process.
					// record v and distance to it.
					node_buffer[total_out_num] = v;
					dist_buffer[total_out_num] = sp[u] + edge_weight;
					total_out_num++;
					// get the process that has v.
					int proc_rank = nn2rank(v, N, nproc);
					// we send the node number and its distance.
					sendcnts[proc_rank] += 2;
				}
			}
		}
	} // if (!local_pq.empty())
}

/*
 *@descriptioni - it assumes the node number starts from 0.
 *@param - argv[1] - name of the file storing web graph.
 *@param - argv[2] - the node number (starts from 0) of the source node.
 *@param - argv[3] - file to store the solution vector.
 */
int parallel_johnson(int argc, char *argv[]) {
	int nproc;							// total number of processes.
	int N;								// total number of nodes.
	int rank;							// rank of the process.

	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (argc != 4) {
		if (rank == 0) {
			cerr << "to run this program must supply the following command "
					"line arguments (in order)" << endl;
			cerr << "argv[1]---web graph file." << endl;
			cerr << "argv[2]---source node number." << endl;
			cerr << "argv[3]---file to save the solution." << endl;
		}
		exit(-1);
	}

	//--------process 0 reads in the web graph, change it to CSR format
	//--------and then partition the nodes and distributes them to each process.

	// data only used by process 0 for reading in the webgraph.
	int *val = NULL;
	int *col_ind = NULL;
	int *row_ptr = NULL;

	if (rank == 0) {
		print_msg("process 0 reads in the web graph data......", rank);
		// process 0 reads in the web graph from file in coordinate format.
		int *src = NULL;
		int *dest = NULL;
		int nnz = 0;
		N = read_webgraph(argv[1], &src, &dest, &nnz) + 1;
		cerr << "N = " << N << endl;
		// construct the CSR format of the sparse matrix P from the
		// coordinate format.
		val = new int[nnz];
		col_ind = new int[nnz];
		row_ptr = new int[N+1];
		coord2csr(src, dest, nnz, N, val, col_ind, row_ptr);

		// we no longer need src and dest.
		if (src != NULL) {
			delete [] src;
			src = NULL;
		}
		if (dest != NULL) {
			delete [] dest;
			dest = NULL;
		}
		nnz = 0;
	}

	print_msg("read in the webgraph is done.", rank);

	/***collective communications for scattering necessary information.***/
	// broadcast the total number of nodes.
	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
	// process 0 prepare the information for scattering P.
	// nnz of the row block local to each process. it's also the sendcounts
	// buffer used to scatter val and col_ind.
	int *local_nnz = NULL;
	// the data sent to process p start at location displs_csr[p] when
	// scattering val and col_ind.
	int *displs_csr = NULL;
	// the sendcounts buffer used to scatter row_ptr.
	int *sendcounts_node = NULL;
	// the data sent to process p start at location displs_node[p] when
	// scattering row_ptr.
	int *displs_node = NULL;
	if (rank == 0) {
		int quota = int(N / nproc);
		local_nnz = new int[nproc];
		displs_csr = new int[nproc];
		sendcounts_node = new int[nproc];
		displs_node = new int[nproc];
		for (int p = 0; p < nproc; p++) {
			int sp = p * quota;
			int tp;
			if (p != nproc - 1) {
				tp = (p + 1) * quota - 1;
			} else {
				tp = N - 1;
			}
			local_nnz[p] = row_ptr[tp+1] - row_ptr[sp];
			displs_csr[p] = row_ptr[sp];
			sendcounts_node[p] = get_Nlocal(N, nproc, p);
			displs_node[p] = sp;
		}
	}
	// (local_val, local_col_ind, local_row_ptr) represents the CSR format
	// of the row block local to this process.
	int *local_val = NULL;
	int *local_col_ind = NULL;
	int *local_row_ptr = NULL;
	// allocate space for local_row_ptr according to the row parition scheme.
	int start_nn = get_start_nn(N, nproc, rank);	// first node belong to the
													// process.
	// this process will possess node numbers [start_nn..start_nn+Nlocal]
	// assuming the contiguous row partition.
	int Nlocal = get_Nlocal(N, nproc, rank);
	local_row_ptr = new int[Nlocal + 1];	// local_row_ptr[Nlocal] stores nnz
											// local to this process from
											// row partition.
	// process 0 scatter local_nnz to each process.
	MPI_Scatter(local_nnz, 1, MPI_INT, local_row_ptr+Nlocal, 1, MPI_INT,
				0, MPI_COMM_WORLD);
	// allocate space for local_val and local_col_ind with the size given by
	// local_row_ptr[Nlocal].
	local_val = new int[local_row_ptr[Nlocal]];
	local_col_ind = new int[local_row_ptr[Nlocal]];
	/***process 0 scatterv the sparse matrix P in CSR format.***/
	// process 0 scatterv val and col_ind to each process.
	MPI_Scatterv(val, local_nnz, displs_csr, MPI_INT, local_val,
				 local_row_ptr[Nlocal], MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatterv(col_ind, local_nnz, displs_csr, MPI_INT, local_col_ind,
				 local_row_ptr[Nlocal], MPI_INT, 0, MPI_COMM_WORLD);
	// process 0 scatterv row_ptr to each process.
	MPI_Scatterv(row_ptr, sendcounts_node, displs_node, MPI_INT,
				 local_row_ptr, Nlocal, MPI_INT, 0, MPI_COMM_WORLD);
	// adjust local_row_ptr to make it start from 0.
	for (int i = Nlocal - 1; i >= 0; i--) {
		local_row_ptr[i] -= local_row_ptr[0];
	}

	print_msg("distribute sparse matrix is done.", rank);

	// process 0 cleans up some space no longer needed.
	if (rank == 0) {
		if (val != NULL) {
			delete [] val;
			val = NULL;
		}
		if (col_ind != NULL) {
			delete [] col_ind;
			col_ind = NULL;
		}
		if (row_ptr != NULL) {
			delete [] row_ptr;
			row_ptr = NULL;
		}
		if (local_nnz != NULL) {
			delete [] local_nnz;
			local_nnz = NULL;
		}
		if (displs_csr != NULL) {
			delete [] displs_csr;
			displs_csr = NULL;
		}
		// sendcounts_node and displs_node will be reused when gathering
		// the solution back.
	}

	// we store the sp vector in full. but each process only needs to
	// operate on the elements it need. sp stores the snapshot shortest path
	// distance from s to each node.
	int *sp = new int[N];
	// initialize sp to be INT_INF.
	std::fill(sp + start_nn, sp + start_nn + Nlocal, INT_INF);

	// get the source node number from command line.
	int source_node = atoi(argv[2]);
	if (rank == 0) {
		cerr << "compute shortest paths from source node: "
			 << source_node << endl;
		cerr << "parallel Johnson's algorithm starts......" << endl;
	}

	// time variables for profiling. we need to include the time for
	// constructing the dependency among processes.
	struct timeval start_tv, end_tv;
	// barrier for time profiling.
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		gettimeofday(&start_tv, NULL);
	}

	//--------parallel Johnson's algorithm through MPI.-----------

	fibonacci_heap<pq_data> local_pq;	// local priority queue.
	//typedef fibonacci_heap<pq_data>::handle_type handle_t;
	// handle to local_pq. it provides the random access to queue elements,
	// as well as for changing the key directly.
	handle_t *handle_local_pq = NULL;
	int local_pq_len = 0;				// length of local_pq.
	int total_pq_len = 0;				// total length of all local_pq across
										// processes.

	// initialize the local priority queue and its handle.
	handle_local_pq = new handle_t[Nlocal];
	for (int i = 0; i < Nlocal; i++) {
		int dist;
		if ((i + start_nn) == source_node) {
			dist = 0;
		} else {
			dist = INT_INF;
		}
		pq_data data_item(start_nn + i, dist);
		handle_local_pq[i] = local_pq.push(data_item);
	}

	// space to communicate possible shortest distances to nodes across
	// processes.
	int senddisp[nproc];
	int sendcnts[nproc];
	int recvdisp[nproc];
	int recvcnts[nproc];
	// we will send and recv node number and its distances simultaneously.
	int *sendbuf = new int[2 * N];
	int *recvbuf = new int[2 * N];

	// buffer to store the node number and distance to them, for communicating
	// to other processes.
	int *node_buffer = new int[N];
	int *dist_buffer = new int[N];
	int total_out_num;	// total number of nodes to send out. it will be
						// the effective length of node_buffer and dist_buffer.
	int total_in_num;	// total number of nodes to recv. the effective length
						// of recvbuf is 2*total_in_num.

	do {
		// initialize space for communication.
		std::fill(sendcnts, sendcnts + nproc, 0);
		std::fill(recvcnts, recvcnts + nproc, 0);
		total_out_num = 0;

		// check the local priority queue.
		const int async_iter = 30;
		for (int l = 0; l < async_iter; l++) {
			extract_local_pq(sp, local_pq, handle_local_pq, N, nproc, rank,
							 start_nn, local_row_ptr, local_col_ind, local_val,
							 sendcnts, node_buffer, dist_buffer, total_out_num);
		}
		// use MPI_Alltoall() to let each process aware of the number of
		// (node, distance) it will receive from each other process.
		MPI_Alltoall(sendcnts, 1, MPI_INT, recvcnts, 1, MPI_INT,
					 MPI_COMM_WORLD);
		// prepare senddisp and recvdisp.
		total_in_num = 0;
		for (int i = 0; i < nproc; i++) {
			total_in_num += recvcnts[i]/2;
			if (i > 0) {
				senddisp[i] = senddisp[i-1] + sendcnts[i-1];
				recvdisp[i] = recvdisp[i-1] + recvcnts[i-1];
			} else {
				senddisp[i] = 0;
				recvdisp[i] = 0;
			}
		}
		// prepared the content of the sendbuf using senddisp.
		for(int j = 0; j < total_out_num; j++) {
			int i = nn2rank(node_buffer[j], N, nproc);
			// put the key and value in sendbuf.
			sendbuf[senddisp[i]++] = node_buffer[j];
			sendbuf[senddisp[i]++] = dist_buffer[j];
		}

		// reconstruct senddisp.
		for (int i = 0; i < nproc; i++) {
			if (i > 0) {
				senddisp[i] = senddisp[i-1] + sendcnts[i-1];
			} else {
				senddisp[i] = 0;
			}
		}
		// use MPI_Alltoallv() to communicate the nodes and distances.
		MPI_Alltoallv(sendbuf, sendcnts, senddisp, MPI_INT, recvbuf,
					  recvcnts, recvdisp, MPI_INT, MPI_COMM_WORLD);

		// scan the recvbuf for distances found by other processes.
		for (int j = 0; j < total_in_num; j++) {
			int v = recvbuf[2*j];
			int v_dist = recvbuf[2*j + 1];
			if (sp[v] == INT_INF) { // v still in the queue.
				if ((*handle_local_pq[v-start_nn]).dist > v_dist) {
					// decrease dist value to v.
					pq_data temp_data_item(v, v_dist);
					local_pq.decrease(handle_local_pq[v-start_nn],
									  temp_data_item);
				}
			} else if (sp[v] > v_dist) { // v is already extracted.
				// insert v back to queue.
				pq_data temp_data_item(v, v_dist);
				handle_local_pq[v-start_nn] = local_pq.push(temp_data_item);
				sp[v] = INT_INF;	// mark it in the queue again.
			}
		}

		// MPI_Allreduce to get the total length of local priority queues
		// to detect termination.
		// we need to check the effective size of local_pq, i.e., #elements
		// with finite values in local_pq. We only need to check the top.
		if (!local_pq.empty()) {
			pq_data check_data_item = local_pq.top();
			if (check_data_item.dist != INT_INF) {
				local_pq_len = 1;
			} else {
				local_pq_len = 0;
			}
		} else {
			local_pq_len = 0;
		}
		MPI_Allreduce(&local_pq_len, &total_pq_len, 1,
					  MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		if (total_pq_len == 0) { // when every process is done.
			break;
		}
	} while (true);

	// barrier for time profiling.
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		gettimeofday(&end_tv, NULL);
		double tElapsed = (end_tv.tv_sec + end_tv.tv_usec/1000000.0) -
						  (start_tv.tv_sec + start_tv.tv_usec/1000000.0);
		cerr << "parallel Johnson's algorithm completes." << endl;
		cout << "Time: " << tElapsed << " seconds when using "
			 << nproc << " processes." << endl;
	}

	// process 0 gather the sp vector and stores it.
	int *sp_sol = NULL;			// the final vector of shortest path distance.
	if (rank == 0) {
		sp_sol = new int[N];
	}
	MPI_Gatherv(sp + start_nn, Nlocal, MPI_INT, sp_sol,
			    sendcounts_node, displs_node, MPI_INT, 0,
			    MPI_COMM_WORLD);
	if (rank == 0) {
		// save the sp_sol vector.
		ofstream out(argv[3], ios::trunc);
		output_vector(out, sp_sol, N);
		cerr << "the shortest path distance vector has been saved in file "
			 << argv[3] << endl;
		// we no longer need sendcounts_node and displs_node.
		if (sendcounts_node != NULL) {
			delete [] sendcounts_node;
			sendcounts_node = NULL;
		}
		if (displs_node != NULL) {
			delete [] displs_node;
			displs_node = NULL;
		}
		// we no longer need sp_sol.
		if (sp_sol != NULL) {
			delete [] sp_sol;
			sp_sol = NULL;
		}
	}

	// clean up all the space.
	if (local_val != NULL) {
		delete [] local_val;
		local_val = NULL;
	}
	if (local_col_ind != NULL) {
		delete [] local_col_ind;
		local_col_ind = NULL;
	}
	if (local_row_ptr != NULL) {
		delete [] local_row_ptr;
		local_row_ptr = NULL;
	}
	if (sp != NULL) {
		delete [] sp;
		sp = NULL;
	}
	if (handle_local_pq != NULL) {
		delete [] handle_local_pq;
		handle_local_pq = NULL;
	}
	if (sendbuf != NULL) {
		delete [] sendbuf;
		sendbuf = NULL;
	}
	if (recvbuf != NULL) {
		delete [] recvbuf;
		recvbuf = NULL;
	}
	if (node_buffer != NULL) {
		delete [] node_buffer;
		node_buffer = NULL;
	}
	if (dist_buffer != NULL) {
		delete [] dist_buffer;
		dist_buffer = NULL;
	}

	return 0;
}

int main(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);

	parallel_johnson(argc, argv);

	MPI_Finalize();
}
