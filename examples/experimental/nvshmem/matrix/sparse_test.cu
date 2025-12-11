
#define __thrust_compiler_fence() __sync_synchronize()
#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include <cusp/array2d.h>
#include <cusp/multiply.h>
#include <cusp/array2d.h>
#include <cusp/print.h>

#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>
#include <bcl/containers/experimental/cuda/CudaMatrix.hpp>
#include <bcl/containers/experimental/cuda/launch_kernel.cuh>
#include <thrust/sort.h>

#include <bcl/containers/experimental/cuda/CudaSPMatrix.hpp>

#include <unordered_map>

#include <bcl/containers/experimental/cuda/sequential/cusparse_util.cuh>
#include <bcl/containers/experimental/cuda/algorithms/spgemm.hpp>

#include <chrono>

template <typename T, typename U>
struct PairHash {
  std::size_t operator()(const std::pair<T, U>& value) const noexcept {
    return std::hash<T>{}(value.first) ^ std::hash<U>{}(value.second);
  }
};

int main(int argc, char** argv) {
  BCL::init(16);
  BCL::cuda::init();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("Hello from rank %d\n");

  using T = float;
  using index_type = int;

  bool verify_result = true;

  std::string fname = std::string(argv[1]);

  auto matrix_shape = BCL::matrix_io::matrix_info(fname);
  size_t m = matrix_shape.shape[0];
  size_t n = matrix_shape.shape[1];
  assert(m == n);
  size_t k = m;

  BCL::print("Choosing blocks...\n");
  auto blocks = BCL::block_matmul(m, n, k);

  BCL::print("Reading matrices...\n");
  BCL::cuda::SPMatrix<T, index_type> a(fname, std::move(blocks[0]));
  BCL::cuda::SPMatrix<T, index_type> b(fname, std::move(blocks[1]));


  BCL::print("Info:\n");
  if (BCL::rank() == 0) {
    printf("A:\n");
    a.print_info();
    printf("B:\n");
    b.print_info();
  }

  cusparseStatus_t status = cusparseCreate(&BCL::cuda::bcl_cusparse_handle_);
  BCL::cuda::throw_cusparse(status);

  assert(a.grid_shape()[1] == b.grid_shape()[0]);

  using allocator_type = BCL::cuda::bcl_allocator<T>;
  int ntrials = 6;
  BCL::barrier();

  for (int i=0; i<ntrials; i++)
  {
      BCL::cuda::duration_issue = 0;
      BCL::cuda::duration_sync = 0;
      BCL::cuda::duration_compute = 0;
      BCL::cuda::duration_accumulate = 0;
      BCL::cuda::duration_barrier = 0;

      BCL::print("Beginning SpGEMM...\n");

      BCL::cuda::SPMatrix<T, index_type> c(m, n, std::move(blocks[2]));

      BCL::barrier();
      auto begin = std::chrono::high_resolution_clock::now();
      BCL::cuda::gemm<T, index_type, allocator_type>(a, b, c);
      BCL::barrier();
      auto end = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double>(end - begin).count();

      double max_issue = BCL::allreduce(BCL::cuda::duration_issue, BCL::max<double>{});
      double max_sync = BCL::allreduce(BCL::cuda::duration_sync, BCL::max<double>{});
      double max_compute = BCL::allreduce(BCL::cuda::duration_compute, BCL::max<double>{});
      double max_accumulate = BCL::allreduce(BCL::cuda::duration_accumulate, BCL::max<double>{});
      double max_barrier = BCL::allreduce(BCL::cuda::duration_barrier, BCL::max<double>{});

      double min_issue = BCL::allreduce(BCL::cuda::duration_issue, BCL::min<double>{});
      double min_sync = BCL::allreduce(BCL::cuda::duration_sync, BCL::min<double>{});
      double min_compute = BCL::allreduce(BCL::cuda::duration_compute, BCL::min<double>{});
      double min_accumulate = BCL::allreduce(BCL::cuda::duration_accumulate, BCL::min<double>{});
      double min_barrier = BCL::allreduce(BCL::cuda::duration_barrier, BCL::min<double>{});

      BCL::cuda::duration_issue = BCL::allreduce(BCL::cuda::duration_issue, std::plus<double>{});
      BCL::cuda::duration_sync = BCL::allreduce(BCL::cuda::duration_sync, std::plus<double>{});
      BCL::cuda::duration_compute = BCL::allreduce(BCL::cuda::duration_compute, std::plus<double>{});
      BCL::cuda::duration_accumulate = BCL::allreduce(BCL::cuda::duration_accumulate, std::plus<double>{});
      BCL::cuda::duration_barrier = BCL::allreduce(BCL::cuda::duration_barrier, std::plus<double>{});

      BCL::barrier();
      fflush(stdout);
      BCL::barrier();
      fprintf(stderr, "RANK(%lu) A has %lu nnz, B has %lu nnz, C has %lu nnz\n",
              BCL::rank(), a.my_nnzs(), b.my_nnzs(), c.my_nnzs());
      BCL::barrier();
      fflush(stderr);
      BCL::barrier();

      if (BCL::rank() == 0) {
        printf("duration_issue %lf (%lf -> %lf)\n",
               BCL::cuda::duration_issue / BCL::nprocs(),
               min_issue, max_issue);
        printf("duration_sync %lf (%lf -> %lf)\n",
               BCL::cuda::duration_sync / BCL::nprocs(),
               min_sync, max_sync);
        printf("duration_compute %lf (%lf -> %lf)\n",
               BCL::cuda::duration_compute / BCL::nprocs(),
               min_compute, max_compute);
        printf("duration_accumulate %lf (%lf -> %lf)\n",
               BCL::cuda::duration_accumulate / BCL::nprocs(),
               min_accumulate, max_accumulate);
        printf("duration_barrier %lf (%lf -> %lf)\n",
               BCL::cuda::duration_barrier / BCL::nprocs(),
               min_barrier, max_barrier);
      }

      BCL::barrier();
      fflush(stdout);
      BCL::barrier();

      BCL::print("Matrix multiply finished in %lf s\n", duration);
  }

  BCL::finalize();
  return 0;
}
