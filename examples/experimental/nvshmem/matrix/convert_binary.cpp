#include <bcl/containers/sequential/CSRMatrix.hpp>
#include <chrono>

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("usage: ./convert_binary [input.mtx] [output.binary]\n");
    exit(1);
  }

  std::string input_fname = argv[1];
  std::string output_fname = argv[2];

  using T = float;
  using I = int;

  printf("Reading Matrix Market file. This may take a while...\n");
  auto begin = std::chrono::high_resolution_clock::now();
  BCL::CSRMatrix<T, I> mat(input_fname, BCL::FileFormat::MatrixMarket);
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();
  printf("Reading Matrix Market file in %ss.\n", duration);

  printf("Writing binary file. This should be somewhat faster...\n");
  begin = std::chrono::high_resolution_clock::now();
  mat.write_Binary(output_fname);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - begin).count();
  printf("Wrote binary file in %lfs.\n", duration);

  return 0;
}
