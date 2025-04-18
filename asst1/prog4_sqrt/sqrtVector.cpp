#include <immintrin.h>


void sqrtVector(int N, float initialGuess, float values[], float output[]) {

    static const float kThreshold = 0.00001f;
    // broadcast kThreshold to 256-wide vector
    __m256 vec_kThreshold = _mm256_set1_ps(kThreshold);

    // some constant vectors
    __m256 ones = _mm256_set1_ps(1.f);
    __m256 threes = _mm256_set1_ps(3.f);
    __m256 halves = _mm256_set1_ps(0.5f);

    __m256 vec_values;
    __m256 vec_error;
    __m256 vec_output;

    for (int i=0; i<N; i+=8) {

      // load 8 consecutive floating points
      vec_values = _mm256_loadu_ps(&values[i]); 
      // broadcast initialGuess to 256-wide vector
      // remember to reset initial guess for each iteration
      __m256 vec_guess = _mm256_set1_ps(initialGuess);

      // guess*guess*x -1.f
      vec_error = _mm256_sub_ps(
          _mm256_mul_ps(_mm256_mul_ps(vec_guess, vec_guess), vec_values),
          ones
      );
      // fabs()
      // _mm256_set1_ps(-0.0f);
      // The order of 2 arguments matter here.
      // not in a (-0.0f) and then and with b (error)
      vec_error = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vec_error);

      // get mask in 256-bit vector
      // operator is ordered greater-than
      __m256 gtMask = _mm256_cmp_ps(vec_error, vec_kThreshold, _CMP_GT_OQ); 

      // need to count number of lanes that are still active
      while(_mm256_movemask_ps(gtMask) != 0) {
        // I also should probably only update guess whose lanes are active
        // guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
        __m256 new_vec_guess = _mm256_mul_ps(
            halves,
            _mm256_sub_ps(
                _mm256_mul_ps(threes, vec_guess),
                _mm256_mul_ps(
                  vec_values,
                  _mm256_mul_ps(vec_guess, _mm256_mul_ps(vec_guess, vec_guess))
                )
            )
        );
        // now need to update vec_guess by new_vec_guess for active lanes
        // active means those lanes have not converged yet.
        vec_guess = _mm256_blendv_ps(vec_guess, new_vec_guess, gtMask);
        
        // Re-compute error
        vec_error = _mm256_sub_ps(
            _mm256_mul_ps(_mm256_mul_ps(vec_guess, vec_guess), vec_values),
            ones
        );
        // fabs
        vec_error = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vec_error);

        // now need to recompute the gtMask with newly calculated error
        gtMask = _mm256_cmp_ps(vec_error, vec_kThreshold, _CMP_GT_OQ); 
      }

      // output[i] = x * guess;
      vec_output = _mm256_mul_ps(vec_values, vec_guess);
      // store to output
      _mm256_storeu_ps(&output[i], vec_output);
    }
}

