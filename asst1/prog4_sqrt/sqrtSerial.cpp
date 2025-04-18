#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

void sqrtSerial(int N,
                float initialGuess,
                float values[],
                float output[])
{

    static const float kThreshold = 0.00001f;

    for (int i=0; i<N; i++) {

        float x = values[i];
        float guess = initialGuess;

        float error = fabs(guess * guess * x - 1.f);

        while (error > kThreshold) {
            guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            error = fabs(guess * guess * x - 1.f);
        }

        output[i] = x * guess;
    }
}

void sqrtVector(int N, float initialGuess, float values[], float output[]) {

    static const float kThreshold = 0.00001f;
    // broadcast kThreshold to 256-wide vector
    __m256 vec_kThreshold = _m256_set1_ps(kThreshold);

    // broadcast initialGuess to 256-wide vector
    __m256 vec_guess = _mm256_set1_ps(initialGuess);
    __m256 ones = _mm256_set1_ps(1.f);
    __m256 threes = _mm256_set1_ps(3.f);
    __m256 halves = _mm256_set_ps(0.5f);
    // 01111...
    __m256 absMask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

    __m256 vec_values;
    __m256 vec_error;
    __m256 vec_output;

    for (int i=0; i<N; i+=8) {

      // load 8 consecutive floating points
      vec_values = _mm256_loadu_ps(&values[i]); 

      // guess*guess*x -1.f
      error = _mm256_sub_ps(
          _mm256_mul_ps(_mm256_mul_ps(vec_guess, vec_guess), vec_values),
          ones
      );
      // fabs()
      // _mm256_set1_ps(-0.0f);
      // The order of 2 arguments matter here.
      // not in a (-0.0f) and then and with b (error)
      error = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), error);

      // FIXME: this does not work.
      __mmask8 gtMask = _mm256_cmp_ps_mask(error, vec_kThreshold, _CMP_GT_OQ); 

      // need to count number of lanes that are still active
      while(_mm_popcnt_u32(_mm_cvtmask8_u32(gtMask)) > 0) {
        // I also should probably only update guess whose lanes are active
        // guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
        __mm256 new_vec_guess = _mm256_mul_ps(
            halves,
            __mm256_sub_ps(
                __mm256_mul_ps(threes, vec_guess),
                __mm256_mul_ps(
                  vec_values,
                  __mm256_mul_ps(vec_guess, __mm256_mul_ps(vec_guess, vec_guess))
                )
            )
        )
        // now need to update vec_guess by new_vec_guess for active lanes
        // vec_guess = ...
        
        error = _mm256_sub_ps(
            _mm256_mul_ps(_mm256_mul_ps(vec_guess, vec_guess), vec_values),
            ones
        );
        error = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), error);

        // now need to recompute the gtMask with newly calculated error
        gtMask = _m256_cmp_ps_mask(error, vec_kThreshold, _CMP_GT_OQ); 
      }

      // output[i] = x * guess;
      vec_output = _mm256_mul_ps(vec_values, vec_guess);
      // store to output
      _mm256_storeu_ps(&output[i], vec_output);
    }
}
