## Program 4: Iterative `sqrt` (15 points) ##

Program 4 is an ISPC program that computes the square root of 20 million
random numbers between 0 and 3. It uses a fast, iterative implementation of
square root that uses Newton's method to solve the equation ${\frac{1}{x^2}} - S = 0$.
The value 1.0 is used as the initial guess in this implementation. The graph below shows the 
number of iterations required for `sqrt` to converge to an accurate solution 
for values in the (0-3) range. (The implementation does not converge for 
inputs outside this range). Notice that the speed of convergence depends on the 
accuracy of the initial guess.

Note: This problem is a review to double-check your understanding, as it covers similar concepts as programs 2 and 3.

![Convergence of sqrt](../handout-images/sqrt_graph.jpg "Convergence of sqrt on the range 0-3 with starting guess 1.0. Note that iterations until convergence is immediate for an input value of 1 and increases as the input value goes toward 0 or 3 (highest value is for input of 3).")


## Q4-1
Build and run `sqrt`. Report the ISPC implementation speedup for 
single CPU core (no tasks) and when using all cores (with tasks). What 
is the speedup due to SIMD parallelization? What is the speedup due to 
multi-core parallelization?

## Q4-2
Modify the contents of the array values to improve the relative speedup 
of the ISPC implementations. Construct a specifc input that __maximizes speedup over the sequential version of the code__ and report the resulting speedup achieved (for both the with- and without-tasks ISPC implementations). Does your modification improve SIMD speedup?
Does it improve multi-core speedup (i.e., the benefit of moving from ISPC without-tasks to ISPC with tasks)? Please explain why.

## Q4-3
Construct a specific input for `sqrt` that __minimizes speedup for ISPC
(without-tasks) over the sequential version of the code__. Describe this input,
describe why you chose it, and report the resulting relative performance of
the ISPC implementations. What is the reason for the loss in efficiency? 
__(keep in mind we are using the `--target=avx2` option for ISPC,
which generates 8-wide SIMD instructions)__. 

## Q4-4
_Extra Credit: (up to 2 points)_ Write your own version of the `sqrt` 
function manually using AVX2 intrinsics. To get credit your 
implementation should be nearly as fast (or faster) than the binary 
produced using ISPC. You may find the [Intel Intrinsics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/) 
very helpful.

