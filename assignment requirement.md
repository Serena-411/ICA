First, one needs generate m signal samples which follow Laplace(0,1) distribution, and generate another m signal samples whose probability density function is p(x)=g’(x), where g(x)=1/(1+e^(-x)) is the normal Sigmoid function. Denote these m combined 2-dimensional data as S. Then, pick some 2-by-2 mixing matrix whose rows are unit vectors, say A. Now we have m recording samples: X=AS (or S’s transpose).

This homework will use ICA algorithm to find the mixing matrix A. In the updating step, you may want to use '-2*tanh(w_i x_k)' to replace '1-2g(w_i x_k)' .

After the training finishes, you need print out : **all the model parameters, ** **and the best approximations to A.**

 

0. you MUST use seed() before applying any random operations;

1. data pre-processing is recommended;

2. 300 <= m <= 2000;

3. you may use 1-sample mini-batch (SGA), or n-samples mini-batch (2<= n <= 15) to update the un-mixing matrix W; **try all parameters to find suitable ones** so your program will finish in about 15 seconds;

4. find a way to test when the training should stop, and how to measure one approximation of A is better than another; and explain this in your code comments;

output example:

![img](file:///C:/Users/huawei/AppData/Local/Temp/msohtmlclip1/01/clip_image002.png)

 