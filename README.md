# Fully connected Neural Network CUDA C Library.
MEng Thesis implemented in 2016-2017 in pure C/CUDA 8.
Looking backwards, I see a lot of things I didnt know back then.
In this quick readme I will attempt to distill the usefull lesson from this project.

## Key lessons and intuitions

### API
My approach in the API structure. It is not the best API but from an memory size it has no memory leaks. All the memory is dynamically allocated and freed in the end. It supports only batch size of 1.
![Alt text](./figures/f1.png?raw=true "title")

### Implementation 1
The basic parallel implementation of a fully connected layer. Each 'neuron' is an CUDA thread. Each thread is completely independent. 
![Alt text](./figures/f2.png?raw=true "title")

### Implementation 2
The first optimized version of a 'neuron'. Each neuron is a block of threads and each thread performs a single element multiplication that is then written in the shared memory. This technique is called reduction in parallel programming.
![Alt text](./figures/f3.png?raw=true "title")

### Implementation 3
The second optimized version of a 'neuron'. Each neuron is a block of threads and each thread performs 2 multiplications that its sum is then written in the shared memory.
![Alt text](./figures/f4.png?raw=true "title")

###  Qualitative Results
![Alt text](./figures/f6.png?raw=true "title")
###  Quantitative Results
![Alt text](./figures/f5.png?raw=true "title")

### Reduction Speedup
This should have been a scatter plot, because we only care about the powers of 2, but nothing is perfect :)
The expiriments were conducted in a NVIDIA GeForce 920M in my laptop.
![Alt text](./figures/f7.png?raw=true "title")
