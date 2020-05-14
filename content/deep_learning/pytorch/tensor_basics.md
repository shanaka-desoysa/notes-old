---
title: "PyTorch Tensor Basics"
author: "Shanaka DeSoysa"
date: 2020-05-14T00:00:00-07:00
description: "Converting NumPy arrays to PyTorch tensors and creating tensors from scratch"
type: technical_note
draft: false
---

This section covers:
* Converting NumPy arrays to PyTorch tensors
* Creating tensors from scratch

## Perform standard imports


```python
import torch
import numpy as np
```

Confirm you're using PyTorch version 1.1.0


```python
torch.__version__
```




    '1.5.0+cu101'



## Converting NumPy arrays to PyTorch tensors
A <a href='https://pytorch.org/docs/stable/tensors.html'><strong><tt>torch.Tensor</tt></strong></a> is a multi-dimensional matrix containing elements of a single data type.<br>
Calculations between tensors can only happen if the tensors share the same dtype.<br>
In some cases tensors are used as a replacement for NumPy to use the power of GPUs (more on this later).


```python
arr = np.array([1,2,3,4,5])
print(arr)
print(arr.dtype)
print(type(arr))
```

    [1 2 3 4 5]
    int64
    <class 'numpy.ndarray'>



```python
x = torch.from_numpy(arr)
# Equivalent to x = torch.as_tensor(arr)

print(x)
```

    tensor([1, 2, 3, 4, 5])



```python
# Print the type of data held by the tensor
print(x.dtype)
```

    torch.int64



```python
# Print the tensor object type
print(type(x))
print(x.type()) # this is more specific!
```

    <class 'torch.Tensor'>
    torch.LongTensor



```python
arr2 = np.arange(0.,12.).reshape(4,3)
print(arr2)
```

    [[ 0.  1.  2.]
     [ 3.  4.  5.]
     [ 6.  7.  8.]
     [ 9. 10. 11.]]



```python
x2 = torch.from_numpy(arr2)
print(x2)
print(x2.type())
```

    tensor([[ 0.,  1.,  2.],
            [ 3.,  4.,  5.],
            [ 6.,  7.,  8.],
            [ 9., 10., 11.]], dtype=torch.float64)
    torch.DoubleTensor


Here <tt>torch.DoubleTensor</tt> refers to 64-bit floating point data.

<h2><a href='https://pytorch.org/docs/stable/tensors.html'>Tensor Datatypes</a></h2>
<table style="display: inline-block">
<tr><th>TYPE</th><th>NAME</th><th>EQUIVALENT</th><th>TENSOR TYPE</th></tr>
<tr><td>32-bit integer (signed)</td><td>torch.int32</td><td>torch.int</td><td>IntTensor</td></tr>
<tr><td>64-bit integer (signed)</td><td>torch.int64</td><td>torch.long</td><td>LongTensor</td></tr>
<tr><td>16-bit integer (signed)</td><td>torch.int16</td><td>torch.short</td><td>ShortTensor</td></tr>
<tr><td>32-bit floating point</td><td>torch.float32</td><td>torch.float</td><td>FloatTensor</td></tr>
<tr><td>64-bit floating point</td><td>torch.float64</td><td>torch.double</td><td>DoubleTensor</td></tr>
<tr><td>16-bit floating point</td><td>torch.float16</td><td>torch.half</td><td>HalfTensor</td></tr>
<tr><td>8-bit integer (signed)</td><td>torch.int8</td><td></td><td>CharTensor</td></tr>
<tr><td>8-bit integer (unsigned)</td><td>torch.uint8</td><td></td><td>ByteTensor</td></tr></table>

## Copying vs. sharing

<a href='https://pytorch.org/docs/stable/torch.html#torch.from_numpy'><strong><tt>torch.from_numpy()</tt></strong></a><br>
<a href='https://pytorch.org/docs/stable/torch.html#torch.as_tensor'><strong><tt>torch.as_tensor()</tt></strong></a><br>
<a href='https://pytorch.org/docs/stable/torch.html#torch.tensor'><strong><tt>torch.tensor()</tt></strong></a><br>

There are a number of different functions available for <a href='https://pytorch.org/docs/stable/torch.html#creation-ops'>creating tensors</a>. When using <a href='https://pytorch.org/docs/stable/torch.html#torch.from_numpy'><strong><tt>torch.from_numpy()</tt></strong></a> and <a href='https://pytorch.org/docs/stable/torch.html#torch.as_tensor'><strong><tt>torch.as_tensor()</tt></strong></a>, the PyTorch tensor and the source NumPy array share the same memory. This means that changes to one affect the other. However, the <a href='https://pytorch.org/docs/stable/torch.html#torch.tensor'><strong><tt>torch.tensor()</tt></strong></a> function always makes a copy.


```python
# Using torch.from_numpy(), shares same memory
arr = np.arange(0,5)
t = torch.from_numpy(arr)
print(t)
```

    tensor([0, 1, 2, 3, 4])



```python
arr[2]=77
print(t)
```

    tensor([ 0,  1, 77,  3,  4])



```python
# Using torch.tensor(), makes a copy
arr = np.arange(0,5)
t = torch.tensor(arr)
print(t)
```

    tensor([0, 1, 2, 3, 4])



```python
arr[2]=77
print(t)
```

    tensor([0, 1, 2, 3, 4])


## Class constructors
<a href='https://pytorch.org/docs/stable/tensors.html'><strong><tt>torch.Tensor()</tt></strong></a><br>
<a href='https://pytorch.org/docs/stable/tensors.html'><strong><tt>torch.FloatTensor()</tt></strong></a><br>
<a href='https://pytorch.org/docs/stable/tensors.html'><strong><tt>torch.LongTensor()</tt></strong></a>, etc.<br>

There's a subtle difference between using the factory function <font color=black><tt>torch.tensor(data)</tt></font> and the class constructor <font color=black><tt>torch.Tensor(data)</tt></font>.<br>
The factory function determines the dtype from the incoming data, or from a passed-in dtype argument.<br>
The class constructor <tt>torch.Tensor()</tt>is simply an alias for <tt>torch.FloatTensor(data)</tt>. Consider the following:


```python
data = np.array([1,2,3])
```


```python
a = torch.Tensor(data)  # Equivalent to cc = torch.FloatTensor(data)
print(a, a.type())
```

    tensor([1., 2., 3.]) torch.FloatTensor



```python
b = torch.tensor(data)
print(b, b.type())
```

    tensor([1, 2, 3]) torch.LongTensor



```python
c = torch.tensor(data, dtype=torch.long)
print(c, c.type())
```

    tensor([1, 2, 3]) torch.LongTensor


## Creating tensors from scratch
### Uninitialized tensors with <tt>.empty()</tt>
<a href='https://pytorch.org/docs/stable/torch.html#torch.empty'><strong><tt>torch.empty()</tt></strong></a> returns an <em>uninitialized</em> tensor. Essentially a block of memory is allocated according to the size of the tensor, and any values already sitting in the block are returned. This is similar to the behavior of <tt>numpy.empty()</tt>.


```python
x = torch.empty(4, 3)
print(x)
```

    tensor([[4.4866e-36, 0.0000e+00, 3.3631e-44],
            [0.0000e+00,        nan, 0.0000e+00],
            [1.1578e+27, 1.1362e+30, 7.1547e+22],
            [4.5828e+30, 1.2121e+04, 7.1846e+22]])


### Initialized tensors with <tt>.zeros()</tt> and <tt>.ones()</tt>
<a href='https://pytorch.org/docs/stable/torch.html#torch.zeros'><strong><tt>torch.zeros(size)</tt></strong></a><br>
<a href='https://pytorch.org/docs/stable/torch.html#torch.ones'><strong><tt>torch.ones(size)</tt></strong></a><br>
It's a good idea to pass in the intended dtype.


```python
x = torch.zeros(4, 3, dtype=torch.int64)
print(x)
```

    tensor([[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]])


### Tensors from ranges
<a href='https://pytorch.org/docs/stable/torch.html#torch.arange'><strong><tt>torch.arange(start,end,step)</tt></strong></a><br>
<a href='https://pytorch.org/docs/stable/torch.html#torch.linspace'><strong><tt>torch.linspace(start,end,steps)</tt></strong></a><br>
Note that with <tt>.arange()</tt>, <tt>end</tt> is exclusive, while with <tt>linspace()</tt>, <tt>end</tt> is inclusive.


```python
x = torch.arange(0,18,2).reshape(3,3)
print(x)
```

    tensor([[ 0,  2,  4],
            [ 6,  8, 10],
            [12, 14, 16]])



```python
x = torch.linspace(0,18,12).reshape(3,4)
print(x)
```

    tensor([[ 0.0000,  1.6364,  3.2727,  4.9091],
            [ 6.5455,  8.1818,  9.8182, 11.4545],
            [13.0909, 14.7273, 16.3636, 18.0000]])


### Tensors from data
<tt>torch.tensor()</tt> will choose the dtype based on incoming data:


```python
x = torch.tensor([1, 2, 3, 4])
print(x)
print(x.dtype)
print(x.type())
```

    tensor([1, 2, 3, 4])
    torch.int64
    torch.LongTensor



```python
# Converting type
x = x.type(torch.int16)
print(x.dtype)
print(x.type())
```

    torch.int16
    torch.ShortTensor


Alternatively you can set the type by the tensor method used.
For a list of tensor types visit https://pytorch.org/docs/stable/tensors.html


```python
x = torch.FloatTensor([5,6,7])
print(x)
print(x.dtype)
print(x.type())
```

    tensor([5., 6., 7.])
    torch.float32
    torch.FloatTensor


You can also pass the dtype in as an argument. For a list of dtypes visit https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.dtype<br>


```python
x = torch.tensor([8,9,-3], dtype=torch.int)
print(x)
print(x.dtype)
print(x.type())
```

    tensor([ 8,  9, -3], dtype=torch.int32)
    torch.int32
    torch.IntTensor


### Changing the dtype of existing tensors
Don't be tempted to use <tt>x = torch.tensor(x, dtype=torch.type)</tt> as it will raise an error about improper use of tensor cloning.<br>
Instead, use the tensor <tt>.type()</tt> method.


```python
print('Old:', x.type())

x = x.type(torch.int64)

print('New:', x.type())
```

    Old: torch.IntTensor
    New: torch.LongTensor


### Random number tensors
<a href='https://pytorch.org/docs/stable/torch.html#torch.rand'><strong><tt>torch.rand(size)</tt></strong></a> returns random samples from a uniform distribution over [0, 1)<br>
<a href='https://pytorch.org/docs/stable/torch.html#torch.randn'><strong><tt>torch.randn(size)</tt></strong></a> returns samples from the "standard normal" distribution [σ = 1]<br>
&nbsp;&nbsp;&nbsp;&nbsp;Unlike <tt>rand</tt> which is uniform, values closer to zero are more likely to appear.<br>
<a href='https://pytorch.org/docs/stable/torch.html#torch.randint'><strong><tt>torch.randint(low,high,size)</tt></strong></a> returns random integers from low (inclusive) to high (exclusive)


```python
x = torch.rand(4, 3)
print(x)
```

    tensor([[0.6682, 0.3914, 0.5425],
            [0.2000, 0.3056, 0.9103],
            [0.7039, 0.5021, 0.9170],
            [0.4305, 0.7270, 0.6577]])



```python
x = torch.randn(4, 3)
print(x)
```

    tensor([[ 0.4623, -0.2561, -0.5399],
            [-0.6609, -0.6707,  0.6866],
            [-0.9742, -0.3833,  0.1253],
            [ 0.1251, -0.7600, -1.8088]])



```python
x = torch.randint(0, 5, (4, 3))
print(x)
```

    tensor([[0, 2, 2],
            [3, 1, 2],
            [3, 2, 1],
            [2, 0, 4]])


### Random number tensors that follow the input size
<a href='https://pytorch.org/docs/stable/torch.html#torch.rand_like'><strong><tt>torch.rand_like(input)</tt></strong></a><br>
<a href='https://pytorch.org/docs/stable/torch.html#torch.randn_like'><strong><tt>torch.randn_like(input)</tt></strong></a><br>
<a href='https://pytorch.org/docs/stable/torch.html#torch.randint_like'><strong><tt>torch.randint_like(input,low,high)</tt></strong></a><br> these return random number tensors with the same size as <tt>input</tt>


```python
x = torch.zeros(2,5)
print(x)
```

    tensor([[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]])



```python
x2 = torch.randn_like(x)
print(x2)
```

    tensor([[-0.5288,  0.5442,  1.8976,  0.5154,  2.7177],
            [-1.7115, -1.4005,  0.2681, -0.0782, -0.6214]])


The same syntax can be used with<br>
<a href='https://pytorch.org/docs/stable/torch.html#torch.zeros_like'><strong><tt>torch.zeros_like(input)</tt></strong></a><br>
<a href='https://pytorch.org/docs/stable/torch.html#torch.ones_like'><strong><tt>torch.ones_like(input)</tt></strong></a>


```python
x3 = torch.ones_like(x2)
print(x3)
```

    tensor([[1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.]])


### Setting the random seed
<a href='https://pytorch.org/docs/stable/torch.html#torch.manual_seed'><strong><tt>torch.manual_seed(int)</tt></strong></a> is used to obtain reproducible results


```python
torch.manual_seed(42)
x = torch.rand(2, 3)
print(x)
```

    tensor([[0.8823, 0.9150, 0.3829],
            [0.9593, 0.3904, 0.6009]])



```python
torch.manual_seed(42)
x = torch.rand(2, 3)
print(x)
```

    tensor([[0.8823, 0.9150, 0.3829],
            [0.9593, 0.3904, 0.6009]])


## Tensor attributes
Besides <tt>dtype</tt>, we can look at other <a href='https://pytorch.org/docs/stable/tensor_attributes.html'>tensor attributes</a> like <tt>shape</tt>, <tt>device</tt> and <tt>layout</tt>


```python
x.shape
```




    torch.Size([2, 3])




```python
x.size()  # equivalent to x.shape
```




    torch.Size([2, 3])




```python
x.device
```




    device(type='cpu')



PyTorch supports use of multiple <a href='https://pytorch.org/docs/stable/tensor_attributes.html#torch-device'>devices</a>, harnessing the power of one or more GPUs in addition to the CPU. We won't explore that here, but you should know that operations between tensors can only happen for tensors installed on the same device.


```python
x.layout
```




    torch.strided



PyTorch has a class to hold the <a href='https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.layout'>memory layout</a> option. The default setting of <a href='https://en.wikipedia.org/wiki/Stride_of_an_array'>strided</a> will suit our purposes throughout the course.
