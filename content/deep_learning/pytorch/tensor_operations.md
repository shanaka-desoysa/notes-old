---
title: "PyTorch Tensor Operations"
author: "Shanaka DeSoysa"
date: 2020-05-14T00:00:00-07:00
description: "PyTorch Tensor Operations."
type: technical_note
draft: false
---

This section covers:
* Indexing and slicing
* Reshaping tensors (tensor views)
* Tensor arithmetic and basic operations
* Dot products
* Matrix multiplication
* Additional, more advanced operations

## Perform standard imports


```python
import torch
import numpy as np
```

## Indexing and slicing
Extracting specific values from a tensor works just the same as with NumPy arrays<br>
<img src='https://github.com/shanaka-desoysa/pytorch-deep-learning/blob/master/Images/arrayslicing.png?raw=1' width="500" style="display: inline-block"><br><br>
Image source: http://www.scipy-lectures.org/_images/numpy_indexing.png


```python
x = torch.arange(6).reshape(3,2)
print(x)
```

    tensor([[0, 1],
            [2, 3],
            [4, 5]])



```python
# Grabbing the right hand column values
x[:,1]
```




    tensor([1, 3, 5])




```python
# Grabbing the right hand column as a (3,1) slice
x[:,1:]
```




    tensor([[1],
            [3],
            [5]])



## Reshape tensors with <tt>.view()</tt>
<a href='https://pytorch.org/docs/master/tensors.html#torch.Tensor.view'><strong><tt>view()</tt></strong></a> and <a href='https://pytorch.org/docs/master/torch.html#torch.reshape'><strong><tt>reshape()</tt></strong></a> do essentially the same thing by returning a reshaped tensor without changing the original tensor in place.<br>
There's a good discussion of the differences <a href='https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch'>here</a>.


```python
x = torch.arange(12)
print(x)
```

    tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])



```python
x.view(2,6)
```




    tensor([[ 0,  1,  2,  3,  4,  5],
            [ 6,  7,  8,  9, 10, 11]])




```python
x.view(6,2)
```




    tensor([[ 0,  1],
            [ 2,  3],
            [ 4,  5],
            [ 6,  7],
            [ 8,  9],
            [10, 11]])




```python
# x is unchanged
x
```




    tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])



### Views reflect the most current data


```python
z = x.view(2,6)
x[0]=234
print(z)
```

    tensor([[234,   1,   2,   3,   4,   5],
            [  6,   7,   8,   9,  10,  11]])


### Views can infer the correct size
By passing in <tt>-1</tt> PyTorch will infer the correct value from the given tensor


```python
# infer number of columns for given rows
x.view(2,-1)
```




    tensor([[234,   1,   2,   3,   4,   5],
            [  6,   7,   8,   9,  10,  11]])




```python
# infer number of rows for given columns
x.view(-1,3)
```




    tensor([[234,   1,   2],
            [  3,   4,   5],
            [  6,   7,   8],
            [  9,  10,  11]])



### Adopt another tensor's shape with <tt>.view_as()</tt>
<a href='https://pytorch.org/docs/master/tensors.html#torch.Tensor.view_as'><strong><tt>view_as(input)</tt></strong></a> only works with tensors that have the same number of elements.


```python
x.view_as(z)
```




    tensor([[234,   1,   2,   3,   4,   5],
            [  6,   7,   8,   9,  10,  11]])



## Tensor Arithmetic
Adding tensors can be performed a few different ways depending on the desired result.<br>

As a simple expression:


```python
a = torch.tensor([1,2,3], dtype=torch.float)
b = torch.tensor([4,5,6], dtype=torch.float)
print(a + b)
```

    tensor([5., 7., 9.])


As arguments passed into a torch operation:


```python
print(torch.add(a, b))
```

    tensor([5., 7., 9.])


With an output tensor passed in as an argument:


```python
result = torch.empty(3)
torch.add(a, b, out=result)  # equivalent to result=torch.add(a,b)
print(result)
```

    tensor([5., 7., 9.])


**Changing a tensor in-place** with *_*


```python
a.add_(b)  # equivalent to a=torch.add(a,b)
print(a)
```

    tensor([5., 7., 9.])


<div class="alert alert-info"><strong>NOTE:</strong> Any operation that changes a tensor in-place is post-fixed with an underscore _.
    <br>In the above example: <tt>a.add_(b)</tt> changed <tt>a</tt>.</div>

### Basic Tensor Operations
<table style="display: inline-block">
<caption style="text-align: center"><strong>Arithmetic</strong></caption>
<tr><th>OPERATION</th><th>FUNCTION</th><th>DESCRIPTION</th></tr>
<tr><td>a + b</td><td>a.add(b)</td><td>element wise addition</td></tr>
<tr><td>a - b</td><td>a.sub(b)</td><td>subtraction</td></tr>
<tr><td>a * b</td><td>a.mul(b)</td><td>multiplication</td></tr>
<tr><td>a / b</td><td>a.div(b)</td><td>division</td></tr>
<tr><td>a % b</td><td>a.fmod(b)</td><td>modulo (remainder after division)</td></tr>
<tr><td>a<sup>b</sup></td><td>a.pow(b)</td><td>power</td></tr>
<tr><td>&nbsp;</td><td></td><td></td></tr>
</table>

<table style="display: inline-block">
<caption style="text-align: center"><strong>Monomial Operations</strong></caption>
<tr><th>OPERATION</th><th>FUNCTION</th><th>DESCRIPTION</th></tr>
<tr><td>|a|</td><td>torch.abs(a)</td><td>absolute value</td></tr>
<tr><td>1/a</td><td>torch.reciprocal(a)</td><td>reciprocal</td></tr>
<tr><td>$\sqrt{a}$</td><td>torch.sqrt(a)</td><td>square root</td></tr>
<tr><td>log(a)</td><td>torch.log(a)</td><td>natural log</td></tr>
<tr><td>e<sup>a</sup></td><td>torch.exp(a)</td><td>exponential</td></tr>
<tr><td>12.34  ==>  12.</td><td>torch.trunc(a)</td><td>truncated integer</td></tr>
<tr><td>12.34  ==>  0.34</td><td>torch.frac(a)</td><td>fractional component</td></tr>
</table>

<table style="display: inline-block">
<caption style="text-align: center"><strong>Trigonometry</strong></caption>
<tr><th>OPERATION</th><th>FUNCTION</th><th>DESCRIPTION</th></tr>
<tr><td>sin(a)</td><td>torch.sin(a)</td><td>sine</td></tr>
<tr><td>cos(a)</td><td>torch.sin(a)</td><td>cosine</td></tr>
<tr><td>tan(a)</td><td>torch.sin(a)</td><td>tangent</td></tr>
<tr><td>arcsin(a)</td><td>torch.asin(a)</td><td>arc sine</td></tr>
<tr><td>arccos(a)</td><td>torch.acos(a)</td><td>arc cosine</td></tr>
<tr><td>arctan(a)</td><td>torch.atan(a)</td><td>arc tangent</td></tr>
<tr><td>sinh(a)</td><td>torch.sinh(a)</td><td>hyperbolic sine</td></tr>
<tr><td>cosh(a)</td><td>torch.cosh(a)</td><td>hyperbolic cosine</td></tr>
<tr><td>tanh(a)</td><td>torch.tanh(a)</td><td>hyperbolic tangent</td></tr>
</table>

<table style="display: inline-block">
<caption style="text-align: center"><strong>Summary Statistics</strong></caption>
<tr><th>OPERATION</th><th>FUNCTION</th><th>DESCRIPTION</th></tr>
<tr><td>$\sum a$</td><td>torch.sum(a)</td><td>sum</td></tr>
<tr><td>$\bar a$</td><td>torch.mean(a)</td><td>mean</td></tr>
<tr><td>a<sub>max</sub></td><td>torch.max(a)</td><td>maximum</td></tr>
<tr><td>a<sub>min</sub></td><td>torch.min(a)</td><td>minimum</td></tr>
<tr><td colspan="3">torch.max(a,b) returns a tensor of size a<br>containing the element wise max between a and b</td></tr>
</table>

<div class="alert alert-info"><strong>NOTE:</strong> Most arithmetic operations require float values. Those that do work with integers return integer tensors.<br>
For example, <tt>torch.div(a,b)</tt> performs floor division (truncates the decimal) for integer types, and classic division for floats.</div>

#### Use the space below to experiment with different operations


```python
a = torch.tensor([1,2,3], dtype=torch.float)
b = torch.tensor([4,5,6], dtype=torch.float)
print(torch.add(a,b).sum())
```

    tensor(21.)


## Dot products
A <a href='https://en.wikipedia.org/wiki/Dot_product'>dot product</a> is the sum of the products of the corresponding entries of two 1D tensors. If the tensors are both vectors, the dot product is given as:<br>

$\begin{bmatrix} a & b & c \end{bmatrix} \;\cdot\; \begin{bmatrix} d & e & f \end{bmatrix} = ad + be + cf$

If the tensors include a column vector, then the dot product is the sum of the result of the multiplied matrices. For example:<br>
$\begin{bmatrix} a & b & c \end{bmatrix} \;\cdot\; \begin{bmatrix} d \\ e \\ f \end{bmatrix} = ad + be + cf$<br><br>
Dot products can be expressed as <a href='https://pytorch.org/docs/stable/torch.html#torch.dot'><strong><tt>torch.dot(a,b)</tt></strong></a> or `a.dot(b)` or `b.dot(a)`


```python
a = torch.tensor([1,2,3], dtype=torch.float)
b = torch.tensor([4,5,6], dtype=torch.float)
print(a.mul(b)) # for reference
print()
print(a.dot(b))
```

    tensor([ 4., 10., 18.])
    
    tensor(32.)


<div class="alert alert-info"><strong>NOTE:</strong> There's a slight difference between <tt>torch.dot()</tt> and <tt>numpy.dot()</tt>. While <tt>torch.dot()</tt> only accepts 1D arguments and returns a dot product, <tt>numpy.dot()</tt> also accepts 2D arguments and performs matrix multiplication. We show matrix multiplication below.</div>

## Matrix multiplication
2D <a href='https://en.wikipedia.org/wiki/Matrix_multiplication'>Matrix multiplication</a> is possible when the number of columns in tensor <strong><tt>A</tt></strong> matches the number of rows in tensor <strong><tt>B</tt></strong>. In this case, the product of tensor <strong><tt>A</tt></strong> with size $(x,y)$ and tensor <strong><tt>B</tt></strong> with size $(y,z)$ results in a tensor of size $(x,z)$
<div>
<div align="left"><img src='https://github.com/shanaka-desoysa/pytorch-deep-learning/blob/master/Images/Matrix_multiplication_diagram.png?raw=1' align="left"><br><br>

$\begin{bmatrix} a & b & c \\
d & e & f \end{bmatrix} \;\times\; \begin{bmatrix} m & n \\ p & q \\ r & s \end{bmatrix} = \begin{bmatrix} (am+bp+cr) & (an+bq+cs) \\
(dm+ep+fr) & (dn+eq+fs) \end{bmatrix}$</div></div>

<div style="clear:both">Image source: <a href='https://commons.wikimedia.org/wiki/File:Matrix_multiplication_diagram_2.svg'>https://commons.wikimedia.org/wiki/File:Matrix_multiplication_diagram_2.svg</a></div>

Matrix multiplication can be computed using <a href='https://pytorch.org/docs/stable/torch.html#torch.mm'><strong><tt>torch.mm(a,b)</tt></strong></a> or `a.mm(b)` or `a @ b`


```python
a = torch.tensor([[0,2,4],[1,3,5]], dtype=torch.float)
b = torch.tensor([[6,7],[8,9],[10,11]], dtype=torch.float)

print('a: ',a.size())
print('b: ',b.size())
print('a x b: ',torch.mm(a,b).size())
```

    a:  torch.Size([2, 3])
    b:  torch.Size([3, 2])
    a x b:  torch.Size([2, 2])



```python
print(torch.mm(a,b))
```

    tensor([[56., 62.],
            [80., 89.]])



```python
print(a.mm(b))
```

    tensor([[56., 62.],
            [80., 89.]])



```python
print(a @ b)
```

    tensor([[56., 62.],
            [80., 89.]])


### Matrix multiplication with broadcasting
Matrix multiplication that involves <a href='https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics'>broadcasting</a> can be computed using <a href='https://pytorch.org/docs/stable/torch.html#torch.matmul'><strong><tt>torch.matmul(a,b)</tt></strong></a> or `a.matmul(b)` or `a @ b`


```python
t1 = torch.randn(2, 3, 4)
t2 = torch.randn(4, 5)
```


```python
t1
```

    tensor([[[ 0.0495, -1.2814,  0.4144,  0.3883],
             [-2.1511,  0.0932,  2.0666,  0.8509],
             [ 0.4211, -2.1292,  0.9620, -1.6141]],
    
            [[ 0.6840, -0.7749,  0.7027,  0.0369],
             [-0.0445,  0.4145, -0.2296,  1.2467],
             [ 0.2800, -1.7043,  0.2537,  0.1963]]])



```python
t2
```




    tensor([[ 1.9903,  0.3279, -0.2475,  0.5449,  0.0568],
            [-0.5038, -0.0790, -0.1920,  0.1574, -0.2723],
            [ 0.1912,  0.8469, -1.7464,  1.1971,  2.7874],
            [-0.8376,  0.5609,  0.8387,  1.5994,  0.0535]])




```python
print(torch.matmul(t1, t2).size())
```

    torch.Size([2, 3, 5])


However, the same operation raises a <tt><strong>RuntimeError</strong></tt> with <tt>torch.mm()</tt>:


```python
print(torch.mm(t1, t2).size())
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-46-edaac219da2b> in <module>()
    ----> 1 print(torch.mm(t1, t2).size())
    

    RuntimeError: matrices expected, got 3D, 2D tensors at /pytorch/aten/src/TH/generic/THTensorMath.cpp:36


___
# Advanced operations

## L2 or Euclidian Norm
See <a href='https://pytorch.org/docs/stable/torch.html#torch.norm'><strong><tt>torch.norm()</tt></strong></a>

The <a href='https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm'>Euclidian Norm</a> gives the vector norm of $x$ where $x=(x_1,x_2,...,x_n)$.<br>
It is calculated as<br>

${\displaystyle \left\|{\boldsymbol {x}}\right\|_{2}:={\sqrt {x_{1}^{2}+\cdots +x_{n}^{2}}}}$


When applied to a matrix, <tt>torch.norm()</tt> returns the <a href='https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm'>Frobenius norm</a> by default.


```python
x = torch.tensor([2.,5.,8.,14.])
x.norm()
```




    tensor(17.)



## Number of elements
See <a href='https://pytorch.org/docs/stable/torch.html#torch.numel'><strong><tt>torch.numel()</tt></strong></a>

Returns the number of elements in a tensor.


```python
x = torch.ones(3,7)
x.numel()
```




    21



This can be useful in certain calculations like Mean Squared Error:<br>
<tt>
def mse(t1, t2):<br>
&nbsp;&nbsp;&nbsp;&nbsp;diff = t1 - t2<br>
    &nbsp;&nbsp;&nbsp;&nbsp;return torch.sum(diff * diff) / diff<strong>.numel()</strong></tt>

<a href="https://colab.research.google.com/github/shanaka-desoysa/notes/blob/master/content/deep_learning/pytorch/tensor_operations.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
