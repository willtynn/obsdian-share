---
{"dg-publish":true,"permalink":"/论文阅读笔记/DNN拆分调研/Adaptive DNN Partition in Edge Computing Environments/"}
---

Adaptive DNN Partition in Edge Computing Environments
Doi：[10.1109/ICPADS51040.2020.00097](https://sci-hub.ru/10.1109/ICPADS51040.2020.00097)
年份：2020
## Summary
本文提出了一种负载均衡算法，用于根据边缘计算环境中的环境自适应地通过有向无环图（DAG）拓扑来拆分深度神经网络（DNN）。目的是在计算资源有限的终端设备上加快 DNN 模型的推理速度。所提出的算法可以有效地将DNN推理任务分成几个小任务，并将其分配给不同的边缘设备。该论文还提到了未来的改进，包括考虑设备计算能力的动态变化，使拆分更加精细，改进算法以及测试更多的DNN。
## Research Objective
本文重点研究了在没有任何数据中心的边缘计算环境下，具有DAG拓扑结构的DNN的拆分策略。我们提出了一种有效的算法来**根据当前环境自适应地划分DNN**，包括带宽、计算设备的数量、这些设备的计算能力、给定DNN的特定拓扑结构等。

所提算法主要采用负载均衡和贪婪策略。我们首先考虑所有设备具有完全相同计算能力的情况，然后将其推广到设备可能具有不同计算能力的情况。我们通过大量的实验对所提出的算法进行了评估。实验结果表明，本文提出的算法能够有效地提高DNN的推理速度。
## Background and Challenges
随着深度神经网络( Deep Neural Network，DNN )模型越来越复杂，从链式拓扑结构演进到有向无环图( Directed Acyclic Graph，DAG )拓扑结构，终端设备受其便携性的限制，计算能力往往较低，无法满足DNN模型推理的需求。 

DNN划分的核心思想是将DNN拆分成若干个部分，在不同的边缘节点完成不同的部分。在这种情况下，如何拆分DNN以及如何根据具体环境将任务分配给不同的边缘节点是主要的挑战。
## Method
### Problem Definition
- $L=\{l_0,l_1,...,l_{n-1}\}$：DNN模型的n层集合。
- $Br=\{B_0,B_1,...,B_{n-1}\}$：DNN模型中各层的分支情况，其中$B_i=\{b_0,b_1,...,b_{x-1}\}$。
- $T=\{T_{0},T_{1},...,T_{n-1}\}$：DNN模型中各层的计算时间，如果该层有多个分支，则 $T_i=\{t_0,t_1,...,t_{x-1}\}$。
- $D=\{d_0,d_1,...,d_{n-1}\}$：各层的输入大小。
### **设备同质情况**
![Pasted image 20230728164710.png|300](/img/user/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/DNN%E6%8B%86%E5%88%86%E8%B0%83%E7%A0%94/_resource/Pasted%20image%2020230728164710.png)
将同一层中的不同分支分配给不同的边缘设备以减少整个时间开销，本质上是一个负载均衡问题。拆分时，整个时间成本取决于最后得到结果的边缘设备。因此，为了最小化整个推理时间，任务应该尽可能均匀地分布，使得不同设备的时间消耗尽可能接近。对于运行在主机设备上的分支，预测的时间消耗为分配给设备的分支的计算时间之和；对于运行在辅助设备上的分支，预测的时间消耗为分配给设备的分支的计算时间和中间数据传输时间之和。 
### **设备异质情况**
差别在于每种设备的计算能力不同，那么每台设备进行DNN推理计算的时间也就不同。
## Evaluation
### Setting
实验的硬件环境为Raspberry Pi 4B ( 4G RAM )。操作系统为2020年2月13日发布的Lite版本的Raspbian。代码语言为python，版本为3.5。安装的机器学习框架为Py - torch0.4 . 0。 
### Detail
- DNN：Inceptionv4
- 实验主要在多设备条件下，设置不同的设备运算能力，即对CPU进行分级，分别进行实验，以延迟为指标进行对比。
## Conclusion
计算能力有限的边缘设备无法独立运行现代DNN推理。为了解决这个问题，提出了在边缘设备之间拆分DNN的算法。我们对算法进行了实现和评估，结果表明算法是有效的。我们将在未来改进我们的工作，包括考虑设备计算能力的动态变化，使分裂更加细粒度，改进我们的算法，测试更多的DNN等等。 
## Note
实际上这是一种模型并行的分布式推理场景，算法和问题定义都比较简单。

