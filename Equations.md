

### Adversarial training in a view of frequency analysis



Consider an input $x \in R^N$ be a sequence with N timestamp. Its DFT can be written as 

​										$F_k = \sum^{N-1}_{i=0} x_i e^{-j\frac{2\pi}{N}ki}$

And the inverser DFT

​										$x_i = \frac{1}{N}\sum^{N-1}_{k=0} F_k e^{j\frac{2\pi}{N}ki}$

where $j = \sqrt{-1}, i, k $ are the indices. 



First we consider the general adversarial training objective 



​															$$\min_\theta \rho (\theta)$$

Where 

​														$$\rho(\theta) = \mathbb{E}_{(x, y)\sim D} [\max_{||x'-x||_p \leq \delta_p}L(\theta, x', y)]$$



We first expand $||x' - x||_p$
$$
\begin{align}
||x' - x||_p &= (\sum^{N-1}_{i=0}(x'_i-x_i)^p)^{\frac{1}{p}}\\
&=(\sum^{N-1}_{i=0}(\frac{1}{N}\sum^{N-1}_{k=0} F'_k e^{j\frac{2\pi}{N}ki} - \frac{1}{N}\sum^{N-1}_{k=0} F_k e^{j\frac{2\pi}{N}ki})^p)^{\frac{1}{p}}\\
&=\frac{1}{N}(\sum^{N-1}_{i=0}(\sum^{N-1}_{k=0} (F'_k-F_k) e^{j\frac{2\pi}{N}ki})^p)^{\frac{1}{p}}\\
\end{align}
$$
if $p=1$, the lower bound can be found:
$$
\begin{align}
||x' - x||_1 &= \frac{1}{N}\sum^{N-1}_{i=0}|\sum^{N-1}_{k=0} (F'_k-F_k) e^{j\frac{2\pi}{N}ki}|\\
&\geq\frac{1}{N}|\sum^{N-1}_{i=0}\sum^{N-1}_{k=0} (F'_k-F_k) e^{j\frac{2\pi}{N}ki}|\\
&\geq\frac{1}{N}|\sum^{N-1}_{k=0} (F'_k-F_k) \sum^{N-1}_{i=0}e^{j\frac{2\pi}{N}ki}|\\
&\geq\frac{1}{N}|\sum^{N-1}_{k=0} (F'_k-F_k) w_k| \quad \text{(weighted $\ell_1$)}\\
&\geq|F'_0-F_0|\\
\end{align}
$$
Where 
$$
w_k = \frac{\sin(\pi k)}{\sin(\frac{\pi}{N}k)}e^{jk\pi\frac{N-1}{N}} =  \begin{cases}
                                   N & \text{if $k=0$} \\
                                   0 & \text{if $1 \leq k \leq N-1$} \\
  \end{cases}
$$
Source of exponential sum: https://archive.lib.msu.edu/crcmath/math/math/e/e413.htm

while the upperbound is given by 
$$
\begin{align}
||x' - x||_1 &= \frac{1}{N}\sum^{N-1}_{i=0}|\sum^{N-1}_{k=0} (F'_k-F_k) e^{j\frac{2\pi}{N}ki}|\\
&\leq\frac{1}{N}\sum^{N-1}_{i=0}\sum^{N-1}_{k=0} |(F'_k-F_k) e^{j\frac{2\pi}{N}ki}|\\
&\leq\frac{1}{N}\sum^{N-1}_{k=0} |(F'_k-F_k)|\\

\end{align}
$$








