# Interpretability

## Modality Contribution Computation

# Algorithm 1
$$
\begin{algorithm}
	\caption{Computation of Modality Contribution $m_i$}\label{alg:modality_contrib}
	\begin{algorithmic}[1]
		\State $ \vec{d} \gets 0$
		\For{$i$ \textbf{in} $1$ \textbf{to} $n$}
		\State $\vec{d}_i \gets 0$
		\For{$k$ \textbf{in} $1$ \textbf{to} $N$}
		\State $\vec{x}^k \gets$ input\textsubscript{k}
		\State $\vec{p}_0^k \gets$ model($\vec{x}^k$)
		\State $\vec{d}_i^k \gets 0$
		\For{$l$ \textbf{in} $0$ \textbf{to} $h_i-1$}
		\State $\vec{x}_{i,l}^k \gets$ masked\_input\textsubscript{i,l,k}
		\State $\vec{p}_{i,l}^k \gets$ model($\vec{x}_{i,l}^k$)
		\State $\vec{d}_{i,l}^k \gets \lvert\vec{p}_{0}^k-\vec{p}_{i,l}^k\rvert$
		\State $\vec{d}_i^k \gets \vec{d}_i^k + \vec{d}_{i,l}^k$ %/ h_i$   %CHECK IF DIVISION THROUGH hi IS REALIZED IN CODE
		\EndFor
		\State $\vec{d}_i \gets \vec{d}_i + \vec{d}_i^k / N$
		\EndFor
		\State $\vec{d} \gets \vec{d} + \vec{d}_i$
		\EndFor
		\For{$i$ \textbf{in} $1$ \textbf{to} $n$}
		\State $m_i \gets \vec{1}^\tp\vec{d}_i / \vec{1}^\tp \vec{d}$
		\State $\vec{m} \gets [\vec{m},m_i]$
		\EndFor
	\end{algorithmic}
\end{algorithm}
$$
