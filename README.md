
![](./assets/demo.png)

*dem0* is a chess engine trained *tabula rasa* using the  [AlphaGo Zero](https://www.nature.com/articles/nature24270) self-play pipeline. dem0 trains on the [PettingZoo chess environment](https://pettingzoo.farama.org/environments/classic/chess/). dem0 is evaluated using [ELO rating](https://en.wikipedia.org/wiki/Elo_rating_system) computed by the  [BayesELO](https://www.remi-coulom.fr/Bayesian-Elo/) software. 

## Project TODOs
- Pipeline
    1. Self-play
        - Everything
    2. Evaluator
        - Everything
    3. Optimizer
        - Everything
    - Stockfish 5 was released on 2014-05-31
- Presentation
    - Make presentation
- For Fun
    - Make pixelart of dem0 sitting on dead competitors
    - Make animated gif of dem0 moving its head
    - Make dem0 with traffic cone on its head

## Network Architecture

As per the paper:

---

### **Convolutional Block**
> *The convolutional block applies the following modules:*  
> **(1)** A convolution of 256 filters of kernel size 3 × 3 with stride 1  
> **(2)** Batch normalization  
> **(3)** A rectifier nonlinearity  

---

### **Residual Block**
> *Each residual block applies the following modules sequentially to its input:*  
> **(1)** A convolution of 256 filters of kernel size 3 × 3 with stride 1  
> **(2)** Batch normalization  
> **(3)** A rectifier nonlinearity  
> **(4)** A convolution of 256 filters of kernel size 3 × 3 with stride 1  
> **(5)** Batch normalization  
> **(6)** A skip connection that adds the input to the block  
> **(7)** A rectifier nonlinearity
---

### **Policy Head**
> *The output of the residual tower is passed into two separate ‘heads’ for computing the policy and value. The policy head applies the following modules:*  
> **(1)** A convolution of 2 filters of kernel size 1 × 1 with stride 1  
> **(2)** Batch normalization  
> **(3)** A rectifier nonlinearity  
> **(4)** A fully connected linear layer that outputs a vector of size $19^2 + 1 = 362$, corresponding to logit probabilities for all intersections and the pass move **(NOTE: DIFFERENT SIZE FOR CHESS!)**

---

### **Value Head**
> *The value head applies the following modules:*  
> **(1)** A convolution of 1 filter of kernel size 1 × 1 with stride 1  
> **(2)** Batch normalization  
> **(3)** A rectifier nonlinearity  
> **(4)** A fully connected linear layer to a hidden layer of size 256  
> **(5)** A rectifier nonlinearity  
> **(6)** A fully connected linear layer to a scalar  
> **(7)** A tanh nonlinearity outputting a scalar in the range [−1, 1] 
---

