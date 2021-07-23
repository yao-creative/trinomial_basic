<<<<<<< HEAD
# trinomial_basic
Basic trinomial tree for pricing options
=======
# MAA107 project trinomial basic model simulation by Yi Yao Tan, Ecole Polytechnique BX23
## Desciption
Self-guided research project for the class mathematical modeling in the Bachelor Program.
### trinomial.py
One step derivation based on the probabilities and then plotted on matplotlib. Derivation is based on paper attached.

### trinomial_old.py
Using the online wikipedia formulas creates a multistep trinomial and plots using matplotlib

## Functions:
Prints out price matrix and then options matrix. 
Then uses an optimized method to plot the basic trinomial tree and pricing

## Running the program:
1) Just put: "$ python3 trinomial.py "
into your terminal and then key in the input variables

2) Uncommment test inputs

3) Key in the variables:
initial price of stock
Period of Time
Risk free interest rate
Volatility of stock
Number of time steps
Dividend yield
Strike price

## Note:
to keep probability of staying the middle p_m we have to have the following condition: 
dt < 2 * sigma^2/(r-q)^2


>>>>>>> 82a70e2471f9c1ffb342228ff46fa794dc3f1f81
