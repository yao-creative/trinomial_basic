import numpy as np
import matplotlib.pyplot as plt
import math

def get_params():
    """Asks the user to input the parameters of the model"""
    """TEST INPUTS
    S0 = 20
    T = 3
    option = None
    r = 0.05 
    sigma = 0.6
    n = 5
    dt = T/n
    q = 0.2"""
    S0 = float(input("initial price of stock:\n"))
    T = float(input("Period of Time:\n"))
    option = None
    r = float(input("Risk free interest rate:\n"))
    sigma = float(input("Volatility of stock:\n"))
    n = int(input("Number of time steps:\n"))
    dt = T/n
    q = float(input("Dividend yield:\n"))
    #makes sure p_m is within (0,1)
    while dt>= 2*(sigma/(r-q))**2:
        print(f"invalid delta t")
        r = float(input("Risk free interest rate:\n"))
        sigma = float(input("Volatility of stock:\n"))
        n = int(input("Number of time steps:\n"))
        dt = T/n
        q = float(input("Dividend yield:\n"))
    K = 25#float(input("Strike price:\n"))
    return n,r,sigma,dt,q,option,S0,K

def calc_formulas(r,sigma,dt,q):
    """Using the given paramters derive other parameters based on wikipedia"""
    u = np.exp(sigma*np.sqrt(2*dt))
    d = 1/u
    m = 1
    p_u = ((np.exp((r-q)*dt/2.)-np.exp((-sigma)*np.sqrt(dt/2.)))/(np.exp((sigma)*np.sqrt(dt/2.))-np.exp((-sigma)*np.sqrt(dt/2))))**2
    p_d = ((np.exp((sigma)*np.sqrt(dt/2.))-np.exp((r-q)*dt/2.))/(np.exp((sigma)*np.sqrt(dt/2.))-np.exp((-sigma)*np.sqrt(dt/2))))**2
    p_m = 1- (p_u +p_d)
    print(f"u = {u}, d = {d}, p_u = {p_u}, p_d = {p_d}, p_m= {p_m}")
    return u,d,m,p_u,p_d,p_m

def calc_underlying_price_tree(n,u,d,m,S0):
    """out put a trinomial matrix of prices that starts from (0,0) will increment two more rows used every time"""

    tree = np.zeros([1+2*(n-1),n+1])
    for i in range(n+1):
        #go through tree vertically and calculate the prices based on previous prices
        for j in range(i+2):
            if i == 0 and j==0:
                tree[j,i] = S0
            if tree[j,i] == 0: 
                if j == i+1:
                    #calculate down step
                    tree[j,i]= d * tree[j-2,i-1]
                    if tree[j,i]<0:
                        #print(f"less than zero")
                        tree[j,i] = 0
                elif j == 0:
                    tree[j,i] = u * tree[0,i-1]
                    #calculate up step
                else:
                    #calculate middle step
                    tree[j,i] = tree[j-1,i-1]
    return tree
print(calc_underlying_price_tree(4,1.5,0.5,1,100))
def options_tree(u,d,m,dt,n,S0,K,r,p_u,p_d,p_m):
    #create a tree of underlying price value
    price_tree =calc_underlying_price_tree(n,u,d,m,S0)
    print(f"price_tree: {price_tree}")
    options = np.zeros([np.shape(price_tree)[0],n+1])
    #options on the last column is the maximum of S_n - K, 0
    #print(f"np.zeros(n+1): {n+1} np.shape(price_tree)[0]: {np.shape(price_tree)[0]} price_tree[:,n]: {np.shape(price_tree[:,n])}")
    options[:,n] = np.maximum(np.zeros(np.shape(price_tree)[0]),price_tree[:,n]-K)
    #calc option price at the t=0 from the back starting from second
    #last column since last column already calculated
    print(f"Initial options_tree: {options}")
    for i in np.arange(n-1,-1,-1):
        for j in np.arange(0,i+1):
            print(f"i: {i} j: {j} options[j+2,i+1]: {options[j+2,i+1]}")
            if options[j,i] == 0 :
                #working backwards we calculate options 
                options[j,i]= np.exp(-r*dt)*(p_u*options[j,i+1] + p_m*options[j+1,i+1]+ p_d*options[j+2,i+1]) 
    return [options[0,0], price_tree, options]
def plot_graph(tree, title):
    """plot points on the graph"""
    m,n = np.shape(tree)
    tabk = np.arange(0,n)
    plt.title(title)
    plt.xlabel("Time steps n")
    plt.ylabel("Price")
    for i in range(n-1):
        for j in range(i+1):
            plt.plot([i,i+1], [tree[j][i],tree[j][i+1]], marker ='o')
            plt.plot([i,i+1], [tree[j][i],tree[j+1][i+1]], marker ='o')
            plt.plot([i,i+1], [tree[j][i],tree[j+2][i+1]], marker ='o')
    plt.show()
def main():

    n,r,sigma,dt,q,option,S0,K = get_params() #get parameters
    u,d,m,p_u,p_d, p_m = calc_formulas(r,sigma,dt,q)  #calculate some formulas
    initial_opt, price_tree, options = options_tree(u,d,m,dt,n,S0,K,r,p_u,p_d,p_m) #calculate the options tree and price trees
    print(f"intial option: {initial_opt}\n price tree:\n {price_tree}\n options tree:\n {options}")
    plot_graph(price_tree, "Price tree")
    plot_graph(options, "Options tree")
    
if __name__ == "__main__":
    main()

