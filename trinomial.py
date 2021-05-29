import numpy as np
import matplotlib.pyplot as plt
from threading import Thread




def get_params():
    """Asks the user to input the parameters of the model and calculates the variables"""
    """S0 = float(input("initial price of stock:\n"))
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
    
    TEST INPUTS 1 comment out previous and uncomment this
    -----------------------------------------------------"""
    S0 = 20
    T = 3
    option = None
    r = 0.05 
    sigma = 0.6
    n = 5
    dt = T/n
    q = 0.2
    K = 25#float(input("Strike price:\n"))
    #---------------------------- 
    u = np.exp(sigma*np.sqrt(2*dt))
    d = 1/u
    m = 1
    p_u = ((np.exp((r-q)*dt/2.)-np.exp((-sigma)*np.sqrt(dt/2.)))/(np.exp((sigma)*np.sqrt(dt/2.))-np.exp((-sigma)*np.sqrt(dt/2))))**2
    p_d = ((np.exp((sigma)*np.sqrt(dt/2.))-np.exp((r-q)*dt/2.))/(np.exp((sigma)*np.sqrt(dt/2.))-np.exp((-sigma)*np.sqrt(dt/2))))**2
    p_m = 1- (p_u +p_d)
    return n,r,sigma,dt,q,option,S0,K,u,d,m,p_u,p_d,p_m

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
    tree = np.zeros([1+2*(n),n+1])
    for i in range(n+1):
        #go through tree vertically and calculate the prices based on previous prices
        for j in range(1+ 2*i):
            if i == 0 and j==0:
                tree[j,i] = S0
            if tree[j,i] == 0:
                if j ==  2*i:
                    #calculate down step
                    tree[j,i]= d * tree[j-2,i-1]
                    if tree[j,i]<0:
                        print(f"less than zero")
                        tree[j,i] = 0
                elif j == 0:
                    tree[j,i] = u * tree[0,i-1]
                    #calculate up step
                else:
                    #calculate middle step
                    tree[j,i] = tree[j-1,i-1]
    print(f"tree[0,1], {tree[1,0]}")
    return tree
#print(calc_underlying_price_tree(4,1.5,0.5,1,100))
def options_tree_tree(u,d,m,dt,n,S0,K,r,p_u,p_d,p_m):
    #create a tree of underlying price value
    price_tree =calc_underlying_price_tree(n,u,d,m,S0)
    #print(f"price_tree: {price_tree}")
    options_tree = np.zeros([np.shape(price_tree)[0],n+1])
    #options_tree on the last column is the maximum of S_n - K, 0
    #print(f"np.zeros(n+1): {n+1} np.shape(price_tree)[0]: {np.shape(price_tree)[0]} price_tree[:,n]: {np.shape(price_tree[:,n])}")
    options_tree[:,n] = np.maximum(np.zeros(np.shape(price_tree)[0]),price_tree[:,n]-K)
    #calc option price at the t=0 from the back starting from second
    #last column since last column already calculated
    #print(f"Initial options_tree_tree: {options_tree}")
    for i in np.arange(n-1,-1,-1):
        for j in np.arange(0,i+1):
            if options_tree[j,i] == 0 :
                #working backwards we calculate options_tree 
                options_tree[j,i]= np.exp(-r*dt)*(p_u*options_tree[j,i+1] + p_m*options_tree[j+1,i+1]+ p_d*options_tree[j+2,i+1]) 
    return [options_tree[0,0], price_tree, options_tree]
def plot_graph(tree, title):
    """plot points on the graph"""
    m,n = np.shape(tree)
    tabk = np.arange(0,n)
    plt.title(title)
    plt.xlabel("Time steps n")
    plt.ylabel("Price")
    for i in range(n-1):
        for j in range(1+2*i):
            plt.plot([i,i+1], [tree[j][i],tree[j][i+1]], marker ='o')
            plt.plot([i,i+1], [tree[j][i],tree[j+1][i+1]], marker ='o')
            plt.plot([i,i+1], [tree[j][i],tree[j+2][i+1]], marker ='o')
    plt.show()
    
def generate_tree():
    n,r,sigma,dt,q,option,S0,K,u,d,m,p_u,p_d,p_m = get_params() #get parameters
    initial_opt, underlying_tree, options_tree = options_tree_tree(u,d,m,dt,n,S0,K,r,p_u,p_d,p_m) #calculate the options_tree tree and price trees
    return options_tree, underlying_tree,initial_opt

""""
TEST INPUTS
---------------------------------------
S0 = 20
T = 3
option = None
r = 0.05
sigma = 0.2
n = 5
q = 0.2
K = 25
dt = T/n
dS = S0/100

A test run calculating changing initial price parameters
options1, underlying1 = generate_tree(S0, T, option,r, sigma, n ,q, dt) 
options2, underlying2 = generate_tree(S0+dS, T, option,r, sigma, n ,q, dt) 

print("underlying tree shape",np.shape(options1), "underlying tree\n", underlying1)

def calc_Delta(option_tree1, option_tree2, dS):
    Delta_tree = np.zeros([1+2*(n-1),n+1])
def main():
    plot_graph(options1, "options_tree tree")
    plot_graph(underlying1, "Price tree")
    plot_graph(options2, "options_tree +dS")
    plot_graph(underlying2, "Price tree + dS")"""

def main():
    options_tree, underlying_tree,initial_opt = generate_tree()
    print(f"initial option price: {initial_opt}")
    #threads to run at the same time however not working
    t1 = Thread(target = plot_graph(options_tree, "Options Tree"))
    t2 = Thread(target = plot_graph(underlying_tree, "Underlying Tree"))
    t1.start()
    t2.start()
if __name__ == "__main__":
    main()

