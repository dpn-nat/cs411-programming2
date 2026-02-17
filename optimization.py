import constraint
import math
import random
from simanneal import Annealer
import cvxpy as cp
import numpy as np

############################## PROBLEM 1 ######################################
# In problem 1, you are going to implement CSP for Sudoku problem. Implement cstAdd,
# which adds the constraints.  It takes a problem object (problem), a matrix of variable
# names (grid), a list of legal values (domains), and the side length of the inner squares
# (psize, which is 3 in an ordinary sudoku and 2 in the smaller version we provide as
# the easier test case).

""" A helper function to visualize ouput.  You do not need to change this """
""" output: the output of your solver """
""" psize: the problem size (e.g. 3 for a 3x3 grid of 3x3 squares in a standard Sudoku) """
def sudokuCSPToGrid(output,psize):
    if output is None:
        return None
    dim = psize**2
    return np.reshape([[output[str(dim*i+j+1)] for j in range(dim)] for i in range(dim)],(dim,dim))

""" helper function to add variables to the CSP """
""" you do not need to change this"""
""" Note how we initialize the domains to the supplied values on the marked line """
def addVar(problem, grid, domains, init):
    numRow = grid.shape[0]
    numCol = grid.shape[1]
    for rowIdx in range(numRow):
        for colIdx in range(numCol):
            if grid[rowIdx, colIdx] in init: #use supplied value
                problem.addVariable(grid[rowIdx,colIdx], [init[grid[rowIdx, colIdx]]])
            else:
                problem.addVariable(grid[rowIdx,colIdx], domains)

                    
""" here you want to add all of the constraints needed.
    problem: the CSP problem instance we have created for you
    grid: a psize ** 2 by psize ** 2 array containing the CSP variables
    domains: the domain for the variables representing non-pre-filled squares
    psize: the problem size (e.g. 3 for a 3x3 grid of 3x3 squares in a standard Sudoku)
    # Hint: Use loops!
    #       Remember problem.addConstraint() to add constraints
    #       Example syntax for adding a constraint that two variable are not equal:
    #       problem.addConstraint(lambda a, b: a !=b, (variable1,variable2)
    #       See the example file for more"""
def cstAdd(problem, grid, domains,psize):
    # --------------------
    # Your code
    # this is for the total size of the sudoku board
    sz = psize ** 2
    # this is to add the row constraints and each row should contain all different values
    for r in range(sz):
        row = []
        for c in range(sz):
            row.append(grid[r][c])
        problem.addConstraint(constraint.AllDifferentConstraint(), row)
    # this is for the column constraints and each column should have all different values
    for c in range(sz):
        col = []
        for r in range(sz):
            col.append(grid[r][c])
        problem.addConstraint(constraint.AllDifferentConstraint(), col)
    # this is for the subgrid constraits and each subgrid should have different values
    for sum in range(psize):
        for other in range(psize):
            # store all the variables in teh subggrid
            sqare = []
            startRow = sum * psize
            startCol = other * psize

            for dr in range(psize):
                for dc in range(psize):
                    sqare.append(grid[startRow + dr][startCol + dc])
            # add all the different constraints to the subgrid
            problem.addConstraint(constraint.AllDifferentConstraint(), sqare)
    # --------------------

""" Implementation for a CSP Sudoku Solver """
""" positions: list of (row,column,value) triples representing the already filled in cells"""
""" psize: the problem size (e.g. 3 for a 3x3 grid of 3x3 squares in a standard Sudoku) """
""" You do not need to change this """
def sudokuCSP(positions,psize):
    sudokuPro = constraint.Problem()
    dim = psize ** 2
    numCol = dim
    numRow = dim
    domains = list(range(1,dim+1))
    init = {str(dim*p[0]+p[1]+1):p[2] for p in positions}
    sudokuList = [str(i) for i in range(1,dim**2+1)]
    sudoKuGrid = np.reshape(sudokuList, [numRow, numCol])
    addVar(sudokuPro, sudoKuGrid, domains, init)
    cstAdd(sudokuPro, sudoKuGrid, domains,psize)
    return sudokuPro.getSolution()

############################## PROBLEM 2 ######################################
# In the fractional knapsack problem you have a knapsack with a fixed weight capacity
# and want to fill it with valuable items so that we maximize the total value in it
# while ensuring the weight does notexceed the capacity. Fractions of items are allowed
#

""" Frational Knapsack Problem
    c: the capacity of the knapsack
    Hint: Think carefully about the range of values your variables can be, and include them in the constraints"""
def fractionalKnapsack(c):
    # -------------------
    # Your code
    # First define some variables
    v = cp.Variable(3)    # defining 3 different varibales using python function. 

    weights = [5, 3, 1]  # given in problem     
    values = [2, 3, 1]   # given in problem 

    # Put your constraints here
    constraints = [   # it has to be between 0 and 1 and is capacity should be less than c 
        v >= 0 ,      
        v <= 1 ,
        # the following constraint will ensure that our total weight is less than our capacity
        # so what I did is i took weight values one by one and multiplied by fraction of item 1 taken 
        # so weight[0] = 5 and v[0] fraction of that item 1 is taken. 
        # so an example to understand is v[0] = 0.2 i took 20% of item 1 weight is 5 = so 5 * 0. 2 = 1  
        # so weight used of item 1 is  = 1  
        weights[0]*v[0] + weights[1]*v[1] + weights[2]*v[2] <= c 
    ]

    # Fix this to be the correct objective function
    # and similarly we do the same thing for our value here
    # becasue we want to maximize our objective. again v[.] is  fraction of that item taken * is value. 
    # so let's continue the example from earlier. v[0] = 0.2 then value is 2 for item 1 so 2 * 0.2 = 0.4 total value of item 1
    obj = cp.Maximize(values[0]*v[0] + values[1]*v[1] + values[2]*v[2])

    # End of your code
    # ------------------
    prob = cp.Problem(obj, constraints)
    return prob.solve()


############################## PROBLEM 3 ######################################
# Integer Programming: Sudoku
# We have provided most of an IP implementation.
# Again, you just need to implement the constraints.  Note however, unlike in the CSP version,
# we have not already “prefilled” the squares for you.  You’ll need to add those constraints yourself.

""" A helper function to visualize ouput.  You do not need to change this """
""" binary: the output of your solver """
""" psize: the problem size (e.g. 3 for a 3x3 grid of 3x3 squares in a standard Sudoku) """
def sudokuIPToGrid(binary,psize):
    if binary is None:
        return None
    dim = psize**2
    x = np.zeros((dim,dim),dtype=int)
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                if binary[dim*i+j][k] >= 0.99:
                    x[i][j] = k+1
    return x

""" Implementation for a IP Sudoku Solver """
""" positions: list of (row,column,value) triples representing the already filled in cells"""
""" psize: the problem size (e.g. 3 for a 3x3 grid of 3x3 squares in a standard Sudoku) """
""" the library does not support 3D variables, so M[i][j] should be your indicator variable """
""" for the ith square having value j where in a 4x4 grid i ranges from 0 to 15 """
def sudokuIP(positions,psize):
    # Define the variables - see comment above about interpretation
    # given code
    dim = psize**2
    M = cp.Variable((dim**2,dim),integer=True) #Sadly we cannot do 3D Variables

    constraints = []
    # --------------------
    # Your code
    # It should define the constraints needed
    # We've given you one to get you started
    # given code
    constraints.extend([0 <= M[x][k] for x in range(dim**2) for k in range (dim)])

    # defining another constraint
    constraints.extend([M[x][k] <= 1 for x in range(dim**2) for k in range(dim)])
    # each square must contain exactly one number
    constraints.extend([cp.sum(M[i, :]) == 1 for i in range(dim**2)])
    # this is for the row constraints and for each row and each value of k, the value must appear exactly once in row r
    for ro in range(dim):
        for x in range(dim):
            constraints.append(cp.sum([M[ro*dim + co, x] for co in range(dim)]) == 1)
    # this is for the column constraints and for each column c and each value k and that value would appar exactly once in column c.
    for co in range(dim):
        for x in range(dim):
            constraints.append(cp.sum([M[ro*dim + co, x] for ro in range(dim)]) == 1)
    # subgrid constraints for each subgrid and each value k
    for bro in range(psize):
        for bco in range(psize):
            for x in range(dim):
                constraints.append(cp.sum([M[(bro*psize + dro)*dim + (bco*psize + dco), x]for dro in range(psize) for dco in range(psize)]) == 1)
    # thsi is for cell constraints
    for (r, c, val) in positions:
        index = r * dim + c
        constraints.append(M[index, val-1] == 1)
    # End your code
    # -------------------

    # Form dummy objective - we only care about feasibility
    # given code
    obj = cp.Minimize(M[0][0])

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    prob.solve()
    #Uncomment the version below instead if you want more detailed information from the solver to see what might be going wrong
    #Please leave commented out when submitting.
    #prob.solve(verbose=True)
    #For debugging you may want to look at some of the information contained in prob before returning
    #See the example file
    return M.value
    # --------------------

############################## PROBLEM 4 ######################################
# Local Search: TSP
# We have provided most of a simulated annealing implementation of the famous traveling salesman problem,
# where you seek to visit a list of cities while minimizing the total distance traveled.
# You need to implement move and energy.
# The former is the operation for finding nearby candidate solutions while the latter
# evaluates how good the current candidate solution is.
# Move should generate a random local move without regard for whether it is beneficial.
# Similarly, to receive credit energyshould calculate the total euclidean distance of the current candidate tour.
# There is a distance function you may wish to implement to help with this.

class TravellingSalesmanProblem(Annealer):

    """problem specific data"""
    # latitude and longitude for the twenty largest U.S. cities
    cities = {
        'New York City': (40.72, 74.00),
        'Los Angeles': (34.05, 118.25),
        'Chicago': (41.88, 87.63),
        'Houston': (29.77, 95.38),
        'Phoenix': (33.45, 112.07),
        'Philadelphia': (39.95, 75.17),
        'San Antonio': (29.53, 98.47),
        'Dallas': (32.78, 96.80),
        'San Diego': (32.78, 117.15),
        'San Jose': (37.30, 121.87),
        'Detroit': (42.33, 83.05),
        'San Francisco': (37.78, 122.42),
        'Jacksonville': (30.32, 81.70),
        'Indianapolis': (39.78, 86.15),
        'Austin': (30.27, 97.77),
        'Columbus': (39.98, 82.98),
        'Fort Worth': (32.75, 97.33),
        'Charlotte': (35.23, 80.85),
        'Memphis': (35.12, 89.97),
        'Baltimore': (39.28, 76.62)
    }

    """problem-specific helper function"""
    """you may wish to implement this """
    def distance(self, a, b):
        """Calculates distance between two latitude-longitude coordinates."""
        # -----------------------------
        lati1, longi1 = self.cities[a]
        lati2, longi2 = self.cities[b]
        # used the math formula to find the distance
        return math.sqrt((lati1-lati2)**2 + (longi1 - longi2)**2)
        # -----------------------------



    """ make a local change to the solution"""
    """ a natural choice is to swap to cities at random"""
    """ current state is available as self.state """
    """ Note: This is just making the move (change) in the state,
              Worry about whether this is a good idea elsewhere. """
    """ Make sure there is a way for enough of your local changes to
              reach a solution """
    def move(self):

        # --------------------
        x = len(self.state)

        # to get a random index that is 1 less than x
        y = random.randint(0, x-1)
        # to get another random index that is 1 less than x
        z = random.randint(0, x-1)
        # the random indexes should not be the same
        while z == y:
            z = random.randint(0,x-1)
        
        # this is to swap each one of them by using a temporary storage for one and then to swap the other
        temp = self.state[y]
        self.state[y] = self.state[z]
        self.state[z] = temp
        # -------------------------


    """ how good is this state? """
    """ lower is better """
    """ current state is available as self.state """
    """ Use self.cities to find a city's coordinates"""
    def energy(self):
        # Initialize the value to be returned
        e = 0
        # number of cities in the current state
        n = len(self.state)
        # goes thoruhg each city
        for i in range(n):
            # gets the current city
            cit1 = self.state[i]
            # gets the next city and the i+1 % n is so that when it reaches the last, it wraps around to get 0
            cit2 = self.state[(i + 1) % n]  
            # adds the distance to get the otal energy
            e = e + self.distance(cit1, cit2)
        
        #-----------------------
        # Your code



        #-----------------------
        # returns it
        return e

# Execution part, please don't change it!!!
def annealTSP(initial_state):
        # initial_state is a list of starting cities
        tsp = TravellingSalesmanProblem(initial_state)
        return tsp.anneal()

############################## PROBLEM 5 ######################################
# Local Search: Sudoku
# Now we have the skeleton of a simulated annealing implemen-tation of Sudoku.
# You need to design the move and energy functions and will receive credit based on
# how many of 10 runs succeed in finding a correct answer:  to achieve k points 2k−1 runs need to pass

class SudokuProblem(Annealer):

    """ positions: list of (row,column,value) triples representing the already filled in cells"""
    """ psize: the problem size (e.g. 3 for a 3x3 grid of 3x3 squares in a standard Sudoku) """
    def __init__(self,initial_state,positions,psize):
        self.psize = psize
        self.positions = positions
        super(SudokuProblem, self).__init__(initial_state)

    """ make a local change to the solution"""
    """ current state is available as self.state """
    """ Hint: Remember this is sudoku, just make one local change
              print self.state may help to get started"""
    """ Note that the initial state we give you is purely random
              and may not even respect the filled in squares. """
    """ Make sure there is a way for enough of your local changes to
              reach a solution """
    def move(self):
        # --------------------
        size = self.psize * self.psize  # here we calculate the total size of the grib by using self.psize 

        i = random.randint(0, size - 1) # to pick a random state we use random pyton function to pick between 0 and size -1(index) 
        j = random.randint(0, size - 1) # we pick i and j number row and random column to check if is filled or not filled 

        while any(position[0] == i and position[1] == j for position in self.positions): # this is important becasue this will check if the random i,j we chose has a filled square meaning 
            # does it have a value already in it. if is fixed then 
            i = random.randint(0, size - 1) # choose another i 
            j = random.randint(0, size - 1) # choose anoter j  will go until position in the grid is not fixed. 

        index  = i * size + j  # because we looked at in 2d we convert back to the 1d array grid. with taking row multiply by the size and adding the colum 
        self.state[index] = random.randint(1, size) # just making one local change by assigning one random value to not fixed in grid. 
 
        for row, column, value in self.positions: # very important becasue we need to restore the fixed position to thier corresponding values else, gives me duplicate and also out of order. 
            self.state[row * size + column] = value # so again convert to 1d array and set the correct value in those self.position grid.
        # -------------------------


    """ how good is this state? """
    """ lower is better """
    """ current state is available as self.state """
    """ Remember what we talked about in class for the energy function for a CSP """
    def energy(self):
        # Initialize the value to be returned
        e = 0 
        #-----------------------
        size = self.psize * self.psize    # here we calculate the total size of the grib by using self.psize

        # to check duplicate for rows 
        for rows in range(size):  # starting with we loop through all the rows(size cause every grid is different)
            start = rows * size # calcualting row starting point to begin checking duplicate 
            end = start + size # and where duplicate check will end is going to be done by starting index and is size.
            row = self.state[start:end]  # we can get the whole row by using slice method. so using : we get the entire row
            duplicates = len(row) - len(set(row)) # the way i did this is by checking element vs set of elemet as we know set will not have any duplicate so we can check by getting the length of 
            # entire row and then getting the set of the row and then counting if = 0 then no duplicate if = any number > 0 than duplicate are present. 
            e += duplicates # add that duplicate number over here to the energy. 

        for c in range(size):  # Loop through each column in the grid 
            col = []  # i made an empty to get all values that are in the column 
            # the reason to me using list is cause in 1d array the index for colum are seprated out and not next to each other like rows... 
            
        
            for r in range(size):  # we go to each row to get all elements for all columns 
                index = r * size + c  # like in move defination, we convert back to 1d index 
                col.append(self.state[index]) # we than increase the value of the position to our column list. 
            
            # Now count duplicates
            duplicates = len(col) - len(set(col)) # again like in rows duplicate. i did by checking element vs set of elemet as we know set will not have any duplicate so we can check by getting the length of 
            # entire row and then getting the set of the row and then counting if = 0 then no duplicate if = any number > 0 than duplicate are present. 
            e += duplicates # and then add duplicate to energy. 

        # we also have to check that each block must have all unique value ;( 
        for block_row in range(self.psize):  # loop through each block row self.prize is not full grid just block. 
            for block_col in range(self.psize):  #  and then go through column. 
                block = [] # made an empty list to collect all values in the block. 
                
                for r in range(block_row * self.psize, (block_row + 1) * self.psize): # i calculate what rows belong to the specific subgrid. so we don't mix values. 
                    for c in range(block_col * self.psize, (block_col + 1) * self.psize):  # and for columns the same thing, which specifc columns belong to this subgrid 
                        index = r * size + c # again like earlier convert back to 1d array. 
                        block.append(self.state[index]) # increase the value to our list. 
                
                duplicates = len(block) - len(set(block)) # again check if duplicate exist
                e += duplicates # then increase that to energy if does. 
        #----------------------- 
        return e

# Execution part, please don't change it!!!
def annealSudoku(positions, psize):
        # initial_state of starting values:
        initial_state = [random.randint(1,psize**2) for i in range(psize ** 4)]
        sudoku = SudokuProblem(initial_state,positions,psize)
        sudoku.steps = 100000
        sudoku.Tmax = 100.0
        sudoku.Tmin = 1.0
        return sudoku.anneal()

