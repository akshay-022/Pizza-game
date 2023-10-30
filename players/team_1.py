from tokenize import String
import numpy as np
from typing import Tuple, List
import constants
from utils import pizza_calculations
import math
import random

class Player:
    def __init__(self, num_toppings, rng: np.random.Generator) -> None:
        """Initialise the player"""
        self.rng = rng
        self.num_toppings = num_toppings

    def customer_gen(self, num_cust, rng = None):
        
        """Function in which we create a distribution of customer preferences

        Args:
            num_cust(int) : the total number of customer preferences you need to create
            rng(int) : A random seed that you can use to generate your customers. You can choose to not pass this, in that case the seed taken will be self.rng

        Returns:
            preferences_total(list) : List of size [num_cust, 2, num_toppings], having all generated customer preferences
        """

        preferences_total = []

        if rng is None:
            np.random.seed(self.rng)
        else:
            np.random.seed(rng)

        # standard norm distribution has mean 0 and variance 1
        mean_vector = np.zeros(self.num_toppings)
        # np.eye = Return a 2-D array with ones on the diagonal and zeros elsewhere.
        covariance_matrix = np.eye(self.num_toppings)

        for i in range(num_cust):
            preferences = np.random.multivariate_normal(mean_vector, covariance_matrix)

            # clip to ensure non-negative values
            preferences = np.clip(preferences, 0, None)
            # normalize
            preferences /= preferences.sum()

            preferences_total.append(preferences.tolist())

        return preferences_total

    #def choose_discard(self, cards: list[str], constraints: list[str]):
    def choose_toppings(self, preferences):
        """Function in which we choose position of toppings

        Args:
            num_toppings(int) : the total number of different topics chosen among 2, 3 and 4
            preferences(list) : List of size 100*2*num_toppings for 100 generated preference pairs(actual amounts) of customers.

        Returns:
            pizzas(list) : List of size [10,24,3], where 10 is the pizza id, 24 is the topping id, innermost list of size 3 is [x coordinate of topping center, y coordinate of topping center, topping number of topping(1/2/3/4) (Note that it starts from 1, not 0)]
        """
        if self.num_toppings == 2:
            #arrange 6 in two lines 
            #arrange 4 in clusters 
            #honestly for 2 topping the lines may make more sense 
            #x_margin = math.sqrt(6**2-4.5**2) #circle geoemetry 
            #pizzas = np.zeros((10, 24, 3))
            pizzas = []
            pizza = np.zeros((24, 3))
            x_margin = 1 
            #new_y_start_change = (6-math.sqrt(35))/2 
            new_y_start_change = .75*6
            center_size = .375 
            #now lets find the starting point 
            x_pos_left = -x_margin - center_size 
            x_pos_right = x_margin + center_size 

            y_start = new_y_start_change
            #loop thru a range of 12 where we place all w x-x_margin 
            #have y start at new_y_start_change and go down .75 each time 
            #pizza = []
            y = y_start
            #for i in range(24):
            for i in range(12):
                pizza[i][0] = x_pos_left
                pizza[i][1] = y
                pizza[i][2] = 1 
                y -= .76 #to move down
                #print("x_pos left: " + str(x_pos_left) + "and y " + str(y))
            y = y_start
            for j in range(12,24):
                pizza[j][0] = x_pos_right
                pizza[j][1] = y
                pizza[j][2] = 2 
                y -= .76 #to move down
                #print("x_pos right: " + str(x_pos_right) + "and y " + str(y))
            
            '''for topping in pizza: 
                    print("this is x " + str(topping[0]))
                    print("this is y " + str(topping[1]))
                    print("this is id " + str(topping[2]))'''
            
            for x in range(10):
                pizzas.append(pizza)

            '''for pizza in pizzas: 
                for topping in pizza: 
                    print("this is x " + str(topping[0]))
                    print("this is y " + str(topping[1]))
                    print("this is id " + str(topping[2]))'''

            print("using this function")
            return list(pizzas)
                

            #do the same with x+x_margin and toppin id 2 

            #repeat 10 times to make all 10 pizzas 

            #return list of pizzas 
        print("using this function oop")
        x_coords = [np.sin(np.pi/2)]
        pizzas = np.zeros((10, 24, 3))
        for j in range(constants.number_of_initial_pizzas):
            pizza_indiv = np.zeros((24,3))
            i = 0
            while i<24:
                angle = self.rng.random()*2*np.pi
                dist = self.rng.random()*6
                x = dist*np.cos(angle)
                y = dist*np.sin(angle)
                clash_exists = pizza_calculations.clash_exists(x, y, pizza_indiv, i)
                if not clash_exists:
                    pizza_indiv[i] = [x, y, i%self.num_toppings + 1]
                    i = i+1
            pizza_indiv = np.array(pizza_indiv)
            pizzas[j] = pizza_indiv
        for pizza in pizzas:
            for topping in pizza:
                print("x postion is " + str(topping[0]))
                print("y postion is " + str(topping[1]))
        return list(pizzas)
    



    #def play(self, cards: list[str], constraints: list[str], state: list[str], territory: list[int]) -> Tuple[int, str]:
    def choose_and_cut(self, pizzas, remaining_pizza_ids, customer_amounts):
        """Function which based n current game state returns the distance and angle, the shot must be played

        Args:
            pizzas (list): List of size [10,24,3], where 10 is the pizza id, 24 is the topping id, innermost list of size 3 is [x coordinate of topping, y coordinate of topping, topping number of topping(1/2/3/4)]
            remaining_pizza_ids (list): A list of remaining pizza's ids
            customer_amounts (list): The amounts in which the customer wants their pizza

        Returns:
            Tuple[int, center, first cut angle]: Return the pizza id you choose, the center of the cut in format [x_coord, y_coord] where both are in inches relative of pizza center of radius 6, the angle of the first cut in radians. 
        """
        pizza_id = remaining_pizza_ids[0]
        x =  random.uniform(-6, 6)
        y = self.circleCoordinates(x)
        return  remaining_pizza_ids[0], [x,y], np.pi/6

    #this function will take in an x and return the other y coordinate from the equation of a circle
    def circleCoordinates(self, x):
        positive = random.randint(0, 1) #make y the top half or bottom half of circle
        y = 0 
        radius = 6
        if positive:
            y = math.sqrt((radius**2) - (x**2))
        else:
            y = -(math.sqrt((radius**2) - (x**2)))
        
        return y

            


