from tokenize import String
import numpy as np
from typing import Tuple, List
import constants
from utils import pizza_calculations

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

        mean = 0.5
        std_dev = 2.0
        
        preferences_total = []
        if rng==None:
            np.random.seed(self.rng)
            for i in range(num_cust):
                preferences_1 = np.random.normal(mean, std_dev, self.num_toppings)
                print(f'preferences 1 self.rng {preferences_1}')
                preferences_1 = np.clip(preferences_1, 0, None)  # Ensure preferences are non-negative
                preferences_1 /= preferences_1.sum()  # Normalize the preferences
                preferences_total.append([preferences_1.tolist(), preferences_1.tolist()])  # Duplicate preferences
        else :
            np.random.seed(rng)
            for i in range(num_cust):
                preferences_1 = np.random.normal(mean, std_dev, self.num_toppings)
                print(f'preferences 1 rng {preferences_1}')
                preferences_1 = np.clip(preferences_1, 0, None)  # Ensure preferences are non-negative
                preferences_1 /= preferences_1.sum()  # Normalize the preferences
                preferences_total.append([preferences_1.tolist(), preferences_1.tolist()])  # Duplicate preferences

        print(f'preferences total {preferences_total}')
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
            pizzas = np.zeros((10, 24, 3))
            pizza = np.zeros((24, 3))
            x_margin = 1 
            new_y_start_change = (6-mat.sqrt(35))/2 
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
                y -= .75 #to move down
            y = y_start
            for j in range(12,24):
                pizza[i][0] = x_pos_right
                pizza[i][1] = y
                pizza[i][2] = 2 
                y -= .75 #to move down
            
            for x in range(10):
                pizzas.append(pizza)
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
        return  remaining_pizza_ids[0], [0,0], np.pi/8
