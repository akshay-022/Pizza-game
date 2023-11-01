from tokenize import String
import numpy as np
from typing import Tuple, List
import constants
from utils import pizza_calculations
import math
import random

#constants
BUFFER = 0.001

class Player:
    def __init__(self, num_toppings, rng: np.random.Generator) -> None:
        """Initialise the player"""
        self.rng = rng
        self.num_toppings = num_toppings
        self.multiplier=40	# Pizza radius = 6*multiplier units
        self.xCenter = 12*self.multiplier	# Center Point x of pizza
        self.yCenter = 10*self.multiplier	# Center Point y of pizza
        self.calculator = pizza_calculations()

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
    def circle_topping_2(self, preferences):
        """
        Return 1 pizza of 24 toppings in a circle, split horizontally
        """
        radius = BUFFER + 0.375 / np.sin(np.pi / 24)
        theta = np.pi / 24
        angle = 0  # use np.pi/2 to make a vertical half
        pizza = [
            [
                radius * np.cos(angle + (2 * i + 1) * theta),
                radius * np.sin(angle + (2 * i + 1) * theta),
                1 + i // 12
            ]
            for i in range(24)
        ]
        return pizza

    def circle_topping_4(self, preferences):
        """
        Return 1 pizza of 12 toppings in an inner circle, split horizontally,
        and 12 in an outer circle, split vertically
        """

        #0.189 (+ buffer) was minimum i could find for the inner circle to NOT overlap for now
        inner_radius = BUFFER + 0.189 / np.sin(np.pi / 24)
        outer_radius = BUFFER + 0.375 / np.sin(np.pi / 24)

        theta = np.pi / 12

        # found angle thru testing around unit circle until appeared ~ vertically split
        outer_angle = 11 * np.pi / 6

        # make a small inner circle, split horizontally
        inner = [
            [
                inner_radius*np.cos((2*i+1)*theta),
                inner_radius*np.sin((2*i+1)*theta),
                1+i//6
            ]
            for i in range(12)
        ]
        # make a larger outer circle, split vertically
        outer = [
            [
                outer_radius*np.cos((outer_angle + (2*i+1))*theta),
                outer_radius*np.sin((outer_angle + (2*i+1))*theta),
                1+i//6
            ]
            for i in range(12, 24)
        ]
        pizza = inner + outer
        return pizza

    def generate_random_distribution(total, num_values):
        """
        Generate random distributions of integers that add up to a given total for multiple distributions.
        We may use this to randomly decide how many of each approach we may use for topping placement
        """

        distribution = []

        for _ in range(num_values - 1):
            value = random.randint(0, total)
            distribution.append(value)
            total -= value

        distribution.append(total)
        random.shuffle(distribution)  # Shuffle the values for randomness

        return distribution

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

            # we can use the approach distribution here to decide how many of each approach we want to use
            # approach_distribution = generate_random_distribution(10, 2)

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


        # DEFAULT FOR 3 TOPPINGS
        elif self.num_toppings == 3:
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

        elif self.num_toppings == 4:

            """once we have multiple approaches, we can randomly choose which one to use
            approach_distribution = generate_random_distribution(10, NUM_OF_APPROACHES)
            THEN, we can loop through them for each value in the distribution to generate 10 pizzas"""

            # For now, we have no other approaches for 4 toppings, so we will just use 10 of these
            return [self.circle_topping_4(preferences)] * 10


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
        maximumS = -1000
        maximumCut = [self.xCenter, self.yCenter, np.pi/6, remaining_pizza_ids[0]] #default cut

        for pizza_id in remaining_pizza_ids:
            pizza = pizzas[pizza_id]
            for radius in range(0, 6): 
                for x in range(-radius*5, radius*5):
                    x =  x/5
                    for ySign in range(-1, 2, 2):
                        y = self.circleCoordinates(x, ySign, radius)
                        cut = [x, y, np.pi/6, pizza_id]
                        
                        xCord = (self.xCenter + x*self.multiplier)
                        yCord = (self.yCenter - y*self.multiplier)
                        obtained_pref, slice_areas_toppings = self.calculator.ratio_calculator(pizza, [xCord, yCord, np.pi/6], self.num_toppings, self.multiplier, self.xCenter, self.yCenter)
                        obtained_pref = np.array(obtained_pref)
                        random_pref, temp = self.calculator.ratio_calculator(pizza, [self.xCenter, self.yCenter, self.rng.random()*2*np.pi], self.num_toppings, self.multiplier, self.xCenter, self.yCenter)
                        random_pref = np.array(random_pref)
                        required_pref = np.array(customer_amounts)
                        uniform_pref = np.ones((2, self.num_toppings))*(12/self.num_toppings)
                        b = np.round(np.absolute(required_pref - uniform_pref), 3)
                        c = np.round(np.absolute(obtained_pref - required_pref), 3)
                        u = np.round(np.absolute(random_pref - uniform_pref), 3)
                        s = (b-c).sum()
                        if s > maximumS:
                            maximumS = s
                            maximumCut = cut           
        x  = maximumCut[0]
        y = maximumCut[1]
        theta = maximumCut[2]
        pizza_id = maximumCut[3]
        return  pizza_id, [x,y], theta

    #this function will take in an x, sign of y, radius and return the y coordinate from the equation of a circle
    def circleCoordinates(self, x, ySign, radius):
        y = ySign*(math.sqrt((radius**2) - (x**2)))
        return y

            
    ###############################

    # def create_radial_lines(self, num_lines, radius, angle_offset=0):
    #     lines = []
    #     toppings_per_line = num_lines // self.num_toppings
    #     print(f'toppings per line: {toppings_per_line}')
    #     # for topping_id in range(1, self.num_toppings + 1):
    #     # print(f'topping id: {topping_id}')
    #     for i in range(toppings_per_line):
    #         angle = (2 * np.pi / toppings_per_line) * i + angle_offset
    #         x = radius * np.cos(angle)
    #         y = radius * np.sin(angle)
    #         lines.append([x, y, topping_id])
    #
    #     for i in range(toppings_per_line, 24):
    #         angle = (2 * np.pi / toppings_per_line) * i + angle_offset
    #         x = (radius + 1) * np.cos(angle)
    #         y = (radius + 1) * np.sin(angle)
    #         lines.append([x, y, topping_id])
    #
    #     return lines
    ##############################
    # def radial_toppings(self, preferences):
    #     """
    #     Function that will place toppings in a radial line pattern
    #
    #     Args:
    #         preferences(list) : List of size 100*2*num_toppings for 100 generated preference pairs(actual amounts) of customers.
    #
    #     Return: 1 pizza with radial pattern of toppings
    #     """
    #
    #     # place lines of only one topping in a radial pattern around the center
    #     if self.num_toppings == 3:
    #         # need to place them so they are right next to each other and will occupy the full 6in radius
    #         pizzas = np.zeros((10, 24, 3))
    #         # constants.number_of_initial_pizzas
    #         for j in range(1):
    #             pizza = np.zeros((24, 3))
    #
    #             cos = np.cos
    #             sin = np.sin
    #             radius = BUFFER  # Start with the buffer distance
    #             spacing = 0.75  # The desired spacing between points
    #
    #             # # Create a list of points with 0 degrees difference between each point
    #             # pizza = []
    #
    #             for i in range(0, 8):
    #                 angle = 0
    #                 x = radius * cos(angle)
    #                 y = radius * sin(angle)
    #                 topping_id = 1 + i // 8  # Update topping_id based on your requirements
    #                 pizza[i] = ([x, y, topping_id])
    #
    #                 # Increase the radius for the next point
    #                 radius += spacing
    #                 # + BUFFER?
    #
    #             for i in range(8, 16):
    #                 angle = 2 * np.pi / 3
    #                 x = radius * cos(angle)
    #                 y = radius * sin(angle)
    #                 topping_id = 1 + i // 8  # Update topping_id based on your requirements
    #                 pizza[i] = ([x, y, topping_id])
    #
    #                 # Increase the radius for the next point
    #                 radius += spacing
    #                 # + BUFFER?
    #
    #             for i in range(16, 24):
    #                 angle = 2 * np.pi / 3
    #                 x = radius * cos(angle)
    #                 y = radius * sin(angle)
    #                 topping_id = 1 + i // 8  # Update topping_id based on your requirements
    #                 pizza[i] = ([x, y, topping_id])
    #
    #                 # Increase the radius for the next point
    #                 radius += spacing
    #                 # + BUFFER?
    #             pizza = np.array(pizza)
    #             pizzas[j] = pizza
    #
    #     elif self.num_toppings == 4:
    #         pizzas = np.zeros((10, 24, 3))
    #         # constants.number_of_initial_pizzas
    #         for j in range(constants.number_of_initial_pizzas):
    #             pizza = np.zeros((24, 3))
    #             cos = np.cos
    #             sin = np.sin
    #             BUFFER = 0  # 0.001
    #             radius = 0.5  # Start with the buffer distance
    #             spacing = 0.8  # The desired spacing between points
    #
    #             for i in range(0, 6):
    #                 angle = 0
    #                 x = radius * cos(angle)
    #                 y = radius * sin(angle)
    #                 topping_id = 1  # Update topping_id based on your requirements
    #                 pizza[i] = [x, y, topping_id]
    #                 print(f'x: {x}, y: {y}, topping_id: {topping_id}')
    #                 # Increase the radius for the next point
    #                 radius += spacing + BUFFER
    #                 # + BUFFER?
    #
    #             for i in range(6, 12):
    #                 angle = np.pi / 2
    #                 x = radius * cos(angle)
    #                 y = radius * sin(angle)
    #                 topping_id = 2  # Update topping_id based on your requirements
    #                 pizza[i] = [x, y, topping_id]
    #                 print(f'x: {x}, y: {y}, topping_id: {topping_id}')
    #
    #                 # Increase the radius for the next point
    #                 radius += spacing + BUFFER
    #                 # + BUFFER?
    #
    #             for i in range(12, 18):
    #                 angle = np.pi
    #                 x = radius * cos(angle)
    #                 y = radius * sin(angle)
    #                 topping_id = 3  # Update topping_id based on your requirements
    #                 pizza[i] = [x, y, topping_id]
    #                 print(f'x: {x}, y: {y}, topping_id: {topping_id}')
    #
    #                 # Increase the radius for the next point
    #                 radius += spacing + BUFFER
    #                 # + BUFFER?
    #
    #             for i in range(18, 24):
    #                 angle = 3 * np.pi / 2
    #                 x = radius * cos(angle)
    #                 y = radius * sin(angle)
    #                 topping_id = 4  # Update topping_id based on your requirements
    #                 pizza[i] = [x, y, topping_id]
    #                 print(f'x: {x}, y: {y}, topping_id: {topping_id}')
    #
    #                 # Increase the radius for the next point
    #                 radius += spacing + BUFFER
    #
    #             pizza = np.array(pizza)
    #             pizzas[j] = pizza
    #
    #         return list(pizzas)
    ###############################

