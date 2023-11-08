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
           rng(numpy generator object) : A random seed that you can use to generate your customers. You can choose to not pass this, in that case the seed taken will be self.rng

       Returns:
           preferences_total(list) : List of size [num_cust, 2, num_toppings], having all generated customer preferences
       """

        preferences_total = []

        if rng is None:
            # standard norm distribution has mean 0 and variance 1
            mean_vector = np.zeros(self.num_toppings)
            # np.eye = Return a 2-D array with ones on the diagonal and zeros elsewhere.
            covariance_matrix = np.eye(self.num_toppings)

            for i in range(num_cust):
                preferences = self.rng.multivariate_normal(mean_vector, covariance_matrix)

                # clip to ensure non-negative values
                preferences = np.clip(preferences, 0, None)
                # normalize
                preferences /= preferences.sum()

                preferences_total.append(preferences.tolist())

        else:
            # standard norm distribution has mean 0 and variance 1
            mean_vector = np.zeros(self.num_toppings)
            # np.eye = Return a 2-D array with ones on the diagonal and zeros elsewhere.
            covariance_matrix = np.eye(self.num_toppings)

            for i in range(num_cust):
                preferences = rng.multivariate_normal(mean_vector, covariance_matrix)

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

    def circle_topping_3_v1(self, preferences):
        """
        Return 1 pizza of 12 toppings in an inner circle, split horizontally,
        and 12 in an outer circle, split vertically.

        Inner circle: 1
        Outer circle: 2, 3
        """
        # toppings 1 (num toppings/n = 24/4 = 6 per topping)
        inner_indices = [1] * 8 + [2] * 8
        # toppings 2 and 3
        outer_indices = [3] * 8

        return self.circle_topping_3(preferences, inner_indices, outer_indices)

    def circle_topping_3_v2(self, preferences):
        """
        Return 1 pizza of 12 toppings in an inner circle, split horizontally,
        and 12 in an outer circle, split vertically.

        Inner circle: 1
        Outer circle: 2, 3
        """
        # toppings 1 (num toppings/n = 24/4 = 6 per topping)
        inner_indices = [1] * 8 + [3] * 8
        # toppings 2 and 3
        outer_indices = [2] * 8

        return self.circle_topping_3(preferences, inner_indices, outer_indices)

    def circle_topping_3_v3(self, preferences):
        """
        Return 1 pizza of 12 toppings in an inner circle, split horizontally,
        and 12 in an outer circle, split vertically.

        Inner circle: 1
        Outer circle: 2, 3
        """
        # toppings 1 (num toppings/n = 24/4 = 6 per topping)
        inner_indices = [2] * 8 + [3] * 8
        # toppings 2 and 3
        outer_indices = [1] * 8

        return self.circle_topping_3(preferences, inner_indices, outer_indices)

    def circle_topping_3(self, preferences, inner_indices, outer_indices):
        """
        Return 1 pizza of 12 toppings in an inner circle, split horizontally,
        and 6 in an outer arc
        """

        theta = np.pi / 16
        inner_radius = BUFFER + 0.375 / np.sin(theta)
        outer_radius = BUFFER + 0.375 / np.sin(theta / 2)

        outer_angle = np.pi / 2

        inner = [
            [
                inner_radius * np.cos((2 * i + 1) * theta),
                inner_radius * np.sin((2 * i + 1) * theta),
                inner_indices[i]
            ]
            for i in range(16)
        ]
        outer = [
            [
                outer_radius * np.cos(outer_angle + (2 * i + 1) * theta),
                outer_radius * np.sin(outer_angle + (2 * i + 1) * theta),
                outer_indices[i]
            ]
            for i in range(8)
        ]
        pizza = inner + outer
        return pizza

    def circle_topping_4_v1(self, preferences):
        """
        Return 1 pizza of 12 toppings in an inner circle, split horizontally,
        and 12 in an outer circle, split vertically.

        Inner circle: 1, 2
        Outer circle: 3, 4
        """
        # toppings 1 and 2 (num toppings/n = 24/4 = 6 per topping)
        inner_indices = [1] * 6 + [2] * 6
        # toppings 3 and 4
        outer_indices = [3] * 6 + [4] * 6

        return self.circle_topping_4(preferences, inner_indices, outer_indices)

    def circle_topping_4_v2(self, preferences):
        """
        Return 1 pizza of 12 toppings in an inner circle, split horizontally,
        and 12 in an outer circle, split vertically

        Inner circle: 3, 4
        Outer circle: 1, 2
        """
        inner_indices = [3] * 6 + [4] * 6
        outer_indices = [1] * 6 + [2] * 6

        return self.circle_topping_4(preferences, inner_indices, outer_indices)

    def circle_topping_4_v3(self, preferences):
        """
        Return 1 pizza of 12 toppings in an inner circle, split horizontally,
        and 12 in an outer circle, split vertically

        Inner circle: 1, 3
        Outer circle: 2, 4
        """
        inner_indices = [1] * 6 + [3] * 6
        outer_indices = [2] * 6 + [4] * 6

        return self.circle_topping_4(preferences, inner_indices, outer_indices)

    def circle_topping_4_v4(self, preferences):
        """
        Return 1 pizza of 12 toppings in an inner circle, split horizontally,
        and 12 in an outer circle, split vertically

        Inner circle: 2, 4
        Outer circle: 1, 3
        """
        inner_indices = [2] * 6 + [4] * 6
        outer_indices = [1] * 6 + [3] * 6

        return self.circle_topping_4(preferences, inner_indices, outer_indices)

    def circle_topping_4_v5(self, preferences):
        """
        Return 1 pizza of 12 toppings in an inner circle, split horizontally,
        and 12 in an outer circle, split vertically

        Inner circle: 3, 2
        Outer circle: 4, 1
        """
        inner_indices = [3] * 6 + [2] * 6
        outer_indices = [4] * 6 + [1] * 6

        return self.circle_topping_4(preferences, inner_indices, outer_indices)

    def circle_topping_4_v6(self, preferences):
        """
        Return 1 pizza of 12 toppings in an inner circle, split horizontally,
        and 12 in an outer circle, split vertically

        Inner circle: 4, 1
        Outer circle: 3, 2
        """
        inner_indices = [4] * 6 + [1] * 6
        outer_indices = [3] * 6 + [2] * 6

        return self.circle_topping_4(preferences, inner_indices, outer_indices)

    def circle_topping_4(self, preferences, inner_indices, outer_indices):
        """
        Return 1 pizza of 12 toppings in an inner circle, split horizontally,
        and 12 in an outer circle, split vertically
        """

        theta = np.pi / 12
        inner_radius = BUFFER + 0.375 / np.sin(theta)
        outer_radius = BUFFER + 0.375 / np.sin(theta / 2)

        outer_angle = np.pi / 2

        inner = [
            [
                inner_radius * np.cos((2 * i + 1) * theta),
                inner_radius * np.sin((2 * i + 1) * theta),
                inner_indices[i]
            ]
            for i in range(12)
        ]
        outer = [
            [
                outer_radius * np.cos(outer_angle + (2 * i + 1) * theta),
                outer_radius * np.sin(outer_angle + (2 * i + 1) * theta),
                outer_indices[i]
            ]
            for i in range(12)
        ]
        pizza = inner + outer
        return pizza
    def radio_topping_3(self, preference):
        numbers = [1, 2, 3] #for ids 
        random.shuffle(numbers)
        pizza = np.zeros((24, 3))
        #start w vertical line 
        y = 1
        id = numbers.pop()
        for i in range(4):
            pizza[i][0] = -.38
            pizza[i][1] = y
            pizza[i][2] = id
            y += .76
        y=1
        for i in range(4,8): #add .76 to x 
            pizza[i][0] = .38
            pizza[i][1] = y
            pizza[i][2] = id
            y += .76
        #now do our spiral ones 
        y = -1 
        x = 1 
        margin = .76/math.sqrt(2)
        x_margin = .76*math.sqrt(3)/2
        y_margin = .38
        mini_margin = .38/math.sqrt(2)
        x_mini_margin = .19
        y_mini_margin = .38*math.sqrt(3)/2
        #start going right 
        id = numbers.pop()
        for i in range(8,12):
            
            y_below = y - y_mini_margin 
            
            x_below = x - x_mini_margin
            pizza[i][0] = x_below
            pizza[i][1] = y_below
            pizza[i][2] = id
            x += x_margin
            y -= y_margin
        y = -1 
        x = 1
        for i in range(12,16):
            y_above = y + y_mini_margin
            x_above = x + x_mini_margin
            pizza[i][0] = x_above
            pizza[i][1] = y_above
            pizza[i][2] = id
            x += x_margin
            y -= y_margin 
        #now going left 
        y = -1 
        x = -1 
        id = numbers.pop()
        for i in range(16,20):
            y_below = y - y_mini_margin 
            x_below = x + x_mini_margin
            pizza[i][0] = x_below
            pizza[i][1] = y_below
            pizza[i][2] = id
            x -= x_margin
            y -= y_margin 
        y = -1 
        x = -1 
        for i in range(20,24):
            y_above = y + y_mini_margin
            x_above = x - x_mini_margin
            pizza[i][0] = x_above
            pizza[i][1] = y_above
            pizza[i][2] = id
            x -= x_margin
            y -= y_margin 
            '''for topping in pizza:
                print("x position is " + str(topping[0]))
                print("y position is " + str(topping[1]))
                print("id is " + str(topping[2]))'''
        return pizza

    def radio_topping_4(self, preferences):
        #pizza = np.zeros((24, 3))
        numbers = [1, 2, 3,4] #for ids 
        random.shuffle(numbers)
        #make these in a line and then going out diagonal 
        pizza = np.zeros((24, 3))
        #start in the line up
        y = 1
        id = numbers.pop()
        for i in range(6):
            pizza[i][0] = 0
            pizza[i][1] = y
            pizza[i][2] = id
            y += .76
        #then down 
        y = -1
        id = numbers.pop()
        for i in range(6,12):
            pizza[i][0] = 0
            pizza[i][1] = y
            pizza[i][2] = id
            y -= .76
        #then right
        y=0
        x = 1
        id = numbers.pop()
        for i in range(12,18):
            pizza[i][0] = x
            pizza[i][1] = 0
            pizza[i][2] = id
            x += .76
        #then left 
        x = -1
        id = numbers.pop()
        for i in range(18,24):
            pizza[i][0] = x
            pizza[i][1] = 0
            pizza[i][2] = id
            x -= .76
        return pizza
    def lines_topping_2(self, preferences):
        # arrange 6 in two lines
        # arrange 4 in clusters
        # honestly for 2 topping the lines may make more sense
        # x_margin = math.sqrt(6**2-4.5**2) #circle geoemetry
        # pizzas = np.zeros((10, 24, 3))
        pizzas = []
        pizza = np.zeros((24, 3))
        x_margin = 1
        # new_y_start_change = (6-math.sqrt(35))/2
        new_y_start_change = .75 * 6
        center_size = .375
        # now lets find the starting point
        x_pos_left = -x_margin - center_size
        x_pos_right = x_margin + center_size

        y_start = new_y_start_change
        # loop thru a range of 12 where we place all w x-x_margin
        # have y start at new_y_start_change and go down .75 each time
        # pizza = []
        y = y_start
        # for i in range(24):
        for i in range(12):
            pizza[i][0] = x_pos_left
            pizza[i][1] = y
            pizza[i][2] = 1
            y -= .76  # to move down
            # print("x_pos left: " + str(x_pos_left) + "and y " + str(y))
        y = y_start
        for j in range(12, 24):
            pizza[j][0] = x_pos_right
            pizza[j][1] = y
            pizza[j][2] = 2
            y -= .76  # to move down
            # print("x_pos right: " + str(x_pos_right) + "and y " + str(y))

        '''for topping in pizza:
                print("this is x " + str(topping[0]))
                print("this is y " + str(topping[1]))
                print("this is id " + str(topping[2]))'''

        #commenting out if we use random choice of number of each algo per pizza
        # for x in range(10):
        #     pizzas.append(pizza)

        '''for pizza in pizzas:
            for topping in pizza:
                print("this is x " + str(topping[0]))
                print("this is y " + str(topping[1]))
                print("this is id " + str(topping[2]))'''

        print("using this function")
        # return list(pizzas)
        return pizza

        # do the same with x+x_margin and toppin id 2

        # repeat 10 times to make all 10 pizzas

        # return list of pizzas


    def lines_topping_3(self, preferences):
        numbers = [1, 2, 3]
        random.shuffle(numbers)
        pizzas = []
        pizza = np.zeros((24, 3))
        x_margin = 1.5
        # new_y_start_change = (6-math.sqrt(35))/2
        new_y_start_change = .75 * 4
        center_size = .375
        # now lets find the starting point
        x_pos_left = -x_margin #- center_size
        x_pos_middle = 0#center_size
        x_pos_right = x_margin #+ center_size

        y_start = new_y_start_change
        y = y_start
        # for i in range(24):
        id = numbers.pop()
        for i in range(8):
            pizza[i][0] = x_pos_left
            pizza[i][1] = y
            pizza[i][2] = id
            y -= .76  # to move down
            # print("x_pos left: " + str(x_pos_left) + "and y " + str(y))
        y = y_start
        id = numbers.pop()
        for j in range(8, 16):
            pizza[j][0] = x_pos_middle
            pizza[j][1] = y
            pizza[j][2] = id
            y -= .76  # to move down
        y = y_start
        id = numbers.pop()
        for k in range(16, 24):
            pizza[k][0] = x_pos_right
            pizza[k][1] = y
            pizza[k][2] = id
            y -= .76  # to move down
            # print("x_pos right: " + str(x_pos_right) + "and y " + str(y))
        return pizza

    def lines_topping_4(self, preferences):
        pizzas = []
        pizza = np.zeros((24, 3))
        numbers = [1, 2, 3, 4]
        random.shuffle(numbers)
        x_margin = 1.5
        # new_y_start_change = (6-math.sqrt(35))/2
        new_y_start_change = .75 * 3
        center_size = .375
        # now lets find the starting point
        x_pos_left1 = -2.25 - center_size
        x_pos_left2 = -.75 - center_size
        x_pos_right1 = .75 + center_size
        x_pos_right2 = 2.25 + center_size

        y_start = new_y_start_change
        y = y_start
        # for i in range(24):
        id = numbers.pop()
        for i in range(6):
            pizza[i][0] = x_pos_left1
            pizza[i][1] = y
            pizza[i][2] = id 
            y -= .76  # to move down
            # print("x_pos left: " + str(x_pos_left) + "and y " + str(y))
        y = y_start
        id = numbers.pop()
        for j in range(6, 12):
            pizza[j][0] = x_pos_left2
            pizza[j][1] = y
            pizza[j][2] = id
            y -= .76  # to move down
        y = y_start
        id = numbers.pop()
        for k in range(12, 18):
            pizza[k][0] = x_pos_right1
            pizza[k][1] = y
            pizza[k][2] = id
            y -= .76  # to move down
        y = y_start
        id = numbers.pop()
        for l in range(18, 24):
            pizza[l][0] = x_pos_right2
            pizza[l][1] = y
            pizza[l][2] = id
            y -= .76  # to move down
            # print("x_pos right: " + str(x_pos_right) + "and y " + str(y))
        return pizza

    def list_sum_to_total(self, total, num_values):
        if num_values < 1:
            raise ValueError("Number of values must be at least 1.")
        if num_values == 1:
            return [total]

        distribution = []

        for _ in range(num_values - 1):
            value = random.randint(0, total)
            distribution.append(value)
            total -= value

        distribution.append(total)
        random.shuffle(distribution)

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
            # We have 2 algos for now
            num_runs_per_approach = self.list_sum_to_total(10, 2)
            print(f'num_runs_per_approach: {num_runs_per_approach}')
            #
            # pizzas = []
            # for i in range(num_runs_per_approach[0]):
            #     pizzas.append(self.lines_topping_2(preferences))
            # for i in range(num_runs_per_approach[1]):
            #     pizzas.append(self.circle_topping_2(preferences))
            # return pizzas

            pizzas = [self.lines_topping_2(preferences)] * 5 + [self.circle_topping_2(preferences)] * 5

            return pizzas


        elif self.num_toppings == 3:
         
            # num_runs_per_approach = self.list_sum_to_total(10, 4)
            # print(f'num_runs_per_approach: {num_runs_per_approach}')
            #
            # pizzas = []
            # for i in range(10):
            #     pizzas.append(self.radio_topping_3(preferences))
            '''for i in range(num_runs_per_approach[0]):
                pizzas.append(self.circle_topping_3_v1(preferences))
            for i in range(num_runs_per_approach[1]):
                pizzas.append(self.circle_topping_3_v2(preferences))
            for i in range(num_runs_per_approach[2]):
                pizzas.append(self.circle_topping_3_v3(preferences))
            for i in range(num_runs_per_approach[3]):
                pizzas.append(self.lines_topping_3(preferences))'''
            # for i in range(10):
            #     pizzas.append(self.lines_topping_3(preferences))
            # return pizzas

            # 2 circles of each type (6) + 2 lines + 2 radial lines
            pizzas = [self.circle_topping_3_v1(preferences)] * 2 + [self.circle_topping_3_v2(preferences)] * 2 + [self.circle_topping_3_v3(preferences)] * 2 + [self.lines_topping_3(preferences)] * 2 + [self.radio_topping_3(preferences)] * 2

            return pizzas


        elif self.num_toppings == 4:

            """once we have multiple approaches, we can randomly choose which one to use
            THEN, we can loop through them for each value in the distribution to generate 10 pizzas"""

            # # For now, we have 2 approaches for 4 toppings. need to manually code for however many approaches we have
            # num_runs_per_approach = self.list_sum_to_total(10, 9)
            # print(f'num_runs_per_approach: {num_runs_per_approach}')
            #
            # pizzas = []
            #
            # for i in range(num_runs_per_approach[0]):
            #     pizzas.append(self.circle_topping_4_v1(preferences))
            # for i in range(num_runs_per_approach[1]):
            #     pizzas.append(self.circle_topping_4_v2(preferences))
            #
            # for i in range(num_runs_per_approach[2]):
            #     pizzas.append(self.lines_topping_4(preferences))
            # for i in range(num_runs_per_approach[3]):
            #     pizzas.append(self.lines_topping_4(preferences))
            #
            # for i in range(num_runs_per_approach[4]):
            #     pizzas.append(self.circle_topping_4_v3(preferences))
            # for i in range(num_runs_per_approach[5]):
            #     pizzas.append(self.circle_topping_4_v4(preferences))
            # for i in range(num_runs_per_approach[6]):
            #     pizzas.append(self.circle_topping_4_v5(preferences))
            # for i in range(num_runs_per_approach[7]):
            #     pizzas.append(self.circle_topping_4_v6(preferences))
            #
            # for i in range(num_runs_per_approach[8]):
            #     pizzas.append(self.radio_topping_4(preferences))
            # return pizzas


            # 1 circle of each type (6) + 2 lines + 2 radial lines
            pizzas = [self.circle_topping_4_v1(preferences)] * 1 + [self.circle_topping_4_v2(preferences)] * 1 + [self.circle_topping_4_v3(preferences)] * 1 + [self.circle_topping_4_v4(preferences)] * 1 + [self.circle_topping_4_v5(preferences)] * 1 + [self.circle_topping_4_v6(preferences)] * 1 + [self.lines_topping_4(preferences)] * 2 + [self.radio_topping_4(preferences)] * 2

            return pizzas

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
