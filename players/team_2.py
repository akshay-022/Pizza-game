from tokenize import String
import numpy as np
from typing import Tuple, List
import constants
from utils import pizza_calculations
import math
import random
import sys
sys.path.append('..')

from utils import pizza_calculations
# from pizza_gui import gui

class Player:
    def __init__(self, num_toppings, rng: np.random.Generator) -> None:
        """Initialise the player"""
        self.rng = rng
        self.num_toppings = num_toppings
        self.calculator = pizza_calculations()
        # self._gui = gui()
        self.multiplier= 1 # self.gui.multiplier	# Pizza radius = 6*multiplier units
        self.x = 12*self.multiplier	# Center Point x of pizza
        self.y = 10*self.multiplier	# Center Point y of pizza

        self.uniform_ = 0
        self.topping_1 = 0
        self.topping_2 = 0
        self.topping_3 = 0
        self.topping_4 = 0
        self.counter_ = 0

    def customer_gen(self, num_cust, rng = None):
        
        """Function in which we create a distribution of customer preferences

        Args:
            num_cust(int) : the total number of customer preferences you need to create
            rng(int) : A random seed that you can use to generate your customers. You can choose to not pass this, in that case the seed taken will be self.rng

        Returns:
            preferences_total(list) : List of size [num_cust, 2, num_toppings], having all generated customer preferences
        """

        def create_inst(_k):
            p = []
            for idx in range(self.num_toppings):
                item = float("nan")
                while math.isnan(item):
                    item = np.random.normal(loc=(24/(2*self.num_toppings)) - k, scale=1)
                    item = max(min(item, 11.9), 0.1)
                p.append(item)
            p = np.array(p)
            p = 12 * p / np.sum(p)
            return p

        preferences_total = []
        if rng==None:
            for i in range(num_cust):
                k = np.random.randint(-6,6)
                # print("k is:", k)
                preferences_1 = create_inst(k)
                preferences_2 = create_inst(k)
                preferences = [preferences_1, preferences_2]
                equal_prob = self.rng.random()
                if equal_prob <= 0.0:
                    preferences = (np.ones((2,self.num_toppings))*12/self.num_toppings).tolist()
                preferences_total.append(preferences)
                """
                preferences_1 = self.rng.random((self.num_toppings,))
                preferences_1 = 12*preferences_1/np.sum(preferences_1)
                preferences_2 = self.rng.random((self.num_toppings,))
                preferences_2 = 12*preferences_2/np.sum(preferences_2)
                preferences = [preferences_1, preferences_2]
                equal_prob = self.rng.random()
                if equal_prob <= 0.0:
                    preferences = (np.ones((2,self.num_toppings))*12/self.num_toppings).tolist()
                preferences_total.append(preferences)
                """
        else : 
            for i in range(num_cust):
                k = np.random.randint(-6,6)
                print("k is:", k)
                preferences_1 = create_inst(k)
                preferences_2 = create_inst(k)
                """
                preferences_1 = rng.random((self.num_toppings,))
                preferences_1 = 12*preferences_1/np.sum(preferences_1)
                preferences_2 = rng.random((self.num_toppings,))
                preferences_2 = 12*preferences_2/np.sum(preferences_2)
                """
                preferences = [preferences_1, preferences_2]
                equal_prob = rng.random()
                if equal_prob <= 0.0:       #change this if you want toppings to show up
                    preferences = (np.ones((2,self.num_toppings))*12/self.num_toppings).tolist()
                preferences_total.append(preferences)
        # print("Preference total", preferences_total)
        return preferences_total

    def isUniform(self, array):
        for index in range(len(array)-1):
            if array[index+1]==array[index]:
                self.uniform_+=1
        
        if self.uniform_== len(array)-1:
            return 1
        else:
            return 0     

    def largest_num(self, array):
        max_ = -1000
        for i in array:
            if i>max_:
                max_ = i
        
        return max_
    
    def avg_by_ten(self, array):
        if len(array)!=0:
            i = 0
            sum_ = []
            #print("this is the array", array)
            preference = array[0][0]
            for item in preference:
                sum_.append(0)

            while i < 10:
                for index in range(len(sum_)):
                    #print(index, i)
                    sum_[index] = sum_[index]+array[i][0][index]+array[i][1][index]
                i+=1

                for index in range(len(sum_)):
                    #print(index, i)
                    sum_[index] = sum_[index]/len(sum_)

            self.counter_+=1
            #print(sum_)

            return sum_

    def choose_toppings(self, preferences):
            """Function in which we choose position of toppings

            Args:
                num_toppings(int) : the total number of different topics chosen among 2, 3 and 4
                preferences(list) : List of size 100*2*num_toppings for 100 generated preference pairs(actual amounts) of customers.

            Returns:
                pizzas(list) : List of size [10,24,3], where 10 is the pizza id, 24 is the topping id, innermost list of size 3 is [x coordinate of topping center, y coordinate of topping center, topping number of topping(1/2/3/4) (Note that it starts from 1, not 0)]
            """
            #uniform = 0
            self.avg_by_ten(preferences)
            #print("PREF:", preferences)

            count=0
            three_topping_tracker = 0
            four_topping_tracker = 0
    
            x_coords = [np.sin(np.pi/2)]
            pizzas = np.zeros((10, 24, 3))
            type_of_topping = 0
            avg_list = []
            #print("this is the num of initial", constants.number_of_initial_pizzas)
            #print("these are the preferences", preferences)
            for j in range(constants.number_of_initial_pizzas):
                if len(preferences)==100:
                    avg_list=self.avg_by_ten(preferences[(j*10):])
                else:
                    avg_list = preferences[j]

                pizza_indiv = np.zeros((24,3))
                i = 0
                first_dist = 0
                second_dist = 0
                while i<24:
                    
                    if self.num_toppings == 2:
                        first_dist = 3
                    elif self.num_toppings == 3:
                        first_dist = 3
                        second_dist = 4.5
                    else:
                        first_dist = 3 
                    #print("This is the distance: ", dist)
                    
                    angle = i/24*2*np.pi
                # print("This is the angle: ", angle)


                    if self.num_toppings == 2: 
                        x = first_dist*np.cos(angle)
                        y = first_dist*np.sin(angle)
                        
                        #print("this is x and y", x , y)
                        if angle < np.pi:
                            type_of_topping = 1
                        else:
                            type_of_topping = 2
                    
                    if self.num_toppings == 3:
                        if i <= 15:
                            angle = i/16*2*np.pi
                            x = first_dist*np.cos(angle)
                            y = first_dist*np.sin(angle)
                            if angle < np.pi:
                                index_ = avg_list.index(self.largest_num(avg_list))+1
                                type_of_topping = index_
                            else:
                                if i == 8:
                                    remove_ = avg_list.index(self.largest_num(avg_list))
                                    avg_list[remove_] = -100
                                index_ = avg_list.index(self.largest_num(avg_list))+1
                                type_of_topping = index_                            
                    # if i < 16:  # Toppings 1 and 2
                    #         angle = 2 * np.pi * i / 16
                    #         x = inner_circle_radius * np.cos(angle)
                    #         y = inner_circle_radius * np.sin(angle)
                    #         topping_type = 1 if i < 8 else 2
                    #     else:  # Topping 3
                    #         angle = 2 * np.pi * (i - 8) / 16 + np.pi/8
                    #         x = outer_circle_radius * np.cos(angle)
                    #         y = outer_circle_radius * np.sin(angle)
                    #         topping_type = 3                    
                    #     #print("this is x and y", x , y)
                        else: 
                            angle = 2 * np.pi * (i - 8) / 16
                            x = second_dist*np.cos(angle)
                            y = second_dist*np.sin(angle)
                            if i == 16:
                                remove_ = avg_list.index(self.largest_num(avg_list))
                                avg_list[remove_] = -100
                            index_ = avg_list.index(self.largest_num(avg_list))+1
                            type_of_topping = index_


                    if self.num_toppings == 4:
                        x = first_dist*np.cos(angle)
                        y = first_dist*np.sin(angle)
                    
                        if angle < (2/4*(np.pi)):
                            index_ = avg_list.index(self.largest_num(avg_list))+1
                            type_of_topping = index_
                        elif angle >= (2/4*(np.pi)) and angle < ((np.pi)):
                            if i == 6:
                                remove_ = avg_list.index(self.largest_num(avg_list))
                                avg_list[remove_] = -100
                            index_ = avg_list.index(self.largest_num(avg_list))+1
                            type_of_topping = index_
                        elif angle >= ((np.pi)) and angle < (6/4*(np.pi)):
                            if i == 12:
                                remove_ = avg_list.index(self.largest_num(avg_list))
                                avg_list[remove_] = -100                        
                            index_ = avg_list.index(self.largest_num(avg_list))+1
                            type_of_topping = index_
                        else: 
                            if i == 18:
                                remove_ = avg_list.index(self.largest_num(avg_list))
                                avg_list[remove_] = -100
                            index_ = avg_list.index(self.largest_num(avg_list))+1
                            type_of_topping = index_
                        #look through the preferences and see which is the most popular, you can do it every 10 pizzas
                        '''
                        i = 0
                        while i<10:
                            do what im doing for the preferences and and build each pizza accordingly 
                            if length is 10 then based on each pizza model toppings off that individual one

                            sort out the placements 
                            need to analyse both pairs see which person prefers what and possibly place based on most liked by both
                            maybe keep in dictionary and if it runs out choose from remaining and brute force
                            
                        '''

            


                        # three_topping_tracker+=1
                        # angle = i/16*2*np.pi
                        # x = first_dist*np.cos(angle)
                        # y = first_dist*np.sin(angle)
                        # print("This is the x, y: ", x, y)

                        # if angle < np.pi and three_topping_tracker<17:
                        #     type_of_topping = 1
                        #     print("Getting in: ", x, y)
                        # elif angle >= np.pi and three_topping_tracker<17:
                        #     type_of_topping = 2
                        #     print("Making the circle ", x, y)
                        # else:
                        #     first_dist = 4.5
                        #     angle = ((i%8)+1)/8*2*np.pi
                        #     # if three_topping_tracker<21:
                        #     print("three_topping_tracker: ", three_topping_tracker)
                        #     x = first_dist*np.cos(angle)
                        #     y = first_dist*np.sin(angle)
                        #     type_of_topping = 3
                        #     # else: 
                            #     x = -first_dist*(((three_topping_tracker%4+1)/2)*np.pi)
                            #     y = -(first_dist*(((three_topping_tracker%4+1)/2)*np.pi))

                        

                    clash_exists = pizza_calculations.clash_exists(x, y, pizza_indiv, i)
                    if not clash_exists:
                        pizza_indiv[i] = [x, y, type_of_topping]
                        i = i+1
                    pizza_indiv = np.array(pizza_indiv)
                    pizzas[j] = pizza_indiv
            # print("These are the pizzas ", list(pizzas))
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
        num_toppings = self.num_toppings

        #print("Customer amount Value: ", customer_amounts)
        customer_1_preference = customer_amounts[0]
        customer_2_preference = customer_amounts[1]

        pizza_id = remaining_pizza_ids[0]
        listOfCenters, listOfAngles  = self.generate_values()

        max_s = -10000
        best_center = None
        best_angle = None

    
        for _center in listOfCenters:
            for angles in listOfAngles:
                cut = [self.x + _center[0]*self.multiplier, self.y - _center[1]*self.multiplier, angles]
                score = self.get_score([pizzas[pizza_id]], [0], [customer_amounts], [cut])


                if score > max_s:
                    max_s = score
                    best_angle = angles
                    best_center = _center 

        # Print the coordinates of the intersections.
        #print("s:", score)
        #print("Best Center" , best_center)
        #print("Best_angle", best_angle)
        #print("max_s:", max_s)

        return pizza_id, best_center, best_angle

    def generate_values(self):
        # Creates a set of coordinates for radius values 1 through 3
        centers = []
        angles = []
        for i in range(1, 100):
            angles.append(i*np.pi/180)
        for R in range(1, 3):
            centers.append([R, 0])
            centers.append([R / 2, -R / 2])
            centers.append([0, -R])
            centers.append([-R / 2, -R / 2])
            centers.append([-R, 0])
            centers.append([-R / 2, R / 2])
            centers.append([0, R])
            centers.append([R / 2, R / 2])
        return centers, angles
    
    def get_score(self, pizzas, ids, preferences, cuts):
        B, C, U, obtained_preferences, center_offsets, slice_amount_metric = self.calculator.final_score(pizzas, ids, preferences, cuts, self.num_toppings, self.multiplier, self.x, self.y)
        usum = self.sum(U[0])
        bsum = self.sum(B[0])
        csum = self.sum(C[0])


        #print("B:", bsum)
        #print("C:", csum)
        return bsum - csum

    def sum(self, array):
        sum = 0
        for a in array:
            for b in a:
                sum += b
        return sum

    def calculate_cut_intersection(self, pizza_radius, angle, center):
        """Calculates the coordinate of the cut intersection.

        Args:
            pizza_radius: The radius of the pizza.
            angle: The angle of the cut in the 0-45 degree range.
            center: The center of the pizza.
        Returns:
            A tuple of (x, y) coordinates of the cut intersection.
        """

        # Convert the angle to radians.
        angle_in_radians = math.radians(angle)

        # Calculate the x and y coordinates of the cut intersection.
        x = pizza_radius * math.cos(angle_in_radians) + center[0]
        y = pizza_radius * math.sin(angle_in_radians) + center[1]

        return round(x, 2), round(y, 2)