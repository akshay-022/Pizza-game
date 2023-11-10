# Mathematical and Geometric Calculations
import math  # For mathematical operations like trigonometric functions
import numpy as np  # For numerical operations, array manipulations
import pandas as pd
import json

# Additional Utilities
from collections import defaultdict  # For easier handling of data structures
from typing import List, Tuple, Dict  # For type annotations.
from tokenize import String
import constants
from utils import pizza_calculations
from ast import literal_eval
from shapely.geometry import LineString, Point
import copy


class Player:
    def __init__(self, num_toppings, rng):
        self.num_toppings = num_toppings
        self.rng = rng
        self.multiplier = 40
        self.pizza_radius = 6 * self.multiplier
        self.topping_radius = 0.375
        self.pizza_center = [12 * self.multiplier, 10 * self.multiplier]
        self.calculations = pizza_calculations()

    def customer_gen(self, num_cust, rng=None):
        def create_inst():
            # Generate preferences using the beta distribution
            alpha, beta = 2, 2  # You can adjust these parameters as needed
            p = np.random.beta(alpha, beta, self.num_toppings)

            # Scale and normalize preferences to sum to 12, and clamp between 0.1 and 11.9
            p = 11.8 * p / np.sum(p) + 0.1
            return p

        preferences_total = []
        rng = rng if rng is not None else self.rng

        for i in range(num_cust):
            preferences_1 = create_inst()
            preferences_2 = create_inst()

            preferences = [preferences_1, preferences_2]
            equal_prob = rng.random()
            if equal_prob <= 0.0:  # Change this if you want toppings to show up
                preferences = (np.ones((2, self.num_toppings))
                               * 12 / self.num_toppings).tolist()

            preferences_total.append(preferences)

        return preferences_total

    def choose_two(self):
        pizzas = np.zeros((10, 24, 3))

        pizza_radius = 3

        for j in range(constants.number_of_initial_pizzas):  # Iterate over each pizza
            pizza_indiv = np.zeros((24, 3))

            ct = 1
            for i in range(24):  # Place 24 toppings on each pizza
                place = True
                angle_increment = 2 * np.pi / 24
                angle = i * angle_increment

                # Calculate x, y coordinates
                x = pizza_radius * np.cos(angle)
                y = pizza_radius * np.sin(angle)

                # Assign topping type based on the number of toppings
                topping_type = 1 if i < 12 else 2

                # if place:
                pizza_indiv[i] = [x, y, topping_type]

            pizzas[j] = pizza_indiv

        return list(pizzas)

    def choose_three(self):
        pizzas = np.zeros((10, 24, 3))

        pizza_radius = 3
        for j in range(constants.number_of_initial_pizzas):  # Iterate over each pizza
            pizza_indiv = np.zeros((24, 3))

            ct = 1
            ends = []
            prev = 0
            for i in range(24):  # Place 24 toppings on each pizza
                angle_increment = 2 * np.pi / 18
                angle = i * angle_increment

                # Calculate x, y coordinates
                x = pizza_radius * np.cos(angle)
                y = pizza_radius * np.sin(angle)

                if j < 3:
                    a, b, c = 1, 2, 3
                elif j < 6:
                    a, b, c = 2, 3, 1
                else:
                    a, b, c = 3, 1, 2

                if i == 0 or i == 9:
                    topping_type = c
                    ends.append(x)
                elif i <= 8:
                    topping_type = a
                elif i <= 17:
                    topping_type = b
                else:
                    topping_type = c
                    y = 0
                    if ct == 1:
                        x = ends[1] - ((pizza_radius)/5 + 0.55)
                    elif ct == 6:
                        x = ends[0] + ((pizza_radius)/5 + 0.55)
                    else:
                        x = ends[1] + \
                            ((pizza_radius - 0.75)/3.5 + 0.55) * (ct - 1)
                    ct += 1
                pizza_indiv[i] = [x, y, topping_type]

            pizzas[j] = pizza_indiv

        return list(pizzas)

    def choose_four(self):
        pizzas = np.zeros((10, 24, 3))
        buff = 0.001

        for j in range(constants.number_of_initial_pizzas):

            if j < 2:
                inner_toppings = [1] * 6 + [2] * 6
                outer_toppings = [3] * 6 + [4] * 6
            elif j < 4:
                inner_toppings = [4] * 6 + [3] * 6
                outer_toppings = [2] * 6 + [1] * 6
            elif j < 6:
                inner_toppings = [3] * 6 + [1] * 6
                outer_toppings = [4] * 6 + [2] * 6
            elif j < 8:
                inner_toppings = [2] * 6 + [3] * 6
                outer_toppings = [1] * 6 + [4] * 6
            else:
                inner_toppings = [4] * 6 + [2] * 6
                outer_toppings = [3] * 6 + [1] * 6

            inner_radius = buff + 0.189 / np.sin(np.pi / 24)
            outer_radius = buff + 0.375 / np.sin(np.pi / 24)

            theta = np.pi / 12
            outer_angle = 11 * np.pi / 6

            inner = [
                [
                    inner_radius * np.cos((2 * i + 1) * theta),
                    inner_radius * np.sin((2 * i + 1) * theta),
                    inner_toppings[i]
                ]
                for i in range(12)
            ]
            outer = [
                [
                    outer_radius * np.cos((outer_angle + (2 * i + 1)) * theta),
                    outer_radius * np.sin((outer_angle + (2 * i + 1)) * theta),
                    outer_toppings[i]
                ]
                for i in range(12)
            ]
            pizza = inner + outer
            pizzas[j] = pizza

        return list(pizzas)

    def choose_toppings(self, preferences):
        # 10 pizzas, 24 toppings each, 3 values per topping (x, y, type)
        if self.num_toppings == 2:
            return self.choose_two()
        elif self.num_toppings == 3:
            return self.choose_three()
        else:
            return self.choose_four()

    def precompute_preferences(self, pizzas, num_toppings, multiplier, pizza_center):
        data = []
        xCenter, yCenter = pizza_center
        file_name = f'{num_toppings}.csv'

        for pizza_id, pizza in enumerate(pizzas):
            for radius in range(0, 6):
                for x in np.linspace(-radius, radius, 24):
                    for ySign in range(-1, 2, 2):
                        y = ySign * (math.sqrt((radius ** 2) - (x ** 2)))
                        for angle in np.arange(0, 2 * np.pi, np.radians(5)):
                            xCord = (xCenter + x * multiplier)
                            yCord = (yCenter - y * multiplier)
                            obtained_pref, _ = self.calculations.ratio_calculator(pizza, [xCord, yCord, angle],
                                                                                  num_toppings, multiplier,
                                                                                  xCenter, yCenter)
                            obtained_pref_str = json.dumps(obtained_pref.tolist())  # Convert numpy array to string
                            data.append([pizza_id, x, y, angle, obtained_pref_str])

        df = pd.DataFrame(data, columns=['pizza_id', 'x', 'y', 'angle', 'obtained_pref'])
        df.to_csv(file_name, index=False)


    def choose_and_cut(self, pizzas, remaining_pizza_ids, customer_amounts):
        #self.precompute_preferences(pizzas, self.num_toppings, self.multiplier, self.pizza_center)
        file_name = f'{self.num_toppings}.csv'
        df = pd.read_csv(file_name)
        maximumS = -1000
        maximumCut = None

        for pizza_id in remaining_pizza_ids:
            pizza_df = df[df['pizza_id'] == pizza_id]
            for index, row in pizza_df.iterrows():
                obtained_pref_str = row['obtained_pref']
                obtained_pref = np.array(json.loads(obtained_pref_str))
                required_pref = np.array(customer_amounts)
                uniform_pref = np.ones((2, self.num_toppings)) * (12 / self.num_toppings)
                b = np.round(np.absolute(required_pref - uniform_pref), 3)
                c = np.round(np.absolute(obtained_pref - required_pref), 3)
                s = (b - c).sum()
                if s > maximumS:
                    maximumS = s
                    maximumCut = [row['x'], row['y'], row['angle'], pizza_id]

        x, y, theta, pizza_id = maximumCut
        #print(f"Chosen cut: x={x}, y={y}, theta={theta}, pizza_id={pizza_id}")
        return pizza_id, [x, y], theta
