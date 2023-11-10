from tokenize import String
import numpy as np
from typing import Tuple, List
import constants
from utils import pizza_calculations
from itertools import chain

# granularity of brute force cut
N = 24

class Player:
    def __init__(self, num_toppings, rng: np.random.Generator) -> None:
        """Initialise the player"""
        self.rng = rng
        self.num_toppings = num_toppings
        self.multiplier = 40
        self.x_center = 12*self.multiplier	# Center Point x of pizza
        self.y_center = 10*self.multiplier	# Center Point y of pizza
        #self._build_lut()
        self.lut = np.load(f'lut{self.num_toppings}.npy') if self.num_toppings > 2 else None
       
    def customer_gen_team4(self, num_cust, rng=None, alpha=2, beta=2):
        """
        Function to create non-uniform customer preferences using a beta distribution.

        Args:
            num_cust (int): The total number of customer preferences to create.
            rng (int): A random seed for generating customers. If None, self.rng will be used.
            alpha (float): Alpha parameter for the beta distribution.
            beta (float): Beta parameter for the beta distribution.

        Returns:
            preferences_total (list): List of size [num_cust, 2, num_toppings], containing generated customer preferences.
        """

        preferences_total = []

        if rng is None:
            for i in range(num_cust):
                preferences_1 = self.rng.beta(alpha, beta, self.num_toppings)
                preferences_1 = 12*preferences_1/np.sum(preferences_1)
                preferences_2 = self.rng.beta(alpha, beta, self.num_toppings)
                preferences_2 = 12*preferences_2/np.sum(preferences_2)
                preferences = [preferences_1, preferences_2]
                preferences_total.append(preferences)
        else:
            for i in range(num_cust):
                preferences_1 = rng.beta(alpha, beta, self.num_toppings)
                preferences_1 = 12*preferences_1/np.sum(preferences_1)
                preferences_2 = rng.beta(alpha, beta, self.num_toppings)
                preferences_2 = 12*preferences_2/np.sum(preferences_2)
                preferences = [preferences_1, preferences_2]
                preferences_total.append(preferences)

        return preferences_total

    def customer_gen(self, num_cust, rng=None):
        """
        Function to create non-uniform customer preferences using a beta distribution.

        Args:
            num_cust (int): The total number of customer preferences to create.
            rng (int): A random seed for generating customers. If None, self.rng will be used.
            alpha (float): Alpha parameter for the beta distribution.
            beta (float): Beta parameter for the beta distribution.

        Returns:
            preferences_total (list): List of size [num_cust, 2, num_toppings], containing generated customer preferences.
        """

        preferences_total = []
        for i in range(num_cust):
            preferences_1 = np.ones((self.num_toppings,))*12/self.num_toppings
            preferences_2 = np.ones((self.num_toppings,))*12/self.num_toppings
            likes_1 = rng.choice(range(self.num_toppings), 2, replace=False)
            likes_2 = rng.choice(range(self.num_toppings), 2, replace=False)
            preferences_1[likes_1[0]] = preferences_1[likes_1[0]] + 6/self.num_toppings
            preferences_1[likes_1[1]] = preferences_1[likes_1[1]] - 6/self.num_toppings
            preferences_2[likes_2[0]] = preferences_2[likes_2[0]] + 6/self.num_toppings
            preferences_2[likes_2[1]] = preferences_2[likes_2[1]] - 6/self.num_toppings
            preferences = [preferences_1, preferences_2]
            preferences_total.append(preferences) 
        return preferences_total

    def _toppings_2(self):
        pizzas = np.zeros((10, 24, 3))

        for j in range(constants.number_of_initial_pizzas):
                pizza_indiv = np.zeros((24,3))
                for i in range(24):
                    angle = (i * np.pi / 12) + np.pi / 24
                    dist = 3.0
                    x = dist * np.cos(angle)
                    y = dist * np.sin(angle)
                    pizza_indiv[i] = [x, y, (i/12) + 1]
                pizza_indiv = np.array(pizza_indiv)
                pizzas[j] = pizza_indiv

        return list(pizzas)

    def _toppings_3(self, prefs):
        pizzas = np.zeros((10, 24, 3))

        for j in range(constants.number_of_initial_pizzas):
                pizza_indiv = np.zeros((24,3))
                for i in range(16):
                    angle = (i * np.pi / 8) + np.pi / 16
                    dist = 2.
                    x = dist * np.cos(angle)
                    y = dist * np.sin(angle)
                    pizza_indiv[i] = [x, y, (i/8) + 1]
                dist = 3.5
                angle = np.pi / 8
                pizza_indiv[16] = [dist, 0, 3]
                pizza_indiv[17] = [dist * np.cos(angle), dist * np.sin(angle), 3]
                pizza_indiv[18] = [dist * np.cos(angle), dist * np.sin(-angle), 3]

                pizza_indiv[19] = [0, dist, 3]

                pizza_indiv[20] = [dist * np.cos(np.pi - angle), dist * np.sin(np.pi - angle), 3]
                pizza_indiv[21] = [dist * np.cos(np.pi + angle), dist * np.sin(np.pi + angle), 3]

                pizza_indiv[22] = [dist * np.cos(3 * np.pi / 2 - angle), dist * np.sin(3 * np.pi / 2 - angle), 3]
                pizza_indiv[23] = [dist * np.cos(3 * np.pi / 2 + angle), dist * np.sin(3 * np.pi / 2 + angle), 3]
                pizza_indiv = np.array(pizza_indiv)
                pizzas[j] = pizza_indiv

        return list(pizzas)


    def _toppings_4(self, prefs):
        #arr = np.stack(prefs)
        #corr = np.corrcoef(arr, rowvar=False)
        #idx = np.argsort(corr)

        pizzas = np.zeros((10, 24, 3))
        for j in range(constants.number_of_initial_pizzas):
                pizza_indiv = np.zeros((24,3))
                for i in range(12):
                    angle = (i * np.pi / 6) + np.pi / 12
                    dist = 1.5
                    x = dist * np.cos(angle)
                    y = dist * np.sin(angle)
                    pizza_indiv[i] = [x, y, (i/6) + 1]
                for i in range(6):
                    angle = (i * np.pi / 3) + np.pi / 6
                    dist = 2.5
                    x = dist * np.cos(angle)
                    y = dist * np.sin(angle)
                    pizza_indiv[12 + i] = [x, y, 3]

                dist = 3.5
                angle = np.pi / 8

                pizza_indiv[18] = [dist, 0, 4]

                pizza_indiv[19] = [0, dist, 4]

                pizza_indiv[20] = [dist * np.cos(np.pi - angle), dist * np.sin(np.pi - angle), 4]
                pizza_indiv[21] = [dist * np.cos(np.pi + angle), dist * np.sin(np.pi + angle), 4]

                pizza_indiv[22] = [dist * np.cos(3 * np.pi / 2 - angle), dist * np.sin(3 * np.pi / 2 - angle), 4]
                pizza_indiv[23] = [dist * np.cos(3 * np.pi / 2 + angle), dist * np.sin(3 * np.pi / 2 + angle), 4]
                pizza_indiv = np.array(pizza_indiv)
                pizzas[j] = pizza_indiv
        """
        pizzas = np.zeros((10, 24, 3))
        for j in range(constants.number_of_initial_pizzas):
                pizza_indiv = np.zeros((24,3))
                for i in range(self.num_toppings):
                    center_d = 2.
                    theta_d = i * np.pi / 2
                    
                    for k in range(6):
                        angle = k * np.pi / 3
                        dist = 1.
                        x = center_d * np.cos(theta_d) + dist * np.cos(angle)
                        y = center_d * np.sin(theta_d) + dist * np.sin(angle)
                        pizza_indiv[6 * i + k] = [x, y, i+1]
                pizza_indiv = np.array(pizza_indiv)
                pizzas[j] = pizza_indiv
        """

        return list(pizzas)
        

    def _opt_ratio(self, amounts):
        p, q, n = amounts[0], amounts[1], self.num_toppings
        return np.array([0.5 * (p[i] - q[i]) + 12 / n for i in range(n)]) 

    def choose_toppings(self, preferences):
        """Function in which we choose position of toppings

        Args:
            num_toppings(int) : the total number of different topics chosen among 2, 3 and 4
            preferences(list) : List of size 100*2*num_toppings for 100 generated preference pairs(actual amounts) of customers.

        Returns:
            pizzas(list) : List of size [10,24,3], where 10 is the pizza id, 24 is the topping id, innermost list of size 3 is [x coordinate of topping center, y coordinate of topping center, topping number of topping(1/2/3/4) (Note that it starts from 1, not 0)]
        """

        if self.num_toppings == 2:
            return self._toppings_2()

        pref_opt = [self._opt_ratio(p) for p in preferences]
        
        if self.num_toppings == 3:
            return self._toppings_3(pref_opt)
        else:
            return self._toppings_4(pref_opt)

    def _cut2(self, pizzas, remaining_pizza_ids, customer_amounts):
        pref = self._opt_ratio(customer_amounts)
        angle = pref[0] / 12 * np.pi
        dist = 4.75
        x = dist*np.cos(np.pi + angle)
        y = dist*np.sin(np.pi + angle)

        return remaining_pizza_ids[0], [x, y], angle

    def choose_and_cut(self, pizzas, remaining_pizza_ids, customer_amounts):
        """Function which based n current game state returns the distance and angle, the shot must be played

        Args:
            pizzas (list): List of size [10,24,3], where 10 is the pizza id, 24 is the topping id, innermost list of size 3 is [x coordinate of topping, y coordinate of topping, topping number of topping(1/2/3/4)]
            remaining_pizza_ids (list): A list of remaining pizza's ids
            customer_amounts (list): The amounts in which the customer wants their pizza

        Returns:
            Tuple[int, center, first cut angle]: Return the pizza id you choose, the center of the cut in format [x_coord, y_coord] where both are in inches relative of pizza center of radius 6, the angle of the first cut in radians. 
        """
        if self.num_toppings == 2:
            return self._cut2(pizzas, remaining_pizza_ids, customer_amounts)
        best_S = -np.inf
        best_cut = None
        delta = 6 / N
        theta = (2 * np.pi) / N
        pizza_id = remaining_pizza_ids[0]

        for i in range(N):
            d = delta * i
            for j in range(N):
                angle = theta * j
                x = d * np.cos(angle)
                y = d * np.sin(angle)

                for k in range(N):
                    rotation = (np.pi * k) / (2 * N)
                    cut = [x, y, rotation]

                    x_mod = (self.x_center + x * self.multiplier)
                    y_mod = (self.y_center - y * self.multiplier)
                    cut_mod = [x_mod, y_mod, rotation]
                    #obtained_pref = np.array(pizza_calculations().ratio_calculator(
                    #                pizzas[pizza_id], 
                    #                cut_mod, 
                    #                self.num_toppings, 
                    #                self.multiplier, 
                    #                self.x_center, 
                    #                self.y_center)[0])
                    obtained_pref = self.lut[i][j][k]
                    rand_cut = [self.x_center, self.y_center, self.rng.random()*2*np.pi]
                    required_pref = np.array(customer_amounts)
                    uniform_pref = np.ones((2, self.num_toppings))*(12/self.num_toppings)

                    b = np.round(np.absolute(required_pref - uniform_pref), 3)
                    c = np.round(np.absolute(obtained_pref - required_pref), 3)

                    s = (b-c).sum()
                    if s > best_S:
                        best_S = s
                        best_cut = cut

        return remaining_pizza_ids[0], [best_cut[0], best_cut[1]], best_cut[2]

    '''
    def _build_lut(self):
        delta = 6 / N
        theta = (2 * np.pi) / N

        for t in (3, 4):
            pizza = self._toppings_3([])[0] if t == 3 else self._toppings_4([])[0]

            lut = np.zeros((N, N, N, 2, t))
            for i in range(N):
                d = delta * i
                for j in range(N):
                    angle = theta * j
                    x = d * np.cos(angle)
                    y = d * np.sin(angle)

                    for k in range(N):
                        rotation = (np.pi * k) / (2 * N)
                        cut = [x, y, rotation]

                        x_mod = (self.x_center + x * self.multiplier)
                        y_mod = (self.y_center - y * self.multiplier)
                        cut_mod = [x_mod, y_mod, rotation]

                        obtained_pref = np.array(pizza_calculations().ratio_calculator(
                                            pizza, 
                                            cut_mod, 
                                            t, 
                                            self.multiplier, 
                                            self.x_center, 
                                            self.y_center)[0])
                        lut[i][j][k] = obtained_pref
                        print(i, j, k)
                        print(obtained_pref)
            np.save(f'lut{t}.npy', lut)
            print(f'{t} done')
    '''
