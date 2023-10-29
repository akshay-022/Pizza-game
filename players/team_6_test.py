# Standard Library Imports
import random  # For generating random numbers and choices

# Mathematical and Geometric Calculations
import math  # For mathematical operations like trigonometric functions
import numpy as np  # For numerical operations, array manipulations

# Statistical Analysis
from scipy import stats  # For statistical analysis and pattern recognition

# Optimization Algorithms
from scipy.optimize import minimize  # For optimizing pizza selection and cutting

# Additional Utilities
from collections import defaultdict  # For easier handling of data structures
from typing import List, Tuple, Dict  # For type annotations
from tokenize import String
import constants


class Player:
    def __init__(self, num_toppings, rng=None):
        """
        Initialize the player with the number of toppings and a random number generator.

        Args:
        num_toppings (int): Number of different toppings.
        rng (np.random.Generator): Random number generator instance.
        """
        self.num_toppings = num_toppings
        self.rng = rng
        self.pizzas = []
        self.preference_analysis = []

    def customer_gen(self, num_cust, rng=None):
        """Function in which we create a distribution of customer preferences

        Args:
            num_cust(int) : the total number of customer preferences you need to create
            rng(int) : A random seed that you can use to generate your customers. You can choose to not pass this, in that case the seed taken will be self.rng

        Returns:
            preferences_total(list) : List of size [num_cust, 2, num_toppings], having all generated customer preferences
        """

        # https://www.statisticshowto.com/beta-distribution/
        alpha = 2.0  # test & adjust as neededs
        beta = 2.0  # test & adjust as needed

        preferences_total = []
        if rng == None:
            np.random.seed(self.rng)
            print("beta distribution")
            for i in range(num_cust):
                preferences_1 = np.random.beta(alpha, beta, self.num_toppings)
                # ensure non-neg values
                preferences_1 = np.clip(preferences_1, 0, None)
                preferences_1 /= preferences_1.sum()  # normalize
                preferences_total.append(
                    [preferences_1.tolist(), preferences_1.tolist()])  # duplicate
        else:
            for i in range(num_cust):
                preferences_1 = rng.random((self.num_toppings,))
                preferences_1 = 12 * preferences_1 / np.sum(preferences_1)
                preferences_2 = rng.random((self.num_toppings,))
                preferences_2 = 12 * preferences_2 / np.sum(preferences_2)
                preferences = [preferences_1, preferences_2]
                equal_prob = rng.random()
                if equal_prob <= 0.0:  # change this if you want toppings to show up
                    preferences = (np.ones((2, self.num_toppings))
                                   * 12 / self.num_toppings).tolist()
                preferences_total.append(preferences)

        return preferences_total

    def choose_toppings(self, preferences):
        """
        Choose the position of toppings based on market research.

        Args:
        preferences (list): List of size 100*2*num_toppings for generated preference pairs.

        Returns:
        list: List of pizzas in the format [10, 24, 3].
        """
        # Analyze common patterns in customer preferences
        self.preference_analysis = analyze_preferences(preferences)

        pizzas = []
        for _ in range(constants.number_of_initial_pizzas):
            toppings = place_toppings_optimally(
                self.preference_analysis, self.num_toppings)
            pizza_array = np.zeros((24, 3))

            for i, (x, y, topping_type) in enumerate(toppings):
                # Adjust topping_type to be 1-indexed
                pizza_array[i] = [x, y, topping_type + 1]

            pizzas.append(pizza_array)

        return pizzas

    def choose_and_cut(self, pizzas, remaining_pizza_ids, customer_amounts):
        """
        Select the best pizza and calculate the optimal cut based on the current game state.

        Args:
        pizzas (list): List of pizzas in the format [10, 24, 3].
        remaining_pizza_ids (list): List of remaining pizza IDs.
        customer_amounts (list): The amounts in which the customer wants their pizza.

        Returns:
        Tuple[int, list, float]: Selected pizza ID, cut position, and cut angle.
        """
        best_score = float('inf')
        best_pizza_id = -1
        best_cut_position = (0, 0)
        best_cut_angle = 0

        for pizza_id in remaining_pizza_ids:
            # Convert pizza format to the one expected by optimize_pizza_selection_and_cut
            pizza = [(x, y, topping_type - 1)
                     for x, y, topping_type in pizzas[pizza_id]]

            # Find the best cut for this pizza
            cut_position, cut_angle, score = find_best_cut_for_pizza(
                pizza, customer_amounts)

            # Check if this pizza and cut is better than the current best
            if score < best_score:
                best_score = score
                best_pizza_id = pizza_id
                best_cut_position = cut_position
                best_cut_angle = cut_angle

        return best_pizza_id, list(best_cut_position), best_cut_angle

# Analyze customer preferences to identify common patterns


def analyze_preferences(preferences):
    """
    Analyze customer preferences to identify common patterns.

    Args:
    preferences (List[List[float]]): List of customer preferences.

    Returns:
    Dict[str, float]: A dictionary containing analysis results,
                       like most common ratios.
    """
    # Convert to numpy array for easier manipulation
    preferences_array = np.array(preferences)

    # Initialize dictionary to store analysis results
    analysis_results = {
        'mean_preferences': np.mean(preferences_array, axis=0),
        'std_preferences': np.std(preferences_array, axis=0)
    }

    return analysis_results


def clash_exists(x, y, existing_toppings, topping_radius):
    """
    Check if a new topping placement clashes with existing toppings.

    Args:
    x (float): X-coordinate of the new topping.
    y (float): Y-coordinate of the new topping.
    existing_toppings (List[Tuple[float, float, int]]): List of existing toppings.
    topping_radius (float): Radius of the toppings.

    Returns:
    bool: True if clash exists, False otherwise.
    """
    for existing_x, existing_y, _ in existing_toppings:
        if math.sqrt((existing_x - x)**2 + (existing_y - y)**2) < 2 * topping_radius:
            return True
    return False

# Place toppings on a pizza considering the common preference patterns


def place_toppings_optimally(preference_analysis, num_toppings):
    """
    Place toppings on a pizza considering the common preference patterns.

    Args:
    preference_analysis (Dict[str, float]): Analysis results of preferences.
    num_toppings (int): Number of topping types.

    Returns:
    List[Tuple[float, float, int]]: List of toppings with their positions and types.
    """
    toppings = []
    pizza_radius = 6  # Radius of the pizza
    topping_radius = 0.375  # Radius of each topping

    # Algorithm to distribute toppings on pizza
    for i in range(24):  # Assuming 24 toppings per pizza
        # Place toppings randomly within pizza bounds
        while True:
            x = random.uniform(-pizza_radius, pizza_radius)
            y = random.uniform(-pizza_radius, pizza_radius)
            if x**2 + y**2 <= (pizza_radius - topping_radius)**2:
                # Check for overlaps with existing toppings
                if not clash_exists(x, y, toppings, topping_radius):
                    topping_type = i % num_toppings  # Assign type based on index
                    toppings.append((x, y, topping_type))
                    break

    return toppings

# Select the best pizza for the customer and determine the optimal cut


def optimize_pizza_selection_and_cut(pizzas, customer_preferences):
    """
    Select the best pizza for the customer and determine the optimal cut.

    Args:
    pizzas (List[List[Tuple[float, float, int]]]): List of available pizzas with toppings.
    customer_preferences (List[Tuple[float]]): Preferences of the customer.

    Returns:
    Tuple[int, float, float]: Selected pizza index, cut position, and cut angle.
    """
    best_score = float('inf')
    best_pizza = -1
    best_cut_position = (0, 0)
    best_cut_angle = 0

    for i, pizza in enumerate(pizzas):
        # Find the best cut for this pizza
        cut_position, cut_angle, score = find_best_cut_for_pizza(
            pizza, customer_preferences)

        # Check if this pizza and cut is better than the current best
        if score < best_score:
            best_score = score
            best_pizza = i
            best_cut_position = cut_position
            best_cut_angle = cut_angle

    return best_pizza, best_cut_position, best_cut_angle

# Geometric calculations for determining the best cut


def find_best_cut_for_pizza(pizza, preferences):
    """
    Find the best cut for a given pizza based on customer preferences.

    Args:
    pizza (List[Tuple[float, float, int]]): A pizza with toppings.
    preferences (List[Tuple[float]]): Preferences of the customer.

    Returns:
    Tuple[float, float, float]: Best cut position, angle, and score.
    """
    best_score = float('inf')
    best_cut_position = (0, 0)
    best_cut_angle = 0

    # Discretize the search space
    for x in np.linspace(-6, 6, num=20):  # Assuming the pizza diameter is 12
        for y in np.linspace(-6, 6, num=20):
            # Trying different angles
            for angle in np.linspace(0, 2 * math.pi, num=36):
                # Calculate the score for this cut
                score = calculate_cut_score(pizza, (x, y), angle, preferences)

                # Update the best cut if this is better
                if score < best_score:
                    best_score = score
                    best_cut_position = (x, y)
                    best_cut_angle = angle

    return best_cut_position, best_cut_angle, best_score


def calculate_cut_score(pizza, cut_position, cut_angle, preferences):
    """
    Calculate the score for a given cut based on how well it matches the preferences.

    Args:
    pizza (List[Tuple[float, float, int]]): A pizza with toppings.
    cut_position (Tuple[float, float]): Position of the cut.
    cut_angle (float): Angle of the cut.
    preferences (List[Tuple[float]]): Preferences of the customer.

    Returns:
    float: Score representing how well the cut matches the preferences.
    """
    # Initialize distribution of toppings in each slice
    slice_toppings = [{i: 0 for i in range(
        len(preferences[0]))} for _ in range(8)]

    # Loop through each topping to determine its slice
    for x, y, topping_type in pizza:
        slice_index = determine_slice_index(x, y, cut_position, cut_angle)
        slice_toppings[slice_index][topping_type] += 1

    # Compare the distribution with the preferences to calculate the score
    score = 0
    for slice_index, toppings in enumerate(slice_toppings):
        # Alternating preferences
        expected_distribution = preferences[slice_index % 2]
        for topping_type, amount in toppings.items():
            preferred_amount = expected_distribution[topping_type]
            score += abs(amount - preferred_amount)

    return score


def determine_slice_index(x, y, cut_position, cut_angle):
    """
    Determine the index of the slice in which a topping falls.

    Args:
    x (float): X-coordinate of the topping.
    y (float): Y-coordinate of the topping.
    cut_position (Tuple[float, float]): Position of the cut.
    cut_angle (float): Angle of the cut.

    Returns:
    int: Index of the slice.
    """
    # Calculate the angle of the topping relative to the cut position
    angle = math.atan2(y - cut_position[1], x - cut_position[0]) - cut_angle

    # Normalize the angle to be between 0 and 2π
    angle = angle % (2 * math.pi)

    # Determine the slice index based on the angle
    # There are 8 slices, each covering an angle of π/4
    slice_index = int(angle // (math.pi / 4))

    return slice_index

# Optimization technique for selecting the best pizza


def select_best_pizza(pizzas, customer_preferences):
    """
    Select the best pizza based on the customer's preferences.

    Args:
    pizzas (List[List[Tuple[float, float, int]]]): List of available pizzas with toppings.
    customer_preferences (List[Tuple[float]]): Preferences of the customer.

    Returns:
    Tuple[int, Tuple[float, float], float]: Index of the selected pizza, cut position, and cut angle.
    """
    selected_pizza, cut_position, cut_angle = optimize_pizza_selection_and_cut(
        pizzas, customer_preferences)
    return selected_pizza, cut_position, cut_angle
