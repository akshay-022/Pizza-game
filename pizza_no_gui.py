import numpy as np
import constants as constants
from functools import reduce
import time
import copy
import argparse
from utils import pizza_calculations
from players.default_player import Player as default_player
from players.default_player import Player as p1
from players.default_player import Player as p2
from players.default_player import Player as p3
from players.default_player import Player as p4
from players.default_player import Player as p5
from players.default_player import Player as p6

class no_gui():

    def __init__(self, args):
        self.use_gui = True
        self.rng = np.random.default_rng(int(args.seed))
        self.no_of_constraints = constants.number_of_constraints_pp
        
        self.multiplier=40	# Pizza radius = 6*multiplier units
        self.x = 12*self.multiplier	# Center Point x of pizza
        self.y = 10*self.multiplier	# Center Point y of pizza
        self.xs = [30, 60, 90, 120, 150, 30, 60, 90, 120, 150]
        self.ys = [30, 30, 30, 30, 30, 60, 60, 60, 60, 60]

        self.pizzas = np.zeros((10, 24, 3))   #10 pizzas, 27 toppings, each has [x, y, topping type] (x and y wrt pizza center)
        self.pizza_id = 0
        self.cuts = list(np.zeros((10,3)))  #10 pizzas, x, y coordinates of cut intscn wrt pizza center and angle of cut in the 0-45 degree range
        self.type_p = ""
        self.pizzas_drawn = 0
        self.options_player = ["custom_player","default_player", "p1", "p2", "p3","p4","p5", "p6"]
        self.options_topping = [2,3,4]
        self.topping_id = 0 #Current topping id
        self.template_adjustor = 2*self.multiplier #if this goes badly set it to 20     #Just to arrange the difference in sizes of both display screens of 1 and 10 pizzas
        #self.pizza_cuts = list(np.zeros((constants.number_of_initial_pizzas ,3)))           #Here cuts positions are absolute and NOT WRT pizza centers, as it is much easier to deal with them that way. It we are not using gui, we assume self.x, self.y =0 and self.multiplier = 1
        self.customer_id = 0
        self.player_instance = None
        #Inititalise preferences in the begining only
        self.click_indic = 0
        self.final_preferences = []
        self.pizza_choice_order = []
        self.calculator = pizza_calculations()

        #replace with arguments
        self.autoplayer_number = args.autoplayer_number
        self.generator_number = args.generator_number
        self.multiplier = args.interface_size #(default 40)
        self.rng_generator_100 = np.random.default_rng(int(args.gen_100_seed))
        self.rng_generator_10 = np.random.default_rng(int(args.gen_10_seed))
        self.player_nogui = args.player
        self.num_toppings_nogui = args.num_toppings

    def initialise_player(self, player_px, autoplayer) :
            #setting player
            if player_px == 0:
                self.player_instance = default_player(self.num_toppings, self.rng)
            elif player_px == 1:
                self.player_instance = p1(self.num_toppings, self.rng)
            elif player_px == 2:
                self.player_instance = p2(self.num_toppings, self.rng)
            elif player_px == 3:
                self.player_instance = p3(self.num_toppings, self.rng)
            elif player_px == 4:
                self.player_instance = p4(self.num_toppings, self.rng)
            elif player_px == 5:
                self.player_instance = p5(self.num_toppings, self.rng)
            elif player_px == 6:
                self.player_instance = p6(self.num_toppings, self.rng)
            else:
                self.player_instance = default_player(self.num_toppings, self.rng)
            #setting autoplayer
            if autoplayer == 0:
                self.auto_player = default_player(self.num_toppings, self.rng)
            elif autoplayer == 1:
                self.auto_player = p1(self.num_toppings, self.rng)
            elif autoplayer == 2:
                self.auto_player = p2(self.num_toppings, self.rng)
            elif autoplayer == 3:
                self.auto_player = p3(self.num_toppings, self.rng)
            elif autoplayer == 4:
                self.auto_player = p4(self.num_toppings, self.rng)
            elif autoplayer == 5:
                self.auto_player = p5(self.num_toppings, self.rng)
            elif autoplayer == 6:
                self.auto_player = p6(self.num_toppings, self.rng)
            else:
                self.auto_player = default_player(self.num_toppings, self.rng)

    def see_score(self):
        B, C, U, obtained_preferences = self.calculator.final_score(self.pizzas, self.pizza_choice_order, self.preferences, self.cuts, self.num_toppings, self.multiplier, self.x, self.y)
        list_scores = [('Customer Number', "Pizza Number", "U", "B", "C", "S (Score)")]
        with open("summary_log_nogui.txt", "w") as f:
            for i in range(len(self.pizzas)):
                pizza_id = self.pizza_choice_order[i]
                f.write('\n')
                f.write('\n')
                f.write("Customer " + str(i+1) + " :" )
                f.write('\n')
                f.write("Pizza " + str(pizza_id))
                f.write('\n')
                f.write("Cut at (" + str((self.cuts[pizza_id][0]-self.x)/self.multiplier) + ", " + str((self.cuts[pizza_id][1]-self.y)/self.multiplier) + ") and angle " + str(self.cuts[pizza_id][2]) + "radians.")
                f.write('\n')
                for j in range(24):
                    f.write("Topping " + str(self.pizzas[pizza_id][j][2]) + " at (" + str(self.pizzas[pizza_id][j][0]) + "," + str(self.pizzas[pizza_id][j][1]) + ")")
                    f.write('\n')
                f.write("U : " + str(U[i]))
                f.write('\n')
                f.write("B : " + str(B[i]))
                f.write('\n')
                f.write("C : " + str(C[i]))
                f.write('\n')
                f.write("S : " + str(B[i] - C[i]))
                f.write('\n')
                f.write("Desired Preferences : " + str(self.preferences[i]))
                f.write('\n')
                f.write("Obtained Preferences : " + str(obtained_preferences[i]))
                f.write('\n')
                f.write('\n')
                f.write('\n')
                print((str(i+1), str(pizza_id), str(U[i]), str(B[i]), str(C[i]), str(B[i] - C[i])))
            f.write("Total Score U : " + str(np.array(U).sum()))
            f.write('\n')
            f.write("Total score B : " + str(np.array(B).sum()))
            f.write('\n')
            f.write("Total score C : " + str(np.array(C).sum()))
            f.write('\n')
            f.write("Total score S : " + str((np.array(B)-np.array(C)).sum()))
            f.write('\n')
            f.write("Total score S : " + str((np.array(B)-np.array(C)).sum()))
            f.write('\n')
            f.write('\n')

    def run(self):
        self.num_player = self.player_nogui
        self.num_toppings = self.num_toppings_nogui
        self.all_player_instances = [ default_player(self.num_toppings, self.rng), p1(self.num_toppings, self.rng), p2(self.num_toppings, self.rng), p3(self.num_toppings, self.rng), p4(self.num_toppings, self.rng), p5(self.num_toppings, self.rng), p6(self.num_toppings, self.rng)]
        self.preferences = self.all_player_instances[self.generator_number].customer_gen(10, self.rng_generator_10)
        self.initialise_player(self.num_player, self.autoplayer_number)
        self.pizzas = self.player_instance.choose_toppings(self.all_player_instances[self.generator_number].customer_gen(100, self.rng_generator_100))
        self.pizzas_drawn = constants.number_of_initial_pizzas
        clash_exists_overall = False
        for i in range(constants.number_of_initial_pizzas):
            for j in range(24):
                pizza = self.pizzas[i]
                clash_exists = pizza_calculations.clash_exists(self.pizzas[i][j][0], self.pizzas[i][j][1], pizza, j)
                if clash_exists == True:
                    clash_exists_overall = True
        if clash_exists_overall:
            print("Overlapping placement. You cannot serve customers now.")
        else:
            print("Your shop is now open!!!")
            for i in range(constants.number_of_initial_pizzas):
                options_pizza = []
                for i in range(len(self.cuts)):
                    if list(self.cuts[i]) == [0,0,0]:
                        options_pizza.append(i)
                pizza_id, center, theta = self.player_instance.choose_and_cut(self.pizzas, options_pizza, self.preferences[self.customer_id])
                self.pizza_choice_order.append(pizza_id)
                self.pizza_id = pizza_id
                self.cuts[pizza_id][0] = (self.x + center[0]*self.multiplier)
                self.cuts[pizza_id][1] = (self.y + center[1]*self.multiplier)
                self.cuts[pizza_id][2] = theta
            self.see_score()



# Main Function Trigger
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", default = 5, help="Choose seed number")
    args = parser.parse_args()
    instance = no_gui(args)
    instance.run()