from pizza_no_gui import no_gui
import argparse
import pickle as pkl


parser = argparse.ArgumentParser()
parser.add_argument("--gui", "-g", default="False", help="GUI")
parser.add_argument("--interface_size", "-sz", default=40, help="GUI Size")
parser.add_argument("--seed", "-s", default=40, help="General seed for your own functions")
parser.add_argument("--gen_100_seed", "-s100", default=40, help="Seed for generating 100 preferences")
parser.add_argument("--gen_10_seed", "-s10", default=45, help="Seed for generating 10 preferences")
#parser.add_argument("--autoplayer_number", "-a_num", default=0, help="Which player is the autoplayer")
parser.add_argument("--generator_number", "-g_num", default=0, help="Which player is the preference generator")
parser.add_argument("--player", "-p", default=0, help="Team number playing the game if no gui")
parser.add_argument("--num_toppings", "-num_top", default=2, help="Total different types of toppings")
parser.add_argument("--tournament", "-tmnt", default=True, help="Is this a tournament run or not")
args = parser.parse_args()
args.tournament = True

a = [[],[],[],[],[],[]]
with open("tournament_results.pkl", "wb") as fp:
    pkl.dump(a, fp)


for i in [6]:
    for k in [2,3,4]:
        for j in [0,1,3,4]: #placeholder for whatever tournament conditions we have.
            for run in range(30):
                print(f"Player {i},Toppings {k}, Distribution {j}, Run {run}.....")
                args.player = i
                args.num_toppings = k
                args.generator_number = j
                args.gen_100_seed = run
                args.gen_10_seed = run + 1
                args.seed = run + 2
                instance = no_gui(args)
                instance.run()