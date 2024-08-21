import requests
import csv
import pandas as pd
from gekko import GEKKO

with open("Premier_League_Matchday_1.csv",
          encoding='utf-8', newline='') as csvfile:
    reader = csv.reader(csvfile)

    player_name = []  # 35
    player_positions = []  # 8 int
    player_cost = []  # 18 divide by 10.0 float
    points_per_game = []  # 20 float
    player_team = []  # 26 int

    next(reader)
    for row in reader:
        player_name.append(row[35])
        player_positions.append(int(row[8]))
        player_cost.append(float(row[18]) / 10.0)
        points_per_game.append(float(row[20]))
        player_team.append(int(row[26]))

num_of_players = len(player_name)
player_total_requirements = {1: 2, 2: 5, 3: 5, 4: 3}  # 2 GK, 5 DF, 5 MF, 3 FW.
starting_lineup_requirements = {2: 3, 4: 1} #1 GK, at least 3 DF, 1 FW.
budget_requirement = 100
positions_dict = {1:"GK" , 2:"DF" , 3:"MF" , 4:"FW"}

data = pd.DataFrame({
    "Player": player_name,
    "Position": player_positions,
    "Cost": player_cost,
    "Points_Per_Game": points_per_game,
    "Team": player_team,
})

model = GEKKO(remote=False)

#Variables for all players, starting lineup players, and substitutes.
player_var = [model.Var(lb=0, ub=1, integer=True) for _ in range(num_of_players)]
starting_var = [model.Var(lb=0, ub=1, integer=True) for _ in range(num_of_players)]
sub_var = [model.Var(lb=0, ub=1, integer=True) for _ in range(num_of_players)]

#Gets most points for starting lineup.
model.Maximize(model.sum([data.at[i, "Points_Per_Game"] * starting_var[i] for i in range(num_of_players)]))

#Make sure we have 15 total players.
model.Equation(model.sum(player_var) == 15)

#Make sure we don't go over budget requirement.
model.Equation(model.sum([player_var[i] * data.at[i, "Cost"] for i in range(num_of_players)]) <= budget_requirement)

#Make sure we have 11 players in starting lineup.
model.Equation(model.sum(starting_var) == 11)

#Make sure we have 4 players in substitutes.
model.Equation(model.sum(sub_var) == 4)

#Player can be a starter or a substitute if selected.
for i in range(num_of_players):
    model.Equation(starting_var[i] + sub_var[i] <= player_var[i])

#Make sure starting lineup has the minimum requirements.
model.Equation(model.sum([starting_var[i] for i in range(num_of_players) if data.at[i, 'Position'] == 1]) == 1)

for pos, count in starting_lineup_requirements.items():
    model.Equation(model.sum([starting_var[i] for i in range(num_of_players) if data.at[i, 'Position'] == pos]) >= count)

#Make sure the whole squad has the position requirements.
for pos, count in player_total_requirements.items():
    model.Equation(model.sum([player_var[i] for i in data.index if data.at[i, "Position"] == pos]) == count)

#Make sure that there are no more than 3 player from the same team.
teams = data["Team"].unique()
for team in teams:
    model.Equation(model.sum([player_var[i] for i in data.index if data.at[i, "Team"] == team]) <= 3)

model.options.SOLVER = 1
model.solve(disp=True)

#Results
starting_lineup_names = [data.at[i, "Player"] for i in range(num_of_players) if starting_var[i].value[0] == 1]
starting_lineup_position_nums = [data.at[i, "Position"] for i in range(num_of_players) if starting_var[i].value[0] == 1]
starting_lineup_points = [data.at[i, "Points_Per_Game"] for i in range(num_of_players) if starting_var[i].value[0] == 1]
starting_lineup_dict = {}
top_scores_dict = {}

#Connects the player's positions with their names.
for i in range(11):
    starting_lineup_dict[starting_lineup_names[i]] = starting_lineup_position_nums[i]

#Connects the player's points with their names.
for i in range(11):
    top_scores_dict[starting_lineup_names[i]] = starting_lineup_points[i]

#Sort them by positions.
starting_lineup_dict = dict(sorted(starting_lineup_dict.items(),key = lambda x:x[1]))
#Sort them by points
top_scores_dict = sorted(top_scores_dict.items(), key = lambda x:x[1])

#Captain gets double points, and vice gets double points if captain does not play.
captain = top_scores_dict.pop()
vice_captain = top_scores_dict.pop()

#Print results.
print("Starting Lineup:")
for key, value in starting_lineup_dict.items():
    if(captain[0] == key):
        print(positions_dict.get(value) + ": " + key + " (Captain)")
    elif(vice_captain[0] == key):
        print(positions_dict.get(value) + ": " + key + " (Vice-Captain)")
    else:
        print(positions_dict.get(value) + ": " + key)
print("\n")


subs_names = [data.at[i, "Player"] for i in range(num_of_players) if sub_var[i].value[0] == 1]
subs_position_nums = [data.at[i, "Position"] for i in range(num_of_players) if sub_var[i].value[0] == 1]
subs_dict = {}

for i in range(4):
    subs_dict[subs_names[i]] = subs_position_nums[i]

subs_dict = dict(sorted(subs_dict.items(),key = lambda x:x[1]))

print("Substitutes:")
for key, value in subs_dict.items():
    print(positions_dict.get(value) + ": " + key)
print("\n")


total_cost = sum(data.at[i, "Cost"] for i in range(num_of_players) if player_var[i].value[0] == 1)
print("Total Cost: " + "{:.1f}".format(total_cost))

total_points = sum(data.at[i, "Points_Per_Game"] for i in range(num_of_players) if starting_var[i].value[0] == 1)
print("Total Points: " + "{:.1f}".format(total_points + captain[1]))