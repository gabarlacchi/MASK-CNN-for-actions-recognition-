import pickle
import numpy as np
import matplotlib.pyplot as plt

# class_ids = pickle.load(open("evaluationVars/class_ids.pkl", "rb"))
# scores = pickle.load(open("evaluationVars/scores.pkl", "rb"))
# pred_matches = pickle.load(open("evaluationVars/pred_matches.pkl", "rb"))
# gt_matches = pickle.load(open("evaluationVars/gt_matches.pkl", "rb"))

class_ids = pickle.load(open("evaluationVarsM/class_ids.pkl", "rb"))
scores = pickle.load(open("evaluationVarsM/scores.pkl", "rb"))
pred_matches = pickle.load(open("evaluationVarsM/pred_matches.pkl", "rb"))
gt_matches = pickle.load(open("evaluationVarsM/gt_matches.pkl", "rb"))

class_names = ['Background', 'WalkingWithDog', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving',
                'Fencing', 'FloorGymnastics', 'GolfSwing','HorseRiding', 'IceDancing', 'LongJump',
                'PoleVault', 'RopeClimbing', 'SalsaSpin','SkateBoarding', 'Skiing', 'Skijet',
                'SoccerJuggling', 'Surfing', 'TennisSwing','TrampolineJumping', 'VolleyballSpiking', 'Basketball']

targetClass = 3

target_ids = []
target_scores = []
target_pred_m = []
target_gt_len = 0

# fro every image
for i in range( len(class_ids) ):
	# for evry detected subject
	aBool = False
	for j in range( len(class_ids[i]) ):
		# if is of the target class
		if class_ids[i][j] == targetClass:
			# append the relative score and pred_match to global vectors
			target_scores.append(scores[i][j])
			target_pred_m.append(pred_matches[i][j])
			# TO CHECK... Lot's of doubt here
			aBool = True

	if aBool:
		target_gt_len = target_gt_len + len(gt_matches[i])


# sort prediction and scores by scores
score, pm = zip(*sorted(zip(target_scores, target_pred_m)))
# Revert array cause the sort is ascending
score = list(reversed(score))
pm = list(reversed(pm))

# Compute precision and recall
precisions = np.cumsum(pm) / (np.arange(len(pm)) + 1)
recalls = np.cumsum(pm).astype(np.float32) / target_gt_len

#pad with starting and finisching values
precisions = np.concatenate([[0], precisions, [0]])
recalls = np.concatenate([[0], recalls, [1]])

# print(precisions)
# print(recalls)

# Plot Graph
plt.plot(recalls, precisions)
plt.title( class_names[targetClass] )
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()