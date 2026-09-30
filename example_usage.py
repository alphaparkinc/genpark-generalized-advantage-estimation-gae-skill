from client import GAEAdvantageEstimator

rewards = [1.0, 0.0, 3.0]
values = [0.5, 0.8, 1.2]
res = GAEAdvantageEstimator.compute_gae(rewards, values)
print("GAE Advantages:", res["advantages"])
print("Target Returns:", res["returns"])
