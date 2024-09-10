from BasicEnvironment import *

env = BasicEnv()
env.render()
action = int(input("Enter action:"))
state, reward, done, info = env.step(action)
while not done:
    env.render()
    action = int(input("Enter action:"))
    state, reward, done, info = env.step(action)