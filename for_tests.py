import gymnasium
import customEnv
import cv2

env = gymnasium.make("customEnv/GridWorld-v0",render_mode="human")

state = env.reset()

# Run a few steps in the environment
for _ in range(1000):
    action = env.action_space.sample()  # Sample a random action
    print(action)
    state, reward, done,truncated, info = env.step(action)
    cv2.imshow('123', state['agent'])
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    env.render()  # Render the current state in human mode
    if done:
        break

env.close()

