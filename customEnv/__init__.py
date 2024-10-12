from gymnasium.envs.registration import register

register(
    id="customEnv/GridWorld-v0",
    entry_point="customEnv.envs:GridWorld"
)