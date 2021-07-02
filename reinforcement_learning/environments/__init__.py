from gym.envs.registration import register

register(
    id='LidarBat-v0',
    entry_point='environments.bat_flying_env:BatFlyingEnv'

)

register(
    id='LidarBat-v1',
    entry_point='environments.bat_flying_env_four_chains:BatFlyingEnv')
