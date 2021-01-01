from gym.envs.registration import register

register(
    id='Kyoto-v0',
    entry_point='Kyoto_env.env:Kyoto'
)
