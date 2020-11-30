from gym.envs.registration import register

register(
    id='Kyoto_ontime-v0',
    entry_point='Kyoto_env_ontime.env:Kyoto_ontime'
)
