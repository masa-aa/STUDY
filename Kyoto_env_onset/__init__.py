from gym.envs.registration import register

register(
    id='Kyoto_onset-v0',
    entry_point='Kyoto_env_onset.env:Kyoto_onset'
)
