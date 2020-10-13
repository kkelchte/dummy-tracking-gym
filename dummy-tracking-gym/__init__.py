from gym.envs.registration import register

register(
    id='dummy-tracking-gym-v0',
    entry_point='dymmy-tracking-gym.envs:DummyTrackingEnv',
)
register(
    id='dummy-tracking-two-player-v0',
    entry_point='dummy-tracking-gym.envs:DummyTrackingTwoPlayerEnv',
)
