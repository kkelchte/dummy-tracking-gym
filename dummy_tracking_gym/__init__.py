from gym.envs.registration import register

register(
    id='tracking-v0',
    entry_point='dummy_tracking_gym.envs:DummyTrackingEnv',
)
