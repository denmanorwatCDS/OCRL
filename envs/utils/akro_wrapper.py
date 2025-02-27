import akro

class InOutSpec:
    """Describes the input and output spaces of a primitive or module.

    Args:
        input_space (akro.Space): Input space of a module.
        output_space (akro.Space): Output space of a module.

    """

    def __init__(self, input_space, output_space):
        self._input_space = input_space
        self._output_space = output_space

    @property
    def input_space(self):
        """Get input space of the module.

        Returns:
            akro.Space: Input space of the module.

        """
        return self._input_space

    @property
    def output_space(self):
        """Get output space of the module.

        Returns:
            akro.Space: Output space of the module.

        """
        return self._output_space

class EnvSpec(InOutSpec):
    """Describes the action and observation spaces of an environment.

    Args:
        observation_space (akro.Space): The observation space of the env.
        action_space (akro.Space): The action space of the env.

    """

    def __init__(self, observation_space, action_space):
        super().__init__(action_space, observation_space)

    @property
    def action_space(self):
        """Get action space.

        Returns:
            akro.Space: Action space of the env.

        """
        return self.input_space

    @property
    def observation_space(self):
        """Get observation space of the env.

        Returns:
            akro.Space: Observation space.

        """
        return self.output_space

    @action_space.setter
    def action_space(self, action_space):
        """Set action space of the env.

        Args:
            action_space (akro.Space): Action space.

        """
        self._input_space = action_space

    @observation_space.setter
    def observation_space(self, observation_space):
        """Set observation space of the env.

        Args:
            observation_space (akro.Space): Observation space.

        """
        self._output_space = observation_space

    def __eq__(self, other):
        """See :meth:`object.__eq__`.

        Args:
            other (EnvSpec): :class:`~EnvSpec` to compare with.

        Returns:
            bool: Whether these :class:`~EnvSpec` instances are equal.

        """
        return (self.observation_space == other.observation_space
                and self.action_space == other.action_space)

class AkroWrapperTrait:
    @property
    def spec(self):
        return EnvSpec(action_space=akro.from_gym(self.action_space),
                       observation_space=akro.from_gym(self.observation_space))

