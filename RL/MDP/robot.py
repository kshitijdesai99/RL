# Problem Statement:
# A robot is placed in a n*n grid. 
# The robot can move up, down, left, right.
# There are 2 types of states --> Goal state, Penalty state.
# The goal is to navigate the robot from its start state to a goal state.
# The robot stops if it reaches goal or penalty states.
# Goal state and penalty states are fixed
# Its a non deterministic algorithm due to presence of transition probabilities
# We have to find an action for each state such that it maximizes the chance of robot reaching the reward state

# Solution
import numpy as np

class Grid_world:
    def __init__(self, 
                 grid_size, 
                 goal_state,
                 trap_states, 
                 actions, 
                 actions_mapping,    
                 discount_factor, 
                 default_reward, 
                 goal_reward, 
                 trap_penalty,
                 transition_probability_fn):
        self.grid_size = grid_size
        self.goal_state = goal_state
        self.trap_states = trap_states
        self.actions = actions
        self.actions_mapping = actions_mapping
        self.discount_factor = discount_factor
        self.default_reward = default_reward
        self.goal_reward = goal_reward
        self.trap_penalty = trap_penalty
        self.transition_probability_fn = transition_probability_fn  # Function for dynamic probabilities
        self.values = np.zeros((grid_size, grid_size))
        self.policy = np.full((grid_size, grid_size), "", dtype=object)

        # Initialize rewards and policy
        for x in range(grid_size):
            for y in range(grid_size):
                if (x, y) == self.goal_state:
                    self.values[x, y] = goal_reward
                    self.policy[x, y] = "Goal"
                elif (x, y) in self.trap_states:
                    self.values[x, y] = trap_penalty
                    self.policy[x, y] = "Trap"
                else:
                    self.values[x, y] = default_reward

    def is_valid_state(self, state):
        """Check if a state is within grid boundaries."""
        x, y = state
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def get_possible_transitions(self, state, target_action):
        """Get possible transitions for a given state and action."""
        x, y = state
        dx, dy = self.actions_mapping[target_action]
        intended_state = (x + dx, y + dy)

        # Get dynamic probabilities for this state-action pair
        transition_prob = self.transition_probability_fn(state, target_action)
        non_transition_prob = (1 - transition_prob) / (len(self.actions) - 1)

        transitions = []

        # Add intended transition
        if self.is_valid_state(intended_state):
            transitions.append((intended_state, transition_prob))
        else:
            transitions.append((state, transition_prob))  # Stay in the same state if invalid

        # Add unintended transitions
        for action in self.actions:
            if action != target_action:
                dx, dy = self.actions_mapping[action]
                non_intended_state = (x + dx, y + dy)
                if self.is_valid_state(non_intended_state):
                    transitions.append((non_intended_state, non_transition_prob))
                else:
                    transitions.append((state, non_transition_prob))  # Stay in the same state if invalid

        return transitions

    def value_iteration(self, max_iterations: int, convergence_threshold: float):
        """Perform value iteration to calculate optimal values and policy."""
        for _ in range(max_iterations):
            old_values = np.copy(self.values)
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    state = (x, y)

                    # Skip fixed states (goal, trap)
                    if state == self.goal_state or state in self.trap_states:
                        continue

                    action_values = {}
                    for action in self.actions:
                        total_value = 0
                        possible_transitions = self.get_possible_transitions(state, action)
                        for next_state, transition_prob in possible_transitions:
                            # Assign appropriate rewards for transitions
                            if next_state == self.goal_state:
                                reward = self.goal_reward
                            elif next_state in self.trap_states:
                                reward = self.trap_penalty
                            else:
                                reward = self.default_reward

                            # Compute total expected value
                            total_value += transition_prob * (reward + self.discount_factor * old_values[next_state[0], next_state[1]])

                        action_values[action] = total_value

                    # Update value function and policy
                    max_expected_value = max(action_values.values())
                    self.values[x, y] = max_expected_value
                    best_action = max(action_values, key=action_values.get)
                    self.policy[x, y] = best_action

            # Check for convergence
            if np.max(np.abs(self.values - old_values)) < convergence_threshold:
                break

        return self.values, self.policy

# Dynamic Transition Probability Function
def transition_probability_fn(state, action):
    """Example function: Probabilities vary based on proximity to goal state."""
    x, y = state
    if action == "up":
        return 0.8 if x > y else 0.6
    elif action == "down":
        return 0.7 if x < y else 0.5
    elif action == "left":
        return 0.6 if x + y < 5 else 0.4
    elif action == "right":
        return 0.9 if x + y > 5 else 0.7
    return 0.5  # Default probability

# Test the updated class
grid_world = Grid_world(
    grid_size=5, 
    goal_state=(4, 4), 
    trap_states=[(1, 1), (2, 2)], 
    actions=["up", "down", "left", "right"], 
    actions_mapping={"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)},
    transition_probability_fn=transition_probability_fn,
    discount_factor=0.9,
    default_reward=-0.2,
    goal_reward=10,
    trap_penalty=-5,
)

max_iterations = 100
convergence_threshold = 0.001

values, policy = grid_world.value_iteration(max_iterations, convergence_threshold)

# Display results
print("Values:")
print(values)
print("\nPolicy:")
print(policy)
