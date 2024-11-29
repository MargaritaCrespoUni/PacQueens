# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point



#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

#  Value Iteration Strategy for Decision Making, not implemented finally -- in progress
class ValueIterationAgent(CaptureAgent):
    """
    An agent that runs value iteration for strategic decision making.
    """

    def __init__(self, index, discount=0.9, iterations=100):
        super().__init__(index)
        self.discount = discount  # Discount factor for future rewards
        self.iterations = iterations  # Number of iterations to run value iteration
        self.values = {}  # Stores the value of each state
        self.run_value_iteration()  # Run value iteration during initialization

    def run_value_iteration(self):
        for iteration in range(self.iterations):
            new_values = {}
            for state in self.get_states():
                if self.is_terminal(state):
                    new_values[state] = 0  # Terminal states have zero value
                    continue
                max_q_value = float('-inf')
                # Find the action with the highest Q-value
                for action in self.get_possible_actions(state):
                    q_value = self.compute_q_value(state, action)
                    max_q_value = max(max_q_value, q_value)
                new_values[state] = max_q_value
            self.values = new_values  # Update the state values

    def compute_q_value(self, state, action):
        # Compute the Q-value 
        q_value = 0
        for next_state, probability in self.get_transition_states_and_probs(state, action):
            reward = self.get_reward(state, action, next_state)
            q_value += probability * (reward + self.discount * self.values.get(next_state, 0))
        return q_value

    def choose_action(self, game_state):
        # Choose best action 
        actions = game_state.get_legal_actions(self.index)
        best_action = max(actions, key=lambda action: self.compute_q_value(game_state, action))
        return best_action

    def get_states(self):
        """
        Return a list of all possible states in the environment.
        """
        return []
    
    def is_terminal(self, state):
        """
        Check if the given state is terminal (end of the game).
        """
        return False

    def get_possible_actions(self, state):
        """
        Return possible actions that can be taken from the given state.
        """
        return []

    def get_transition_states_and_probs(self, state, action):
        """
        Return the next states and their probabilities for a given state and action.
        """
        return []
    
    def get_reward(self, state, action, next_state):
        return 0

class OffensiveReflexAgent(ReflexCaptureAgent, ValueIterationAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def __init__(self, index, discount=0.9, iterations=100, return_threshold=5):
        # Initialize both the reflex and value iteration components
        ReflexCaptureAgent.__init__(self, index)
        ValueIterationAgent.__init__(self, index, discount=discount, iterations=iterations)

        self.visited_states = set()

        # Run value iteration but not finished 
        # self.run_value_iteration()
    
    def update_visited_states(self, state):
        self.visited_states.add(state)
    
    def get_home_positions(self, game_state):
        """
        Returns a list of positions that represent the agent's home boundary.
        """
        walls = game_state.get_walls()
        width = walls.width
        height = walls.height
        is_red = game_state.is_on_red_team(self.index)

        # Define home boundary (left half for red, right half for blue)
        home_x = range(0, width // 2) if is_red else range(width // 2, width)
        home_positions = [(x, y) for x in home_x for y in range(height) if not walls[x][y]]

        return home_positions
        

    def choose_action(self, game_state):
        """
        Choose an action based on a combination of reflex behavior and value iteration.
        """
        actions = game_state.get_legal_actions(self.index)
        my_pos = game_state.get_agent_state(self.index).get_position()

        # Compute ghost distances
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        visible_ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        ghost_positions = [g.get_position() for g in visible_ghosts]

        # Determine if escape is necessary
        danger_range = 3  
        ghosts_too_close = [
            ghost_pos for ghost_pos in ghost_positions
            if self.get_maze_distance(my_pos, ghost_pos) <= danger_range
        ]

        # If ghosts are too close, escape
        if ghosts_too_close:
            safe_actions = [
                action for action in actions
                if (action != Directions.STOP) and all(
                    self.get_maze_distance(
                        self.get_successor(game_state, action).get_agent_state(self.index).get_position(),
                        ghost_pos
                    ) >= danger_range
                    for ghost_pos in ghosts_too_close
                )
            ]
            if safe_actions:
                return random.choice(safe_actions) 
        
            # No safe actions
            least_dangerous_action = max(
                (action for action in actions if action != Directions.STOP),
                key=lambda action: min(
                    self.get_maze_distance(
                        self.get_successor(game_state, action).get_agent_state(self.index).get_position(),
                        ghost_pos
                    ) for ghost_pos in ghosts_too_close
                )
            )
            return least_dangerous_action

        
        # Run reflex evaluation to determine immediate best actions
        reflex_values = [self.evaluate(game_state, action) for action in actions]
        max_reflex_value = max(reflex_values)
        reflex_best_actions = [a for a, v in zip(actions, reflex_values) if v == max_reflex_value]

        # Choose the best action from reflex evaluation
        if len(reflex_best_actions) > 1:
            # Break ties 
            best_action = min(
                reflex_best_actions,
                key=lambda action: self.get_features(game_state, action)['distance_to_food']
            )
        else:
            best_action = reflex_best_actions[0]

        # Update visited states
        successor = self.get_successor(game_state, best_action)  # Get the state from best action
        new_state = successor.get_agent_state(self.index).get_position() 
        self.update_visited_states(new_state)  # Add positions to visited states to avoid loops


        return best_action
    

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = self.get_score(successor)
        my_state = successor.get_agent_state(self.index)

        # Compute distance to the nearest food
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance


        # Add feature to track distance to power capsules when ghosts are near
        capsules = self.get_capsules(successor)
        if len(capsules) > 0:
            min_capsule_dist = min([self.get_maze_distance(my_pos, cap) for cap in capsules])
            features['distance_to_capsule'] = min_capsule_dist  # Smaller distance to capsules is better
    
        # Create list of scared and active ghosts
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        scared_ghosts = [g for g in ghosts if g.scared_timer > 0]
        active_ghosts = [g for g in ghosts if g.scared_timer == 0]

        # Distance to the nearest scared ghost
        if len(scared_ghosts) > 0: #only loop if we know we have scared ghosts
            min_scared_dist = min([self.get_maze_distance(my_pos, g.get_position()) for g in scared_ghosts])
            features['distance_to_scared_ghost'] = min_scared_dist # Smaller distance to scared ghost 

        # Distance to the nearest active ghost
        if len(active_ghosts) > 0:#only loop if we know we have active ghosts
            min_active_ghost_dist = min([self.get_maze_distance(my_pos, g.get_position()) for g in active_ghosts])
            features['distance_to_active_ghost'] = min_active_ghost_dist # Smaller distance to active ghost 
        else:
            features['distance_to_active_ghost'] = 0  # If no active ghosts feature still exists but we don't mind

        if action == Directions.STOP: features['stop'] = 1 #create feature stop

        successor_state = successor.get_agent_state(self.index).get_position()
        
        # Penalize revisiting states
        if successor_state in self.visited_states:
            features['visited'] = 1
        else:
            features['visited'] = 0
        

        # Feature: Distance to home boundary
        food_carried = my_state.num_carrying
        food_threshold = 5 #when carrying 5 pellets return home

        if food_carried >= food_threshold:
            # Add distance_to_home as a feature 
            home_positions = self.get_home_positions(game_state)
            closest_home = min(home_positions, key=lambda pos: self.get_maze_distance(my_pos, pos))
            features['distance_to_home'] = self.get_maze_distance(my_pos, closest_home)
        else:
            features['distance_to_home'] = 0  # Neutral if not returning home

        return features


    def get_weights(self, game_state, action):
        # Adjust weights dynamically based on ghost proximity
        features = self.get_features(game_state, action)
        
        # If there are active ghosts nearby, prioritize moving away
        active_ghost_proximity = features['distance_to_active_ghost']
        scared_ghost_proximity = features.get('distance_to_scared_ghost', 0) #put it to 0 if no scared ghosts

        my_state = game_state.get_agent_state(self.index)
        food_carried = my_state.num_carrying
        food_threshold = 5 

        successor_score = 200 
        distance_to_food = -50
        distance_to_home = 0

        if food_carried >= food_threshold:
            distance_to_food = -10
            distance_to_home = -100

        #we prioritize move towards scared ghost so me put this 'if' statement after
        if scared_ghost_proximity < 10:
            distance_to_food = 0
            successor_score = 10
            

        #don't eat at your own home (doesn't work)
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()
        home_positions = self.get_home_positions(game_state)
        food_list = self.get_food(successor).as_list()
        if my_pos in home_positions and my_pos in food_list:
            distance_to_food = 0


        return {
            'successor_score': successor_score,
            'distance_to_food': distance_to_food,
            'distance_to_capsule': -15 if active_ghost_proximity < 10 else 0,
            'distance_to_active_ghost': 50 if active_ghost_proximity < 20 else 0,  # scape from active ghosts
            'distance_to_scared_ghost': -100 if scared_ghost_proximity < 20 else 0,  # follow towards scared ghosts
            'stop' : -1000,
            'visited': -500,  # Penalize loops 
            'distance_to_home': distance_to_home
        }

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
        

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        my_state = game_state.get_agent_state(self.index)
        scared_timer = my_state.scared_timer  # Timer indicating how long the ghost will remain scared

        if scared_timer > 0:  # If the ghost is scared
            return {
                'num_invaders': -500,  # Lower priority to stopping invaders when scared
                'on_defense': 50,      # Stay on your side but not as critical
                'invader_distance': 100,  # Encourage moving away from invaders
                'stop': -100,          # Discourage stopping
                'reverse': -2          # Slight penalty for reversing direction
            }
        else:  # If the ghost is not scared
            return {
                'num_invaders': -1000,  # Strong penalty for invaders being present
                'on_defense': 100,      # High priority to staying on defense
                'invader_distance': -1000,  # Encourage moving closer to invaders
                'stop': -100,           # Strong penalty for stopping
                'reverse': -2           # Small penalty for reversing direction
            }

