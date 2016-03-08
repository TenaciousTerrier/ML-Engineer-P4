import random
import os
import sys

# Clear console and Change current working directory
# os.system('cls')
# newPath = 'C:\\Users\\thep3\\Desktop\\Machine Learning Engineer\\P4\\smartcab'
# os.chdir(newPath)
# print os.getcwd()

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        """ Initializes necessary variables and objects for agent.
        """
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # Initialize any additional variables here
        self.Qtable = {} 
        self.previous_state = None
        self.previous_action = None
        self.learning_rate = 0.5
        self.discount_factor = 1.0
        self.epsilon = 1.0
        
        # Variables For Testing Purposes:
        self.success_count = 0
        self.trial_count = 0
        self.correct_action_count = 0
        self.action_count = 0
        self.positive_success_count = 0
        self.negative_rewards = 0
        self.net_reward = 0
        
    def reset(self, destination=None):
        """ Updates variables before starting new trial.
        """
        self.planner.route_to(destination)
        
        # Prepare for a new trip; reset any variables here, if required
        self.previous_state = None
        self.previous_action = None  
        self.epsilon =  1.0 / (self.trial_count + 1.0)
        self.discount_factor = (100.0 - self.trial_count) / 100.0
        self.net_reward = 0 
        self.trial_count = self.trial_count + 1
        
    def update(self, t):
        """ Updates agent for each new trial.
        """
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        self.state = create_state(self, inputs)
        self.previous_state = self.state  
        
        # Select action according to your policy - Choose action with maximum Q-value for given state
        action = select_action(self, inputs, self.Qtable)
        self.previous_action = action
        
        # Execute action and get reward
        reward = self.env.act(self, action)    
        self.net_reward = self.net_reward + reward
 
        # Increments counts - for testing purposes    
        if self.trial_count >= 91 and reward < 0: 
             self.negative_rewards = self.negative_rewards + 1
                              
        if self.trial_count >= 91: 
            self.action_count = self.action_count + 1
            if action == self.state[4][1]:
                self.correct_action_count = self.correct_action_count + 1
                
        if self.env.done is True and self.trial_count >= 91:
            self.success_count = self.success_count + 1
            self.positive_success_count = self.positive_success_count + 1
            
        # Learn policy based on state, action, reward
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        self.state = create_state(self, inputs)
        update_Qtable(self, self.Qtable, action, reward)

        # Print summary for iteration - used for testing purposes
        print ""
        print "Trial: " + str(self.trial_count)
        print "Epsilon: " + str(self.epsilon)
        print "Learning Rate: " + str(self.learning_rate)
        print "Discount Factor: " + str(self.discount_factor)
        print ""
        print "LearningAgent.update(): state = {}, action = {}, reward = {}".format(self.previous_state,self.previous_action, reward)
        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def create_state(self, inputs):
    """ Takes set of inputs and returns a state.
    """
    state1 = ('light',inputs['light'])
    state2 = ('oncoming', inputs['oncoming'])
    state3 = ('left', inputs['left']) 
    state4 = ('right',inputs['right'])
    state5 = ('next_waypoint',self.next_waypoint)
    self.state = (state1, state2, state3, state4, state5)
    return self.state

def select_action(self, inputs, Qtable):
    """ Returns best action from Q-table, or if that option does not exist, randomly chooses
        an action with certain specifications.
    """
    action = None
    max_Qvalue = 0 
    action_selected = False
    for a in self.env.valid_actions:   
        # Initialize to 0 if (state, action) does not exist
        if self.Qtable.get((self.state, a)) is None:
            self.Qtable[(self.state, a)] = 0 
        
        if self.Qtable.get((self.state, a)) > max_Qvalue:
            max_Qvalue = self.Qtable.get((self.state, a))   
            action = a
            action_selected = True
                
    # Choose random action with probability epsilon      
    random_decimal = random.randint(1,100) / 100.0
    if random_decimal <= self.epsilon:
        action = random.choice(['forward', 'right', 'left'])
        action_selected = True
    
    # Choose random action if no action selected
    if action_selected is False:
        action_selected = True 
        action = random.choice(['forward', 'right', 'left'])
 
    return action

def update_Qtable(self, Qtable, action, reward):
    """ Take inputs and updates Qtable. """
    old_Qvalue = 0
    if self.Qtable.get((self.state, action)) is not None:
        old_Qvalue = self.Qtable.get((self.state, action))
    
    next_Qvalue = 0
    for a in self.env.valid_actions:               
        if self.Qtable.get((self.state, a)) > next_Qvalue:
            next_Qvalue = self.Qtable.get((self.state, a))   
          
    Qvalue = old_Qvalue + self.learning_rate * (reward + self.discount_factor * next_Qvalue - old_Qvalue)
    self.Qtable[(self.previous_state, self.previous_action)] = Qvalue


def run():
    """Run the agent for a finite number of trials."""
    # Set random seed (for testing purposes)
    # random.seed(5) # used seeds 5, 200, 201, 203, 206, 348 for testing
            
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.00001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit
    
    # Print Summary of Last 10 Trials
    print ""
    print "Last 10 Trials Summary:"
    print "Number of States: " + str(len(a.Qtable.keys()))
    print "Correct Action: " + str(a.correct_action_count) + " / " + str(a.action_count)
    print "Negative Rewards: " + str(a.negative_rewards)
    print "Positive Net Reward Ratio : " + str(a.positive_success_count) + " / " + str(a.success_count)
    print "Success Ratio: " + str(a.success_count) + " / 10"
    
if __name__ == '__main__':
    run()