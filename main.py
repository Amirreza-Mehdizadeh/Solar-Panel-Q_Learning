import numpy as np
import matplotlib.pyplot as plt

# Defining Environment

class solarpanelenv:
    def __init__(self,angle_step= 1, max_angle=90 ):
        self.angle_step=angle_step       # step of angle change 
        self.max_angle=max_angle         # max of angle
        self.states=np.arange(0,max_angle+1,angle_step)   # our states [0,5,...,90]
        self.actions=[-5,-3,+3,+5]
        self.sun_angle=45
        self.state=None
        
    def reset(self):
        self.state=np.random.choice(self.states)
        return self.state

    def step(self,action):
        next_state= self.state + action      # Calculate the new angle
        next_state= max(0,min(self.max_angle,next_state))      # Limiting the angle
        angle_diff= abs(next_state - self.sun_angle)
        reward= np.cos(np.radians(angle_diff))
        done= (angle_diff < 2.5)
        self.state=next_state
        return next_state,reward,done

# Q-Learning Algorithm 
def q_learning(env,alpha=0.3, gamma=.7, epsilon=0.1,episodes=1000):
    q={}
    for state in env.states:
       q[state]=np.zeros(len(env.actions))
    rewards=[]
    for episode in range(episodes):
        state=env.reset()
        total_reward=0
        done=False
        
        while not done:
            # Epsilon_greedy Policy
            if np.random.uniform(0,1) < epsilon:
                action_idx=np.random.randint(len(env.actions))
            else:
                action_idx=np.argmax(q[state])
        
            action=env.actions[action_idx]
            # Execute action
            next_state,reward,done= env.step(action)
            total_reward+=reward

            # updating Q
            if done:
                best_next_q=0.0
            else:
                best_next_q=np.max(q[state])
                q[state][action_idx] += alpha * (reward + gamma * best_next_q - q[state][action_idx])
                state = next_state
        rewards.append(total_reward)
        epsilon = max(0.01, epsilon * 0.995)
        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.3f}")

    return q, rewards

# Test Policy

def test_policy (env,q,episodes=10):
    for episode in range(episodes):
        state=env.reset()
        done=False
        total_reward=0
        steps=0
        print(f"\nTest Episode {episode + 1}: Starting at angle {state}")
        while not done:
            action_idx= np.argmax(q[state])
            action=env.actions[action_idx]
            next_state,reward,done= env.step(action)
            total_reward+= reward
            steps+=1
            print(f"Step {steps}: Angle = {next_state}, Reward = {reward:.2f}")
            state=next_state
        print(f"Test Episode {episode + 1}: Total Reward = {total_reward:.2f}, Steps = {steps}")
        

# Main Function

def main():
    # making enviroment
    env = solarpanelenv(angle_step=1 ,max_angle=90 )
    q,reward= q_learning (env,episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1)
    print("\nFinal Q-table:")
    
    # Print the table
    for state in env.states:
        print(f"Angle {state}: Q-values = {q[state]} (left, right)")
        
    #Learned Policy Test
    test_policy(env, q, episodes=5)
    
    # Plotting the trainig rewards
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards over Episodes")
    plt.show()

if __name__ == "__main__":
    main()
    





