# @Author:HeRuonan
# @Date:2024-08-26 17:39:36
# @LastModifiedBy:HeRuonan
# @Last Modified time:2024-08-26 17:39:36
import gym
import torch
import torch.nn as nn
import torch.optim as optim

env = gym.make('CartPole-v1')

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        return action_probs

input_dim = env.observation_space.shape[0]  # 状态空间维度
output_dim = env.action_space.n  # 动作空间大小

policy_net = PolicyNetwork(input_dim, output_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-2)

def evaluate_policy(policy_net, env, episodes=10):
    total_rewards = 0
    for i in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs = policy_net(state_tensor)
            action = torch.argmax(action_probs).item()
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_rewards += episode_reward

    average_reward = total_rewards / episodes
    return average_reward

episodes=10
# 使用上文定义的PolicyNetwork和初始化的env
average_reward = evaluate_policy(policy_net, env)
print(f"Average reward over {episodes} episodes: {average_reward}")

# 保存模型
torch.save(policy_net.state_dict(), 'policy_net_model.pth')

# 加载模型
loaded_policy_net = PolicyNetwork(input_dim, output_dim)
loaded_policy_net.load_state_dict(torch.load('policy_net_model.pth'))

# 示例：将PyTorch模型转为ONNX格式
dummy_input = torch.randn(1, input_dim)
torch.onnx.export(policy_net, dummy_input, "policy_net_model.onnx")
