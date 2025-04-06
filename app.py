import streamlit as st
import numpy as np
import time
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from environment import SimpleRoverEnv
from agent import SimpleAgent

# Page config for better appearance
st.set_page_config(
    page_title="Mars Rover SLAM Simulation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set dark theme
st.markdown("""
<style>
    .reportview-container {
        background-color: #0e1117;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #262730;
        color: white;
    }
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    .stProgress > div > div {
        background-color: #4CAF50 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'env' not in st.session_state:
    st.session_state.env = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'obs' not in st.session_state:
    st.session_state.obs = None
if 'running' not in st.session_state:
    st.session_state.running = False
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'rewards' not in st.session_state:
    st.session_state.rewards = []
if 'steps' not in st.session_state:
    st.session_state.steps = 0
if 'episode_rewards' not in st.session_state:
    st.session_state.episode_rewards = []
if 'td_errors' not in st.session_state:
    st.session_state.td_errors = []
if 'exploration_rates' not in st.session_state:
    st.session_state.exploration_rates = []
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = {'avg_reward': [], 'exploration_rate': [], 'episodes': []}
if 'simulation_speed' not in st.session_state:
    st.session_state.simulation_speed = 0.1  # Default simulation speed (delay in seconds)

def main():
    st.title("🚀 Mars Rover SLAM Simulation")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Environment settings
    st.sidebar.subheader("Environment Settings")
    map_size = st.sidebar.slider("Map Size", 10, 50, 25)
    render_scale = st.sidebar.slider("Render Scale", 10, 30, 20)
    
    # Agent settings
    st.sidebar.subheader("Agent Settings")
    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
    gamma = st.sidebar.slider("Discount Factor", 0.8, 0.999, 0.99, format="%.3f")
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    num_episodes = st.sidebar.slider("Training Episodes", 10, 200, 50)
    batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
    
    # Simulation speed control
    st.sidebar.subheader("Simulation Control")
    st.session_state.simulation_speed = st.sidebar.slider(
        "Simulation Speed", 
        min_value=0.01, 
        max_value=1.0, 
        value=st.session_state.simulation_speed,
        format="%.2f"
    )
    
    # Initialize environment and agent if not already done
    if st.session_state.env is None:
        with st.spinner("Initializing environment..."):
            st.session_state.env = SimpleRoverEnv(map_size=map_size, render_scale=render_scale, render_mode='opencv')
            st.session_state.obs = st.session_state.env.reset()
            st.session_state.agent = SimpleAgent(
                state_size=11,  # 8 sensors + 2 position + 1 angle
                action_size=3,  # forward, left, right
                learning_rate=learning_rate,
                gamma=gamma
            )
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display area for the simulation
        st.subheader("Rover Environment")
        map_placeholder = st.empty()
        
        # Render the initial state
        img = st.session_state.env.render_opencv()
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        map_placeholder.image(img_pil, use_column_width=True)
    
    with col2:
        # Controls and metrics
        st.subheader("Controls")
        
        # Training button
        if not st.session_state.training_complete:
            if st.button("🧠 Train Agent", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_text = st.empty()
                
                # Training loop
                episode_rewards = []
                exploration_rates = []
                avg_rewards = []
                total_steps = 0
                
                for episode in range(num_episodes):
                    state = st.session_state.env.reset()
                    total_reward = 0
                    steps = 0
                    done = False
                    
                    # Episode loop
                    while not done:
                        action = st.session_state.agent.act(state)
                        next_state, reward, done, info = st.session_state.env.step(action)
                        st.session_state.agent.remember(state, action, reward, next_state, done)
                        
                        # Train agent if enough experiences
                        if len(st.session_state.agent.memory) > batch_size:
                            st.session_state.agent.replay(batch_size)
                            
                        state = next_state
                        total_reward += reward
                        steps += 1
                        total_steps += 1
                        
                        # Update visualization occasionally during training
                        if episode % max(1, num_episodes // 10) == 0 and steps % 10 == 0:
                            img = st.session_state.env.render_opencv()
                            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                            map_placeholder.image(img_pil, use_column_width=True)
                            
                            # Brief delay to allow visualization
                            time.sleep(0.001)
                    
                    # Record episode results
                    episode_rewards.append(total_reward)
                    exploration_rates.append(st.session_state.agent.get_exploration_rate())
                    
                    # Calculate moving average reward
                    window_size = min(10, episode + 1)
                    avg_reward = sum(episode_rewards[-window_size:]) / window_size
                    avg_rewards.append(avg_reward)
                    
                    # Update progress
                    progress = (episode + 1) / num_episodes
                    progress_bar.progress(progress)
                    status_text.text(f"Episode {episode+1}/{num_episodes} - Reward: {total_reward:.2f}")
                    
                    # Update metrics
                    metrics_text.text(
                        f"Avg Reward (last 10): {avg_reward:.2f}\n"
                        f"Exploration Rate: {st.session_state.agent.get_exploration_rate():.3f}\n"
                        f"Total Steps: {total_steps}"
                    )
                
                # Save training results to session state
                st.session_state.episode_rewards = episode_rewards
                st.session_state.exploration_rates = exploration_rates
                st.session_state.performance_metrics = {
                    'avg_reward': avg_rewards,
                    'exploration_rate': exploration_rates,
                    'episodes': list(range(1, num_episodes + 1))
                }
                st.session_state.training_complete = True
                st.session_state.obs = st.session_state.env.reset()
                
                # Show training complete message
                st.success("Training complete! The agent is ready for testing.")
                
        # Simulation controls (only show after training)
        if st.session_state.training_complete:
            st.subheader("Simulation")
            
            # Start/Stop button
            start_stop_button = st.button(
                "⏹️ Stop Simulation" if st.session_state.running else "▶️ Start Simulation", 
                use_container_width=True
            )
            if start_stop_button:
                st.session_state.running = not st.session_state.running
            
            # Reset button
            if st.button("🔄 Reset Simulation", use_container_width=True):
                st.session_state.env = SimpleRoverEnv(map_size=map_size, render_scale=render_scale, render_mode='opencv')
                st.session_state.obs = st.session_state.env.reset()
                st.session_state.rewards = []
                st.session_state.steps = 0
                
                # Render the reset state
                img = st.session_state.env.render_opencv()
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                map_placeholder.image(img_pil, use_column_width=True)
            
            # Manual control buttons (for testing)
            st.subheader("Manual Control")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("⬅️ Turn Left"):
                    action = 1  # Turn left
                    next_obs, reward, done, _ = st.session_state.env.step(action)
                    st.session_state.obs = next_obs
                    st.session_state.steps += 1
                    
                    # Render the environment
                    img = st.session_state.env.render_opencv()
                    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    map_placeholder.image(img_pil, use_column_width=True)
            
            with col2:
                if st.button("⬆️ Forward"):
                    action = 0  # Move forward
                    next_obs, reward, done, _ = st.session_state.env.step(action)
                    st.session_state.obs = next_obs
                    st.session_state.steps += 1
                    
                    # Render the environment
                    img = st.session_state.env.render_opencv()
                    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    map_placeholder.image(img_pil, use_column_width=True)
            
            with col3:
                if st.button("➡️ Turn Right"):
                    action = 2  # Turn right
                    next_obs, reward, done, _ = st.session_state.env.step(action)
                    st.session_state.obs = next_obs
                    st.session_state.steps += 1
                    
                    # Render the environment
                    img = st.session_state.env.render_opencv()
                    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    map_placeholder.image(img_pil, use_column_width=True)
            
            # Display metrics
            st.subheader("Current Metrics")
            
            # Create 3 columns for metrics
            m1, m2, m3 = st.columns(3)
            
            # Calculate metrics
            total_reward = sum(st.session_state.rewards) if st.session_state.rewards else 0
            explored = np.sum(st.session_state.env.visited_map > 0) / (st.session_state.env.map_size ** 2) * 100
            
            with m1:
                st.metric("Total Reward", f"{total_reward:.2f}")
            with m2:
                st.metric("Steps", st.session_state.steps)
            with m3:
                st.metric("Explored", f"{explored:.1f}%")
            
            # Display exploration rate in second row
            st.metric("Exploration Rate", f"{st.session_state.agent.get_exploration_rate():.4f}")
    
    # Show training results if available
    if st.session_state.training_complete and st.session_state.performance_metrics:
        st.subheader("Training Performance")
        
        # Create tabs for different charts
        tab1, tab2, tab3 = st.tabs(["Rewards", "Exploration Rate", "Evaluation Metrics"])
        
        with tab1:
            # Create a rewards chart with dark mode style
            plt.style.use('dark_background')
            fig, ax = plt.figure(figsize=(10, 5), facecolor='#0e1117'), plt.gca()
            ax.set_facecolor('#0e1117')
            episodes = st.session_state.performance_metrics['episodes']
            rewards = st.session_state.episode_rewards
            avg_rewards = st.session_state.performance_metrics['avg_reward']
            
            ax.plot(episodes, rewards, color='#00a0fc', linestyle='-', alpha=0.3, label='Episode Reward')
            ax.plot(episodes, avg_rewards, color='#fc5a03', linestyle='-', linewidth=2, label='Average Reward (10 episodes)')
            ax.set_xlabel('Episode', color='white')
            ax.set_ylabel('Reward', color='white')
            ax.legend(facecolor='#1e1e1e', edgecolor='#333333')
            ax.grid(True, alpha=0.15)
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')
            
            # Show the plot
            st.pyplot(fig)
        
        with tab2:
            # Create an exploration rate chart with dark mode style
            plt.style.use('dark_background')
            fig, ax = plt.figure(figsize=(10, 5), facecolor='#0e1117'), plt.gca()
            ax.set_facecolor('#0e1117')
            episodes = st.session_state.performance_metrics['episodes']
            exploration = st.session_state.exploration_rates
            
            ax.plot(episodes, exploration, color='#00ff99', linestyle='-', linewidth=2)
            ax.set_xlabel('Episode', color='white')
            ax.set_ylabel('Exploration Rate', color='white')
            ax.grid(True, alpha=0.15)
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')
            
            # Show the plot
            st.pyplot(fig)
            
        with tab3:
            # Evaluation Metrics Section
            st.subheader("SLAM Evaluation")
            
            # Initialize evaluation metrics in session state if they don't exist
            if 'evaluation_metrics' not in st.session_state:
                st.session_state.evaluation_metrics = {
                    'mapping_accuracy': [],
                    'exploration_coverage': [],
                    'path_efficiency': [],
                    'collision_rate': [],
                    'steps': []
                }
                
            # Calculate current metrics
            env = st.session_state.env
            
            # 1. Mapping Accuracy (Compare detected obstacles with actual obstacles)
            # Higher number means better accuracy
            if st.session_state.steps > 0:
                # Count correctly identified obstacle cells
                true_positives = 0
                false_positives = 0
                
                # Check each cell in the map
                for x in range(env.map_size):
                    for y in range(env.map_size):
                        pos = np.array([x, y])
                        
                        # Check if this cell is marked as an obstacle in the SLAM map
                        is_detected_obstacle = env.map[x, y] > 0.8
                        
                        # Check if this cell is actually an obstacle
                        is_actual_obstacle = False
                        for (ox, oy, size) in env.obstacles:
                            if np.linalg.norm(pos - np.array([ox, oy])) < size:
                                is_actual_obstacle = True
                                break
                                
                        if is_detected_obstacle and is_actual_obstacle:
                            true_positives += 1
                        elif is_detected_obstacle and not is_actual_obstacle:
                            false_positives += 1
                
                # Calculate precision
                total_detections = true_positives + false_positives
                mapping_precision = true_positives / max(1, total_detections)
                
                # Calculate coverage (how much of the map has been visited or sensed)
                explored_cells = np.sum(env.visited_map > 0)
                total_cells = env.map_size * env.map_size
                exploration_coverage = explored_cells / total_cells
                
                # Path efficiency (distance traveled versus unique cells visited)
                path_length = len(env.path_history)
                unique_positions = len(np.unique(np.round(np.array(env.path_history), 1), axis=0))
                path_efficiency = unique_positions / max(1, path_length)
                
                # Collision rate
                st.session_state.evaluation_metrics['mapping_accuracy'].append(mapping_precision)
                st.session_state.evaluation_metrics['exploration_coverage'].append(exploration_coverage)
                st.session_state.evaluation_metrics['path_efficiency'].append(path_efficiency)
                st.session_state.evaluation_metrics['steps'].append(st.session_state.steps)
                
                # Display metrics in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Mapping Precision", f"{mapping_precision:.2%}")
                    st.metric("Path Efficiency", f"{path_efficiency:.2%}")
                
                with col2:
                    st.metric("Exploration Coverage", f"{exploration_coverage:.2%}")
                    st.metric("Steps Taken", st.session_state.steps)
                
                # Create visualization of the metrics over time
                if len(st.session_state.evaluation_metrics['steps']) > 1:
                    plt.style.use('dark_background')
                    fig, ax = plt.figure(figsize=(10, 5), facecolor='#0e1117'), plt.gca()
                    ax.set_facecolor('#0e1117')
                    
                    steps = st.session_state.evaluation_metrics['steps']
                    
                    # Plot all metrics on the same chart
                    ax.plot(steps, st.session_state.evaluation_metrics['mapping_accuracy'], 
                           color='#ff9900', linestyle='-', linewidth=2, label='Mapping Precision')
                    ax.plot(steps, st.session_state.evaluation_metrics['exploration_coverage'], 
                           color='#00ccff', linestyle='-', linewidth=2, label='Exploration Coverage')
                    ax.plot(steps, st.session_state.evaluation_metrics['path_efficiency'], 
                           color='#ff3399', linestyle='-', linewidth=2, label='Path Efficiency')
                    
                    ax.set_xlabel('Steps', color='white')
                    ax.set_ylabel('Metric Value', color='white')
                    ax.legend(facecolor='#1e1e1e', edgecolor='#333333')
                    ax.grid(True, alpha=0.15)
                    ax.tick_params(colors='white')
                    ax.set_ylim(0, 1)
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#333333')
                    
                    # Show the plot
                    st.pyplot(fig)
                    
                # Add a comparison of the actual map vs SLAM map
                st.subheader("Map Comparison")
                
                # Create side-by-side maps to compare
                plt.style.use('dark_background')
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor='#0e1117')
                fig.subplots_adjust(wspace=0.3)
                
                # Actual map (ground truth)
                ax1.set_facecolor('#0e1117')
                ax1.set_title('Actual Environment', color='white')
                
                # Create empty map
                actual_map = np.zeros((env.map_size, env.map_size))
                
                # Fill in obstacles
                for x in range(env.map_size):
                    for y in range(env.map_size):
                        for (ox, oy, size) in env.obstacles:
                            if np.linalg.norm(np.array([x, y]) - np.array([ox, oy])) < size:
                                actual_map[x, y] = 1
                
                ax1.imshow(actual_map.T, origin='lower', cmap='plasma', 
                         extent=[0, env.map_size, 0, env.map_size], alpha=0.7)
                
                # Draw rover position
                ax1.plot(env.rover_pos[0], env.rover_pos[1], 'yo', markersize=8)
                
                # Set grid and labels
                ax1.grid(True, alpha=0.2, color='gray')
                ax1.set_xlabel('X', color='white')
                ax1.set_ylabel('Y', color='white')
                ax1.tick_params(colors='white')
                
                # SLAM map (what the agent has detected)
                ax2.set_facecolor('#0e1117')
                ax2.set_title('SLAM Map', color='white')
                
                # Show the SLAM map
                ax2.imshow(env.map.T, origin='lower', cmap='plasma', 
                         extent=[0, env.map_size, 0, env.map_size], alpha=0.7)
                
                # Overlay the visited areas
                ax2.imshow(env.visited_map.T, origin='lower', cmap='viridis', 
                         extent=[0, env.map_size, 0, env.map_size], alpha=0.3)
                
                # Draw rover position and path
                path = np.array(env.path_history)
                ax2.plot(path[:, 0], path[:, 1], color='#00ff99', linestyle='-', alpha=0.7)
                ax2.plot(env.rover_pos[0], env.rover_pos[1], 'yo', markersize=8)
                
                # Set grid and labels
                ax2.grid(True, alpha=0.2, color='gray')
                ax2.set_xlabel('X', color='white')
                ax2.set_ylabel('Y', color='white')
                ax2.tick_params(colors='white')
                
                # Show the plot
                st.pyplot(fig)
            else:
                st.info("Start the simulation to see evaluation metrics.")
                
            # Explanation of metrics
            with st.expander("Metrics Explanation"):
                st.markdown("""
                ### Mapping Precision
                The percentage of detected obstacles that are actual obstacles.
                
                ### Exploration Coverage
                The percentage of the environment that has been explored by the rover.
                
                ### Path Efficiency
                Ratio of unique positions visited to total steps taken. Higher values indicate less repetitive movement.
                
                ### Map Comparison
                - **Left**: Actual environment with true obstacle positions
                - **Right**: SLAM map showing what the rover has detected and where it has explored
                """)
                
            # Reference performance benchmarks
            with st.expander("Performance Benchmarks"):
                st.markdown("""
                ### Typical Performance Values
                
                | Metric | Poor | Average | Good | Excellent |
                |--------|------|---------|------|-----------|
                | Mapping Precision | <40% | 40-60% | 60-80% | >80% |
                | Exploration Coverage | <30% | 30-50% | 50-70% | >70% |
                | Path Efficiency | <30% | 30-50% | 50-70% | >70% |
                
                These benchmarks can help evaluate how well your rover is performing.
                """)
                
            # Download evaluation data
            if len(st.session_state.evaluation_metrics['steps']) > 0:
                evaluation_data = np.column_stack((
                    st.session_state.evaluation_metrics['steps'],
                    st.session_state.evaluation_metrics['mapping_accuracy'],
                    st.session_state.evaluation_metrics['exploration_coverage'],
                    st.session_state.evaluation_metrics['path_efficiency']
                ))
                
                csv_data = "Steps,Mapping Precision,Exploration Coverage,Path Efficiency\n"
                for row in evaluation_data:
                    csv_data += f"{int(row[0])},{row[1]:.4f},{row[2]:.4f},{row[3]:.4f}\n"
                    
                st.download_button(
                    label="📥 Download Evaluation Data",
                    data=csv_data,
                    file_name="slam_evaluation_metrics.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # Run simulation if in running state
    if st.session_state.running and st.session_state.training_complete:
        # Use the trained agent to select an action
        action = st.session_state.agent.act(st.session_state.obs)
        
        # Take a step in the environment
        next_obs, reward, done, info = st.session_state.env.step(action)
        
        # Update state
        st.session_state.obs = next_obs
        st.session_state.rewards.append(reward)
        st.session_state.steps += 1
        
        # Update evaluation metrics (every 5 steps to avoid excessive updates)
        if st.session_state.steps % 5 == 0:
            env = st.session_state.env
            
            # Calculate mapping precision
            true_positives = 0
            false_positives = 0
            
            # Sample a subset of cells for efficiency
            sample_points = np.random.randint(0, env.map_size, size=(100, 2))
            
            for x, y in sample_points:
                pos = np.array([x, y])
                
                # Check if this cell is marked as an obstacle in the SLAM map
                is_detected_obstacle = env.map[x, y] > 0.8
                
                # Check if this cell is actually an obstacle
                is_actual_obstacle = False
                for (ox, oy, size) in env.obstacles:
                    if np.linalg.norm(pos - np.array([ox, oy])) < size:
                        is_actual_obstacle = True
                        break
                        
                if is_detected_obstacle and is_actual_obstacle:
                    true_positives += 1
                elif is_detected_obstacle and not is_actual_obstacle:
                    false_positives += 1
            
            # Calculate precision
            total_detections = true_positives + false_positives
            mapping_precision = true_positives / max(1, total_detections)
            
            # Exploration coverage
            explored_cells = np.sum(env.visited_map > 0)
            total_cells = env.map_size * env.map_size
            exploration_coverage = explored_cells / total_cells
            
            # Path efficiency
            path_length = len(env.path_history)
            # Round to 1 decimal place for more meaningful unique position count
            unique_positions = len(np.unique(np.round(np.array(env.path_history), 1), axis=0))
            path_efficiency = unique_positions / max(1, path_length)
            
            # Store metrics
            st.session_state.evaluation_metrics['mapping_accuracy'].append(mapping_precision)
            st.session_state.evaluation_metrics['exploration_coverage'].append(exploration_coverage)
            st.session_state.evaluation_metrics['path_efficiency'].append(path_efficiency)
            st.session_state.evaluation_metrics['steps'].append(st.session_state.steps)
        
        # Render the environment
        img = st.session_state.env.render_opencv()
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        map_placeholder.image(img_pil, use_column_width=True)
        
        # Check if episode is done
        if done:
            st.session_state.running = False
            st.success(f"Episode complete! Total reward: {sum(st.session_state.rewards):.2f}")
        
        # Add a small delay to make the simulation visible - use the configured speed
        time.sleep(st.session_state.simulation_speed)
        
        # Rerun to update the UI
        st.rerun()

if __name__ == "__main__":
    main()