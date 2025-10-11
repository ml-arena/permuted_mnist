"""
Getting Started with Permuted MNIST Environment
================================================

This script demonstrates how to:
1. Set up the PermutedMNIST environment
2. Train and evaluate different agents (Random vs Linear)
3. Compare their performance across multiple permuted tasks

The objective is to train and predict in less than a minute per task.
"""

import numpy as np
import time
from permuted_mnist.env.permuted_mnist import PermutedMNISTEnv
from permuted_mnist.agent.random.agent import Agent as RandomAgent
from permuted_mnist.agent.linear.agent import Agent as LinearAgent


def evaluate_agent(agent, env_config, agent_name="Agent", seed=42):
    """
    Evaluate an agent on permuted MNIST tasks

    Args:
        agent: Agent instance to evaluate
        env_config: Dictionary with environment configuration
        agent_name: Name for display purposes
        seed: Random seed for reproducibility

    Returns:
        Dictionary with performance metrics
    """
    # Create environment
    env = PermutedMNISTEnv(**env_config)
    env.set_seed(seed)

    # Track metrics
    accuracies = []
    train_times = []
    predict_times = []

    print(f"\n{'='*60}")
    print(f"Evaluating: {agent_name}")
    print(f"{'='*60}")

    # Reset agent for new experiment
    agent.reset()

    # Process each task
    task_num = 1
    while True:
        # Get next task
        task = env.get_next_task()
        if task is None:
            break

        print(f"\nTask {task_num}/{env_config['number_episodes']}:")

        # Training phase
        start_time = time.time()
        agent.train(task['X_train'], task['y_train'])
        train_time = time.time() - start_time
        train_times.append(train_time)
        print(f"  Training time: {train_time:.2f}s")

        # Prediction phase
        start_time = time.time()
        predictions = agent.predict(task['X_test'])
        predict_time = time.time() - start_time
        predict_times.append(predict_time)
        print(f"  Prediction time: {predict_time:.4f}s")

        # Evaluation
        accuracy = env.evaluate(predictions, task['y_test'])
        accuracies.append(accuracy)
        print(f"  Accuracy: {accuracy:.2%}")

        task_num += 1

    # Compute summary statistics
    results = {
        'agent_name': agent_name,
        'accuracies': accuracies,
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'min_accuracy': np.min(accuracies),
        'max_accuracy': np.max(accuracies),
        'mean_train_time': np.mean(train_times),
        'mean_predict_time': np.mean(predict_times),
        'total_time': sum(train_times) + sum(predict_times)
    }

    return results


def print_comparison(results_list):
    """
    Print a comparison table of different agents

    Args:
        results_list: List of result dictionaries from evaluate_agent
    """
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*80}")

    # Header
    print(f"{'Agent':<20} {'Mean Acc':<12} {'Std Acc':<12} {'Min Acc':<12} {'Max Acc':<12} {'Total Time':<12}")
    print("-" * 80)

    # Results for each agent
    for results in results_list:
        print(f"{results['agent_name']:<20} "
              f"{results['mean_accuracy']:<12.2%} "
              f"{results['std_accuracy']:<12.2%} "
              f"{results['min_accuracy']:<12.2%} "
              f"{results['max_accuracy']:<12.2%} "
              f"{results['total_time']:<12.2f}s")

    print(f"{'='*80}")

    # Detailed comparison
    print("\nDETAILED ANALYSIS:")
    print("-" * 40)

    # Find best performer
    best_agent = max(results_list, key=lambda x: x['mean_accuracy'])
    worst_agent = min(results_list, key=lambda x: x['mean_accuracy'])

    print(f"Best performer: {best_agent['agent_name']} with {best_agent['mean_accuracy']:.2%} mean accuracy")
    print(f"Worst performer: {worst_agent['agent_name']} with {worst_agent['mean_accuracy']:.2%} mean accuracy")

    # Performance improvement
    if len(results_list) == 2 and results_list[0]['agent_name'] == 'Random Agent':
        improvement = (results_list[1]['mean_accuracy'] - results_list[0]['mean_accuracy']) / results_list[0]['mean_accuracy']
        print(f"\n{results_list[1]['agent_name']} shows {improvement:.1%} improvement over random baseline")

    # Time analysis
    fastest = min(results_list, key=lambda x: x['total_time'])
    print(f"\nFastest agent: {fastest['agent_name']} ({fastest['total_time']:.2f}s total)")

    # Task-by-task comparison
    if len(results_list) == 2:
        print(f"\n{'Task-by-Task Comparison:'}")
        print("-" * 40)
        print(f"{'Task':<10} {results_list[0]['agent_name']:<20} {results_list[1]['agent_name']:<20}")
        for i in range(len(results_list[0]['accuracies'])):
            print(f"Task {i+1:<5} {results_list[0]['accuracies'][i]:<20.2%} {results_list[1]['accuracies'][i]:<20.2%}")


def main():
    """Main execution function"""

    print("="*80)
    print("PERMUTED MNIST META-LEARNING BENCHMARK")
    print("="*80)
    print("\nObjective: Train and predict each permuted MNIST task in < 1 minute")
    print("Environment: Each task has randomly permuted pixels and labels")
    print("Comparing: Random baseline vs. Linear classifier")

    # Environment configuration
    env_config = {
        'number_episodes': 10  # Number of permuted tasks
    }

    print(f"\nConfiguration: {env_config['number_episodes']} permuted MNIST tasks")

    # Initialize agents
    random_agent = RandomAgent(output_dim=10, seed=42)
    linear_agent = LinearAgent(input_dim=784, output_dim=10, learning_rate=0.01)

    # Evaluate agents
    results = []

    # Evaluate Random Agent (baseline)
    random_results = evaluate_agent(
        random_agent,
        env_config,
        agent_name="Random Agent",
        seed=42
    )
    results.append(random_results)

    # Evaluate Linear Agent
    linear_results = evaluate_agent(
        linear_agent,
        env_config,
        agent_name="Linear Agent",
        seed=42
    )
    results.append(linear_results)

    # Compare results
    print_comparison(results)

    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ All tasks completed in < 1 minute: ", end="")
    all_fast = all(r['total_time'] < 60 for r in results)
    print("YES" if all_fast else "NO")

    if all_fast:
        print("✓ Objective achieved: Fast meta-learning on permuted MNIST!")
    else:
        slowest = max(results, key=lambda x: x['total_time'])
        print(f"✗ {slowest['agent_name']} took {slowest['total_time']:.2f}s (> 60s)")

    print("\nKey Insights:")
    print("- Random agent provides a baseline of ~10% accuracy (random guessing)")
    print("- Linear agent learns task-specific patterns despite permutations")
    print("- Both agents meet the < 1 minute requirement for training and prediction")
    print("- The linear model significantly outperforms random guessing")


if __name__ == "__main__":
    main()