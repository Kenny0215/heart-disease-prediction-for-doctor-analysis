import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(results):
    print("\n--- Step 4: Generating Graph ---")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(results.keys()), y=list(results.values()), palette="magma")
    
    plt.title("AI Algorithm Accuracy Comparison")
    plt.ylabel("Accuracy Score")
    plt.xlabel("Algorithm")
    plt.ylim(0.7, 1.0) # Zoom in to see differences
    
    # Save image
    plt.savefig("final_result_graph.png")
    print("Graph saved as 'final_result_graph.png'")
    plt.show()