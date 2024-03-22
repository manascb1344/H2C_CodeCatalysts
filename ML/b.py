from flask import Flask, request, jsonify
import pandas as pd
import torch
import networkx as nx
import matplotlib.pyplot as plt

# Define your Flask app
app = Flask(__name__)

# Load df_merge from CSV
df_merge = pd.read_csv("df_merge.csv")  # Adjust the file path as needed

# Define your model loading and prediction logic
# Assuming m1, data_train, args, classified_illicit_idx, classified_licit_idx, output are defined already
def predict_graph(time_period):
    # Fetch list of edges for the given time period
    sub_node_list = df_merge.index[df_merge.loc[:, 1] == time_period].tolist()
    edge_tuples = []
    for row in data_train.cpu().edge_index.view(-1, 2).numpy():
        if (row[0] in sub_node_list) or (row[1] in sub_node_list):
            edge_tuples.append(tuple(row))

    # Fetch predicted results for that time period
    node_color = []
    for node_id in sub_node_list:
        if node_id in classified_illicit_idx:
            label = "red"  # fraud
        elif node_id in classified_licit_idx:
            label = "green"  # not fraud
        else:
            if output['pred_labels'][node_id]:
                label = "orange"  # Predicted fraud
            else:
                label = "blue"  # Not fraud predicted
        node_color.append(label)

    # Setup networkx graph
    G = nx.Graph()
    G.add_edges_from(edge_tuples)

    # Plot the graph
    plt.figure(figsize=(16, 16))
    plt.title("Time period:" + str(time_period))
    nx.draw_networkx(G, nodelist=sub_node_list, node_color=node_color, node_size=6, with_labels=False)

    # Save the plot to a file or memory buffer if you want to return the image
    # plt.savefig('graph.png')  # Uncomment this line if you want to save the plot as an image
    # plt.close()  # Uncomment this line if you want to close the plot after saving it

    # Return the plot or any other desired output
    # return 'graph.png'  # If you saved the plot as an image
    return plt  # If you want to return the plot object


# Define a route for your API
@app.route('/plot_network', methods=['POST'])
def plot_network():
    # Get timestep from request body
    timestep = request.json.get('timestep')

    # Check if timestep is provided
    if timestep is None:
        return jsonify({'error': 'Please provide timestep in the request body'}), 400

    # Call the predict_graph function with the provided timestep
    # If you saved the plot as an image, you can return the filename here
    # plot_filename = predict_graph(timestep)
    # return jsonify({'plot_filename': plot_filename}), 200

    # If you want to return the plot object directly in JSON response
    plot = predict_graph(timestep)
    # Convert plot to base64 or any other suitable format if needed
    # Then return it in the JSON response
    # For demonstration purposes, let's return a message
    return jsonify({'message': 'Graph plotted successfully for timestep: ' + str(timestep)}), 200


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
