import torch
import pickle
import numpy as np
from model_for_guessing_prev import SignalModel
from train import load_and_preprocess_data, to_tensor, split_data
import csv

# Load and preprocess the data
df = load_and_preprocess_data('high_purity_data.csv')

# Split data into training and validation sets
_, val_df = split_data(df)

# Convert validation data to tensors
val_state, val_deltas, val_next_state = to_tensor(val_df)

# Load the best model
input_size = val_state.shape[1] + val_deltas.shape[1]
model = SignalModel(input_size=input_size)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Function to make predictions with an option for custom inputs
def predict_states(model, angle=None, delta=None, use_example=True, example_index=0):
    with torch.no_grad():
        if use_example:
            # Use an example from the validation set
            angle = val_state[example_index].item()
            delta = val_deltas[example_index].tolist()
            true_next_state = val_next_state[example_index].numpy()
        else:
            # Use custom inputs
            if angle is None or delta is None:
                raise ValueError("Custom angle and delta must be provided if use_example is False")

            delta = list(delta)
            delta[2] /= 360.0  # Normalize the custom delta angle
            true_next_state = None

        # Combine angle and delta as input
        input_data = torch.tensor([[angle] + delta], dtype=torch.float32)
        pred_next_state = model(input_data[:, :1], input_data[:, 1:])

        # Convert predictions to binary
        next_state_binary = (pred_next_state > 0.5).float()

    return next_state_binary.numpy()[0], true_next_state

# Function to calculate accuracy
def calculate_accuracy(pred, true):
    correct = np.sum(pred == true)
    total = len(true)
    return correct / total

# Function to evaluate the model on the validation set
def evaluate_model(model, num_examples=10):
    total_next_state_accuracy = 0

    for i in range(num_examples):
        pred_next_state, true_next_state = predict_states(model, use_example=True, example_index=i)

        if true_next_state is not None:
            next_state_accuracy = calculate_accuracy(pred_next_state, true_next_state)
            total_next_state_accuracy += next_state_accuracy

            print(f"Example {i + 1}: Next State Accuracy: {next_state_accuracy:.4f}")

    avg_next_state_accuracy = total_next_state_accuracy / num_examples
    print(f"Average Next State Accuracy: {avg_next_state_accuracy:.4f}")

# Function to get deltas for different movement categories
def get_deltas(movement_category):
    deltas = {
        "Pure Stationary": (0.0, 0.0, 0.0),
        "Counterclockwise Rotation": (2.5347506132461246e-05, -0.00010874897792313957, 10.7191602781906283),
        "Counterclockwise Rotation 2": (3.27198364008179e-05, -1.0224948875255597e-06, 20.7253657788160126),
        "Clockwise Rotation": (-8.350730688935425e-06, -6.332637439109221e-05, -10.833621935025775),
        "Counterclockwise Rotation 3": (-7.534983853605604e-06, 1.614639397201206e-05, 25.3631058232414546),
        "Clockwise Rotation 2": (6.0435132957292426e-05, -0.00014262691377920928, -10.8875862172941424),
        "Clockwise Rotation 3": (2.53906250000002e-05, -0.00025, -20.6666484446343475),
        "Left with Clockwise Rotation": (-0.009477324782409514, -0.001210719193770039, -10.4763928839271045),
        "Right with Clockwise Rotation": (0.005460340993328385, 0.0008239436619718295, -10.851013865893492),
        "Right": (0.006107361963190181, -5.061349693251532e-05, 0.019944460235080896),
        "Right with Counterclockwise Rotation": (0.01407348242811487, 0.0010071884984025545, 10.778498030092095),
        "Left with Counterclockwise Rotation": (-0.008606131160263862, 0.0005805199844780751, 15.384097845123762),
        "Left with Counterclockwise Rotation 2": (-0.0084253666954270874, -0.00011734253666954271, 15.7603462306220763)
    }

    return deltas[movement_category]

# Function to calculate the adjustment action
def calculate_adjustment_action(param_type, channel, target_value):
    index_map = {'X': 0, 'Y': 1, 'Z': 2}
    channel_index = index_map[channel]
    base_action = 6 if param_type == 'amplitude' else 0
    return base_action + channel_index * 2 + (1 if target_value > 0.5 else 0)

# Get actions to reach target
def get_actions_to_reach_target(current_components, target_components):
    actions = []
    for component, target_value in target_components.items():
        if current_components[component] != target_value:
            param_type = 'phase' if 'phase' in component else 'amplitude'
            channel = component.split('_')[-1].upper()
            actions.append(calculate_adjustment_action(param_type, channel, target_value))
    return actions

# Function to save the data to a pickle file
def save_to_pickle(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# Function to save the data to a CSV file
def save_to_csv(filename, data):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["prev_angle", "movement_category", "delta_prev_current_center_x", "delta_prev_current_center_y",
                         "delta_angle", "next_phase_value_x", "next_phase_value_y", "next_phase_value_z",
                         "next_amplitude_value_x", "next_amplitude_value_y", "next_amplitude_value_z",
                         "normalized_causing_actions", "movement_purity"])
        for row in data:
            writer.writerow([float(row[0]), row[1], float(row[2]), float(row[3]), float(row[4])] +
                            list(map(float, row[5:11])) + [list(map(float, row[11])), float(row[12])])

# Main script
if __name__ == "__main__":
    evaluate_model(model, num_examples=308)  # Evaluate the model with 308 validation examples

    all_data = []
    num_angles = 180
    angle_step = 360 / num_angles
    normalized_angle_step = 1.0 / num_angles

    movement_categories = [
        "Pure Stationary", "Counterclockwise Rotation", "Counterclockwise Rotation 2",
        "Clockwise Rotation", "Counterclockwise Rotation 3", "Clockwise Rotation 2",
        "Clockwise Rotation 3", "Left with Clockwise Rotation", "Right with Clockwise Rotation",
        "Right", "Right with Counterclockwise Rotation", "Left with Counterclockwise Rotation",
        "Left with Counterclockwise Rotation 2"
    ]

    for i in range(num_angles):
        custom_angle = i * normalized_angle_step  # normalized angle from 0 to 1
        prev_angle = i * angle_step  # denormalized angle from 0 to 360
        for movement_category in movement_categories:
            custom_delta = get_deltas(movement_category)
            pred_next_state, _ = predict_states(
                model, angle=custom_angle, delta=custom_delta, use_example=False)

            # Store predicted next state components
            next_state_components = {
                'next_phase_value_x': pred_next_state[0],
                'next_phase_value_y': pred_next_state[1],
                'next_phase_value_z': pred_next_state[2],
                'next_amplitude_value_x': pred_next_state[3],
                'next_amplitude_value_y': pred_next_state[4],
                'next_amplitude_value_z': pred_next_state[5]
            }

            actions = get_actions_to_reach_target(next_state_components, next_state_components)

            # Normalize actions
            normalized_actions = [action / 11 for action in actions]

            # Add movement_purity with a constant value of 1 (or adjust based on your logic)
            movement_purity = 1  # or calculate this based on your criteria

            # Save the data
            all_data.append([prev_angle, movement_category, custom_delta[0], custom_delta[1],
                             custom_delta[2] / 360.0] + list(pred_next_state) + [normalized_actions, movement_purity])

    # Save all data to a CSV file and a pickle file
    save_to_csv('predictions_2.csv', all_data)
    save_to_pickle('predictions.pkl', all_data)
