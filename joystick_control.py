import threading
import pygame
from helpers import (
    initialize_camera, start_display_thread, initialize_signal_generator,
    initialize_dataset, find_angle, determine_goal_and_actions,
    extract_actions_from_result, send_actions_to_signal_generator,
    stop_signal_generator
)
import numpy as np

# Initialize SignalGenerator object
sg = initialize_signal_generator()

# Define initial current components
current_components = {
    'prev_phase_value_x': 0,
    'prev_phase_value_y': 0,
    'prev_phase_value_z': 0,
    'prev_amplitude_value_x': 0,
    'prev_amplitude_value_y': 0,
    'prev_amplitude_value_z': 0
}

# Step function to handle movement category and execute actions
def step(movement_category):
    angle = find_angle()
    result = determine_goal_and_actions(angle, current_components, movement_category)
    print(f"Movement Category: {movement_category}")
    print(result)

    actions = extract_actions_from_result(result)
    print(f"Actions to reach target components: {actions}")

    send_actions_to_signal_generator(sg, actions, current_components)

# Function to initialize and listen to Xbox controller
def initialize_controller():
    pygame.init()
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Joystick initialized: {joystick.get_name()}")
    return joystick

# Function to get movement category from joystick input
def get_movement_category_from_joystick(joystick):
    pygame.event.pump()
    axis_left_x = joystick.get_axis(3)  # right stick horizontal
    axis_left_y = joystick.get_axis(1)  # Left stick vertical forward, backward
    trigger_left = joystick.get_axis(4)  # Left trigger
    trigger_right = joystick.get_axis(5)  # Right trigger

    movement_threshold = 0.1  # Threshold for detecting significant movement
    trigger_threshold = 0.5  # Threshold for detecting trigger press

    movement = None
    rotation = None

    # Determine movement direction using left stick
    if np.sqrt(axis_left_x ** 2 + axis_left_y ** 2) >= movement_threshold:
        if abs(axis_left_x) > abs(axis_left_y):
            if axis_left_x > 0:
                movement = 'Right'
            elif axis_left_x < 0:
                movement = 'Left'
        else:
            if axis_left_y > 0:
                movement = 'Backward'
            elif axis_left_y < 0:
                movement = 'Forward'

    # Determine rotation using triggers
    if trigger_left > trigger_threshold and trigger_right > trigger_threshold:
        rotation = 'Both triggers pressed'
    elif trigger_left > trigger_threshold:
        rotation = 'Anti-Clockwise'
    elif trigger_right > trigger_threshold:
        rotation = 'Clockwise'

    # Combine movement and rotation if both are detected
    if movement and rotation:
        return f"{movement} & {rotation}"
    elif movement:
        return movement
    elif rotation:
        return rotation

    # Return None if no significant movement or rotation
    return None

if __name__ == "__main__":
    dataset_file_path = 'predictions.pkl'
    initialize_dataset(dataset_file_path)
    initialize_camera()
    start_display_thread()

    joystick = initialize_controller()

    while True:
        movement_category = get_movement_category_from_joystick(joystick)
        if movement_category:  # Only proceed if a valid movement category is detected
            step(movement_category)
        pygame.time.wait(100)  # Polling delay (in milliseconds)

