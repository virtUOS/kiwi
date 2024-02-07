# Define a mapping for nested menus
options = {
    "Konversation": {
        "Assistant": "You are a friendly and helpful assistant."
    },
    "Virtuelles Büro": {
        "Team": {
            "Programming Expert": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair.",
            "Finances Expert": "You are a financial guru, capable of providing expert advice on personal finance, budgeting, and investments."
        }
    },
    "Lernraum": {
        "Wiss. Arbeiten": {},
        "Organisation": {},
        "Kreativität": {}
    }
}


def get_final_description(selected_path):
    """
    Navigate through the options based on the selected path,
    returning the final description or None if not a final option.

    :param selected_path: List of strings representing the selected menu path.
    :return: The description string if a final option is reached, None otherwise.
    """
    current_option = options
    for step in selected_path:
        # Navigate deeper if the next step exists and is a dictionary
        if step in current_option and isinstance(current_option[step], dict):
            current_option = current_option[step]
        elif step in current_option:  # Final step with a description
            return current_option[step]
        else:
            return None  # Path does not exist
    # Return None if the final option hasn't been reached or doesn't have a description
    return None


def path_changed(current_path, previous_path_serialized):
    """
    Check if the current selected path has changed compared to the previously serialized path.

    :param current_path: List of currently selected path elements.
    :param previous_path_serialized: Previously stored serialized path.
    :return: Boolean indicating whether the path has changed.
    """
    current_path_serialized = '/'.join(current_path)  # Serialize the current path for comparison
    return current_path_serialized != previous_path_serialized