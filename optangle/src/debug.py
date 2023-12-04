import json
import os
import matplotlib.pyplot as plt

def validate(executor, the_test):
    executor.stats.test_generated += 1
    is_valid, validation_msg = executor.validate_test(the_test)
    the_test.set_validity(is_valid, validation_msg)              

    if is_valid:
        executor.stats.test_valid += 1
    else:
        executor.stats.test_invalid += 1
    
    return is_valid, validation_msg

def visualise(executor, the_test):
    executor.road_visualizer.visualize_road_test(the_test)

def analyse_result_features(path_to_result_folder):
    features_list = []
    for filename in os.listdir(path_to_result_folder):
        if filename.startswith("test.") and filename.endswith(".json"):
            file_path = os.path.join(path_to_result_folder, filename)
            with open(file_path) as f:
                data = json.load(f)
                if "features" in data and data["test_outcome"] == "FAIL":
                    features_list.append(data["features"])

    # all features are gathered, now analyse them
    for n in features_list[0].keys():
        x = [features[n] for features in features_list]

        plt.hist(x, bins=10)
        plt.xlabel(f"{n} Values")
        plt.ylabel("Frequency")
        plt.title(f"Frequency of {n} Values")
        save_path = os.path.join(path_to_result_folder, f"_{n}.png")
        plt.savefig(save_path)
        plt.close()
