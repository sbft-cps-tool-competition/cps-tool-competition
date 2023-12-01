def validate(executor, the_test):
    executor.stats.test_generated += 1
    is_valid, validation_msg = executor.validate_test(the_test)
    the_test.set_validity(is_valid, validation_msg)              

    if is_valid:
        executor.stats.test_valid += 1
    else:
        executor.stats.test_invalid += 1
    
    return is_valid

def visualise(executor, the_test):
    executor.road_visualizer.visualize_road_test(the_test)