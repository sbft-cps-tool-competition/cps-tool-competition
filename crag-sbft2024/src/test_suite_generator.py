import os.path
import subprocess


def has_m_match(array, other_arrays, m):
    """Checks if array and any one of other_arrays have at least
    m number of same elements"""
    for other_array in other_arrays:
        count = 0
        for i in range(len(array)):
            if other_array[i] == array[i]:
                count += 1
            if count >= m:
                return True
    return False

def has_m_consecutive_match(array, other_arrays, m):
    """Checks if array and any one of other_arrays have at least
    m number of same elements that appear consecutively"""
    for other_array in other_arrays:
        for i in range(len(array) - m + 1):
            if array[i:i+m] == other_array[i:i+m]:
                return True
    return False

def get_fullpath(filename):
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, filename)


class TestSuiteGenerator:
    def __init__(self, param_count, param_value_count):
        self.param_count = param_count
        self.param_value_count = param_value_count
        self.road_section_count = self.param_count // 2

        d = self.param_count // 2
        self.seed_first_line = "\t".join([f"Length{i}\tKappa{i}" for i in range(d)])
        self.seed_filepath = get_fullpath("Seed.txt")
        self.pict_executable_filepath = get_fullpath("pict")
        self.model_filepath = get_fullpath("Model.txt")

        self.create_model()

    def create_model(self):
        d = self.param_count // 2
        lines = []
        for i in range(d):
            lines.append(f"Length{i}: " + ", ".join([str(j) for j in range(self.param_value_count)]) + "\n")
            lines.append(f"Kappa{i}: " + ", ".join([str(j) for j in range(self.param_value_count)]) + "\n")

        with open(self.model_filepath, "w") as f:
            f.writelines(lines)

    def write_to_seed_file(self, test_suite):
        lines = ["\t".join([str(v) for v in test]) + "\n" for test in test_suite]
        all_lines = [self.seed_first_line + "\n"] + lines
        with open(self.seed_filepath, "w") as f:
            f.writelines(all_lines)

    def call_pict(self, n, seed_test_suite=None):
        if seed_test_suite is not None:
            self.write_to_seed_file(seed_test_suite)
            return subprocess.getoutput(f"{self.pict_executable_filepath} {self.model_filepath} /r /o:{n} /e:{self.seed_filepath}")
        else:
            return subprocess.getoutput(f"{self.pict_executable_filepath} {self.model_filepath} /r")

    def filter(self, ts, n):
        """Consecutive coverage is sufficient. Filter out rest."""
        needed = [False for t in ts]
        for i in range(self.road_section_count - n + 1):
            count = 0
            dic = {}
            # print(i)
            break_set = False
            for index, t in enumerate(ts):
                if needed[index]: # go over the already needed ones
                    part = tuple(t[i:i+n])
                    if part not in dic:
                        count += 1
                        dic[part] = True
                        needed[index] = True
                    if count == self.param_value_count ** n:
                        break_set = True
                        break
            if not break_set:
                for index, t in enumerate(ts):
                    if not needed[index]: # go over the already needed ones
                        part = tuple(t[i:i + n])
                        if part not in dic:
                            count += 1
                            dic[part] = True
                            needed[index] = True
                        if count == self.param_value_count ** n:
                            break
            # print(dic)
        new_ts = []
        for i in range(len(ts)):
            if needed[i]:
                new_ts.append(ts[i])
        return new_ts


    def get_test_suite(self, n, best_of_last_test_suite = None):
        if best_of_last_test_suite is not None and len(best_of_last_test_suite) == 0:
            best_of_last_test_suite = None
        pict_result = self.call_pict(n, best_of_last_test_suite)
        ts_n = [[int(index_str) for index_str in line.split()] for line in pict_result.splitlines()[2:]]
        print(len(ts_n))
        ts_n = self.filter(ts_n, n)
        if best_of_last_test_suite is not None:
            ts_n = [test for test in ts_n if has_m_consecutive_match(test, best_of_last_test_suite, n - 1)]
        return ts_n


if __name__ == "__main__":
    tsg = TestSuiteGenerator(8, 2)
    ts1 = tsg.get_test_suite(2)
    ts2 = tsg.get_test_suite(3, ts1)
    print(ts2)
