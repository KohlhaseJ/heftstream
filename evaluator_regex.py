import re

REGEX = '\:\ (.*)'


def get_proc_samples_accuracy_time(eval_out):
    x = re.findall(REGEX, eval_out)
    return int(x[0]), float(x[1]), float(x[4])


if __name__ == '__main__':
    text = """
    Processed samples: 1000
    Mean performance:
    heft - Accuracy     : 0.9975
    heft - Training time (s)  : 0.29
    heft - Testing time  (s)  : 0.10
    heft - Total time    (s)  : 0.39
    """

    print(get_proc_samples_accuracy_time(text))
