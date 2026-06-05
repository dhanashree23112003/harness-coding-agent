from calculator import divide


def compute_ratio(total, count):
    return divide(total, count)


def compute_percentage(part, whole):
    ratio = divide(part, whole)
    return ratio * 100
