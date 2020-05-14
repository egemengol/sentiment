import random

from clas import ClassifierClassic, tokenize_classic, tokenize_binary
from bern import ClassifierBernoulli
from utils import docs


"""
Recall: Fraction of docs in class i classified correctly.
R = tp/(tp+fp)
Precision: Fraction of docs assigned class i that are actually about class i.
P = tp/(tp+fn)
Accuracy: Fraction of docs classified correctly.

F = 2*P*R / (P+R)
"""


def measure_metrics(marks_pos, marks_neg):
    """
    Takes two lists of [True|False|None] values,
    returns MACRO and MICRO precision, recall and F metrics.
    """
    # Measure
    trues_pos = sum(1 for b in marks_pos if b)
    falses_pos = sum(1 for b in marks_pos if b == False)
    nones_pos = sum(1 for b in marks_pos if b is None)
    # print(f"  -  Of {len(marks_pos)} real positives, {trues_pos} true, {falses_pos} false, {nones_pos} none.")

    trues_neg = sum(1 for b in marks_neg if b)
    falses_neg = sum(1 for b in marks_neg if b == False)
    nones_neg = sum(1 for b in marks_neg if b is None)
    # print(f"  -  Of {len(marks_neg)} real negatives, {trues_neg} true, {falses_neg} false, {nones_neg} none.")

    # Count individual classes
    tp_pos = trues_pos
    fp_pos = trues_neg
    tn_pos = 0
    fn_pos = falses_pos + nones_pos

    tp_neg = falses_neg
    fp_neg = falses_pos
    tn_neg = 0
    fn_neg = trues_neg + nones_neg

    # Aggregate for micro averaging
    tp_micro = tp_pos + tp_neg
    fp_micro = fp_pos + fp_neg
    tn_micro = tn_pos + tn_neg
    fn_micro = fn_pos + fn_neg

    try:
        # Macro average
        P_pos = tp_pos/(tp_pos+fn_pos)
        P_neg = tp_neg/(tp_neg+fn_neg)
        P_macro = (P_pos+P_neg)/2

        R_pos = tp_pos / (tp_pos + fp_pos)
        R_neg = tp_neg / (tp_neg + fp_neg)
        R_macro = (R_pos+R_neg)/2

        F_pos = 2 * P_pos * R_pos / (P_pos + R_pos)
        F_neg = 2 * P_neg * R_neg / (P_neg + R_neg)
        F_macro = (F_pos+F_neg)/2

        # Micro average
        P_micro = tp_micro/(tp_micro+fn_micro)
        R_micro = tp_micro/(tp_micro+fp_micro)
        F_micro = 2*P_micro*R_micro / (P_micro+R_micro)

        return {
            "P_micro": P_micro,
            "P_macro": P_macro,
            "R_micro": R_micro,
            "R_macro": R_macro,
            "F_micro": F_micro,
            "F_macro": F_macro,
        }
    except ZeroDivisionError:
        print("Division by zero occured.")


def measure(classifier, dataset: str = "test", alpha: int = 1):
    marks_pos = [classifier.is_pos(f, alpha) for f in docs(dataset, "pos")]
    marks_neg = [classifier.is_pos(f, alpha) for f in docs(dataset, "neg")]
    ms = measure_metrics(marks_pos, marks_neg)

    print(f"       | Precision | Recall | F-measure")
    print(f' Micro | {ms["P_micro"]:9.2f} | {ms["R_micro"]:6.2f} | {ms["F_micro"]:8.2f}')
    print(f' Macro | {ms["P_macro"]:9.2f} | {ms["R_macro"]:6.2f} | {ms["F_macro"]:8.2f}')
    print()

    return marks_pos, marks_neg


def randomization_test(marks_A_pos, marks_A_neg, marks_B_pos, marks_B_neg):
    """
    Takes two pairs of lists of [True|False|None] values,
    returns randomization test between two classification algorithms.
    """
    assert len(marks_A_neg) == len(marks_A_pos) == len(marks_B_neg) == len(marks_B_pos)

    ms_A = measure_metrics(marks_A_pos, marks_A_neg)
    ms_B = measure_metrics(marks_B_pos, marks_B_neg)
    S = abs(ms_A["F_macro"] - ms_B["F_macro"])

    R = 1024
    count = 0
    for _ in range(R):
        A_ps = marks_A_pos.copy()
        A_ns = marks_A_neg.copy()
        B_ps = marks_B_pos.copy()
        B_ns = marks_B_neg.copy()
        for i, _ in enumerate(A_ps):
            if random.getrandbits(1):
                A_ps[i] = not A_ps[i]
                A_ns[i] = not A_ns[i]
                B_ps[i] = not B_ps[i]
                B_ns[i] = not B_ns[i]
        ms_a = measure_metrics(A_ps, A_ns)
        ms_b = measure_metrics(B_ps, B_ns)
        S_star = abs(ms_a["F_macro"] - ms_b["F_macro"])
        if S_star > S:
            count += 1
    return (count+1)/(R+1)


print("   ===   Training all.")

clas = ClassifierClassic(tokenize_classic)
binary = ClassifierClassic(tokenize_binary)
bern = ClassifierBernoulli()

print("   ===   Classic")
clas_pos, clas_neg = measure(clas, alpha=1)
print("   ===   Binary")
bin_pos, bin_neg = measure(binary, alpha=1)
print("   ===   Bernoulli")
bern_pos, bern_neg = measure(bern, alpha=1)

print("   ===   Randomization Tests\n")

r = randomization_test(clas_pos, clas_neg, bin_pos, bin_neg)
print(f"   Classic & Binary      {r:.2f}\n")

r = randomization_test(clas_pos, clas_neg, bern_pos, bern_neg)
print(f"   Classic & Bernoulli   {r:.2f}\n")

r = randomization_test(bin_pos, bin_neg, bern_pos, bern_neg)
print(f"   Binary & Bernoulli    {r:.2f}\n")
