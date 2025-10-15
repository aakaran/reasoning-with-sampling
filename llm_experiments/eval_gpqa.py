import pandas as pd
from grader_utils.gpqa_grader import grade_answer, parse_answer_gpqa


def safe_grade(ans, correct_ans):
    try:
        return int(grade_answer(ans, correct_ans))
    except Exception:
        return 0


def eval_gpqa(fname):
    df = pd.read_csv(fname)
    base_correct = 0
    temp_correct = 0
    mcmc_correct = 0
    total = len(df)

    for i in range(total):
        base_correct += safe_grade(parse_answer_gpqa(df["std_completion"][i]), df["correct_answer"][i])
        temp_correct += safe_grade(parse_answer_gpqa(df["naive_completion"][i]), df["correct_answer"][i])
        mcmc_correct += safe_grade(parse_answer_gpqa(df["mcmc_completion"][i][len(df["question"][i]):]), df["correct_answer"][i])

    return base_correct, temp_correct, mcmc_correct, total


def gpqa_results(fnames):
    base_total = 0
    temp_total = 0
    mcmc_total = 0
    total = 0

    for fname in fnames:
        base, temp, mcmc, n = eval_gpqa(fname)
        base_total += base
        temp_total += temp
        mcmc_total += mcmc
        total += n

    base_acc = base_total / total
    temp_acc = temp_total / total
    mcmc_acc = mcmc_total / total

    print(f"Base accuracy:  {base_acc:.3f}")
    print(f"Temp accuracy:  {temp_acc:.3f}")
    print(f"MCMC accuracy:  {mcmc_acc:.3f}")

    return {
        "base_acc": base_acc,
        "temp_acc": temp_acc,
        "mcmc_acc": mcmc_acc,
    }
