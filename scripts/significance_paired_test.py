# Perform various significance tests on paired samples statistics.
# To choose between them, see: https://aclanthology.org/P18-1128/
# - The Hitchhiker’s Guide to Testing Statistical Significance in Natural Language Processing
#
# Example of usage:
#
# python3 scripts/significance_paired_test.py \
# -->r1=$1/logs/avg_closest5.$testlog.$module.results \
# -->r2=$2/logs/avg_closest5.$testlog.$module.results \
# -->test-name=wilcoxon
# -->alpha=$alpha

import argparse
import pandas as pd

TEST_NAME_TTEST = "ttest"
TEST_NAME_WILCOXON = "wilcoxon"
TEST_NAME_MCNEMAR = "mcnemar"
TEST_NAME_PERMUTATION = "permutation"

TEST_NAMES = [TEST_NAME_TTEST,
              TEST_NAME_WILCOXON,
              TEST_NAME_MCNEMAR,
              TEST_NAME_PERMUTATION
              ]

def main():
    parser = argparse.ArgumentParser(
        description='Paired t-test to compare two columns of paired floats.'
            )
    # fmt: off
    parser.add_argument(
        '--r1', required=True,
        help='File containing scores/outputs from the first system, matched by row to r2.'
        )
    parser.add_argument(
        '--r2', required=True,
        help='File containing scores/outputs from the second system, matched by row to r1.'
            )
    parser.add_argument(
        '--gold', required=False,
        help='File containing the output of reference.'
            )
    parser.add_argument(
        '--test-name', default='ttest', choices=TEST_NAMES,
        help="Which paired significance test to perform "\
            "(ttest, wilcoxon, mcnemar, permutation)"
        )
    parser.add_argument(
        '--alpha', default=0.05,
        help="Significance level of the test. Assuming that the null hypothesis is true,"\
            "If the p-value < alpha, we can reject the null hypothesis that the two model's performances are equal."
        )
    # fmt: on
    args = parser.parse_args()

    r1 = pd.read_csv(args.r1, names=['r1'])
    r2 = pd.read_csv(args.r2, names=['r2'])


    # log mean value and check for distribution symmetry
    print('SYS1:')
    m=r1.mean()
    g=r1[r1>m].count()/r1.shape[0]
    print('-->mean=%.3f' % m)
    if g.item() > 0.51 or g.item() < 0.49:
        print('-->The distribution of elements is not symmetric')
    else:
        print('-->The distribution of elements is not symmetric')
    print('-->Portion of elements greater than mean: %.3f' % g)
    # same for r2
    print('SYS2:')
    m=r2.mean()
    g=r2[r2>m].count()/r2.shape[0]
    print('-->mean=%.3f' % m)
    if g.item() > 0.51 or g.item() < 0.49:
        print('-->The distribution of elements is not symmetric')
    else:
        print('-->The distribution of elements is not symmetric')
    print('-->Portion of elements greater than mean: %.3f' % g)

    # significance testing
    if args.test_name == "ttest":
        from scipy.stats import ttest_rel
        statistic, pvalue = ttest_rel(r1[0], r2[0])

    elif args.test_name == "wilcoxon":
        from scipy.stats import wilcoxon
        statistic, pvalue = wilcoxon(x=r1['r1'], y=r2['r2'])

    elif args.test_name == "mcnemar":
        import statsmodels.api as sm
        from statsmodels.stats.contingency_tables import mcnemar
        
        df = pd.concat([r1, r2], axis=1)
        contingency_table = sm.stats.Table.from_data(df)
        values = contingency_table.table_orig.values
        if values[0][1] + values[1][0] >= 25:
            # exact=False for using the chisquare distribution as approximation
            # of the distribution of the test statistic (good for large samples)
            # the continuity correction is used
            # ref: http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/
            result = mcnemar(values, exact=False, correction=True)
        else:
            print('Outcome differences between the two systems are too small, \
                using an exact binomial test.')
            result = mcnemar(values, exact=True)
        statistic = result.statistic
        pvalue = result.pvalue
        
    elif args.test_name == "permutation":
        from scipy.stats import permutation_test
        import numpy as np
        def mystat(x, y, axis):
            return np.mean(x, axis=axis) - np.mean(y, axis=axis)
        result = permutation_test(
            data=(r1['r1'], r2['r2']),
            statistic=mystat,
            permutation_type='samples',
            vectorized=True,
            n_resamples=1000
        )
        statistic = result.statistic
        pvalue = result.pvalue
    
    else:
        raise NotImplementedError(f'The {args.test_name} is not available.')

    if pvalue > float(args.alpha):
        print('Sample means are not statistically different (fail to reject H0)')
    else:
        print('Sample means are statistically different (reject H0)')
    print('statistic=%.3f\np-value=%.3f' % (statistic, pvalue))


if __name__ == "__main__":
    main()