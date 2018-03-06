#!/usr/bin/env python3
import sys
# import traceback
import numpy
import scipy.stats
# Import pytransit modules
import pytransit
from pytransit import tnseq_tools as tnseq
from pytransit import transit_tools as transit
from pytransit import norm_tools as norm
from pytransit import stat_tools as stat

__version__ = 1.1


def HDI_from_MCMC(posterior_samples, credible_mass=0.95):
    # Credit to 'user72564'
    # https://stackoverflow.com/questions/22284502/highest-posterior-density-region-and-central-credible-region
    # Computes highest density interval from a sample of representative values,
    # estimated as the shortest credible interval
    # Takes Arguments posterior_samples (samples from posterior) and credible mass (normally .95)
    sorted_points = sorted(posterior_samples)
    ciIdxInc = numpy.ceil(credible_mass * len(sorted_points)).astype('int')
    nCIs = len(sorted_points) - ciIdxInc
    ciWidth = [0] * nCIs
    for i in range(0, nCIs):
        ciWidth[i] = sorted_points[i + ciIdxInc] - sorted_points[i]
    HDImin = sorted_points[ciWidth.index(min(ciWidth))]
    HDImax = sorted_points[ciWidth.index(min(ciWidth))+ciIdxInc]
    return(HDImin, HDImax)


def sample_post(data, S, mu0, s20, k0, nu0):
    n = len(data)
    s2 = numpy.var(data, ddof=1)
    ybar = numpy.mean(data)
    kn = k0 + n
    nun = nu0 + n
    mun = (k0 * mu0 + n * ybar) / float(kn)
    s2n = (1.0 / nun) * (nu0 * s20 + (n - 1) * s2 +
                         (k0 * n / float(kn)) * numpy.power(ybar - mu0, 2))

    s2_post = 1.0/scipy.stats.gamma.rvs(nun/2.0, scale=2.0/(s2n*nun), size=S)

    # Truncated Normal since counts can't be negative
    min_mu = 0
    max_mu = 1000000
    trunc_a = (min_mu-mun)/numpy.sqrt(s2_post/float(kn))
    trunc_b = (max_mu-mun)/numpy.sqrt(s2_post/float(kn))

    mu_post = scipy.stats.truncnorm.rvs(a=trunc_a, b=trunc_b, loc=mun, scale=numpy.sqrt(s2_post/float(kn)), size=S)

    return (mu_post, s2_post)


def FWER_Bayes(X):
    ii = numpy.argsort(numpy.argsort(X))
    P_NULL = numpy.sort(X)
    W = 1 - P_NULL
    N = len(P_NULL)
    P_ALT = numpy.zeros(N)
    for i in range(N):
        P_ALT[i] = 1.0 - numpy.prod(W[:i+1])
    return P_ALT[ii]


def bFDR(X):
    N = len(X)
    ii = numpy.argsort(numpy.argsort(X))
    P_NULL = numpy.sort(X)
    P_ALT = numpy.zeros(N)
    for i in range(N):
        P_ALT[i] = numpy.mean(P_NULL[:i+1])
    return P_ALT[ii]


def classify_interaction(delta_logFC, logFC_KO, logFC_WT):
    if delta_logFC < 0:
        return "Aggravating"
    elif delta_logFC >= 0 and abs(logFC_KO) < abs(logFC_WT):
        return "Alleviating"
    elif delta_logFC >= 0 and abs(logFC_KO) > abs(logFC_WT):
        return "Suppressive"
    else:
        return "N/A"


def error(x):
    print("Error: %s" % x)


def usage():
    print("""python %s -pt <Annotation .prot_table or .gff3>  -a1 <Comma seperated Strain A Time 1 .wig files> -b1 <Comma seperated Strain B Time 1 .wig files> -a2 <Comma seperated Strain A Time 2 .wig files> -b2 <Comma seperated Strain B Time 2 .wig files> [options]

        Optional Arguments:
        -rope <float>   :=  -/+ Margin for Region of Potential Equivalency (ROPE) around a logFC of zero (i.e. defines null region of no change).
                            Default: -rope  0.5
        -s <integer>    :=  Number of samples to take for Monte Carlo estimates. Default: -s 100000
        -n <string>     :=  Normalization method. Default: -n TTR
        --nz            :=  Ignore sites that are empty (missing) in all datasets. Default: False.
        --bfdr          :=  Analyze hits using overlap of delta-logFC with ROPE and adjusted using BFDR method. Default: Binary HDI decision.
        --fwer          :=  Analyze hits using overlap of delta-logFC with ROPE and adjusted using a custom FWER method.
                            More conservative than --bfdr. Default: Binary HDI decision.
        -debug <string> :=  Saves Monter Carlo samples of given comma seperated genes (IDs) to disk.
                            Example: -debug Rv0001,Rv3910c
        -l <string>     :=  Label to use when saving files in debug mode.
        -t <string>     :=  Annotation type to use (ID by default.)
""" % sys.argv[0])


def main(args, kwargs, quite=False, jumble=False):
    missingArgs = False
    if "a1" not in kwargs:
        missingArgs = True
        error("Missing -a1 argument")
    if "a2" not in kwargs:
        missingArgs = True
        error("Missing -a2 argument")
    if "b1" not in kwargs:
        missingArgs = True
        error("Missing -b1 argument")
    if "b2" not in kwargs:
        missingArgs = True
        error("Missing -b2 argument")
    if "pt" not in kwargs:
        missingArgs = True
        error("Missing -pt argument")

    if missingArgs:
        usage()
        sys.exit()

    A_1list = kwargs["a1"].split(",")
    A_2list = kwargs["a2"].split(",")
    B_1list = kwargs["b1"].split(",")
    B_2list = kwargs["b2"].split(",")

    annotation = kwargs["pt"]
    rope = float(kwargs.get("rope", 0.5))
    S = int(kwargs.get("s", 100000))
    norm_method = kwargs.get("n", "TTR")
    label = kwargs.get("l", "debug")
    onlyNZ = kwargs.get("-nz", False)
    doBFDR = kwargs.get("-bfdr", False)
    doFWER = kwargs.get("-fwer", False)
    annotation_type = kwargs.get("t", "ID")
    DEBUG = []
    if "debug" in kwargs:
        DEBUG = kwargs["debug"].split(",")

    wiglist = A_1list + B_1list + A_2list + B_2list

    Nwig = len(wiglist)
    Na1 = len(A_1list)
    Nb1 = len(A_1list)
    Na2 = len(B_2list)
    Nb2 = len(B_2list)

    (data, position) = tnseq.get_data(wiglist)

    # ######## FILTER EMTPY SITES #########
    if onlyNZ:
        ii_good = numpy.sum(data, 0) > 0
        data = data[:, ii_good]
        position = position[ii_good]
    # #####################################

    (data, factors) = norm.normalize_data(data, norm_method, wiglist, sys.argv[1])

    if jumble:
        numpy.random.shuffle(data.flat)
        numpy.random.shuffle(data.flat)

    G_A1 = tnseq.Genes([], annotation, data=data[:Na1],
                       position=position, annotation_type=annotation_type)
    G_B1 = tnseq.Genes([], annotation, data=data[Na1:(Na1+Nb1)],
                       position=position, annotation_type=annotation_type)
    G_A2 = tnseq.Genes([], annotation, data=data[(Na1+Nb1):(Na1+Nb1+Na2)],
                       position=position, annotation_type=annotation_type)
    G_B2 = tnseq.Genes([], annotation, data=data[(Na1+Nb1+Na2):],
                       position=position, annotation_type=annotation_type)
    means_list_a1 = []
    means_list_b1 = []
    means_list_a2 = []
    means_list_b2 = []

    var_list_a1 = []
    var_list_a2 = []
    var_list_b1 = []
    var_list_b2 = []

    # Base priors on empirical observations accross genes.
    for gene in sorted(G_A1):
        if gene.n > 1:
            A1_data = G_A1[gene.orf].reads.flatten()
            B1_data = G_B1[gene.orf].reads.flatten()
            A2_data = G_A2[gene.orf].reads.flatten()
            B2_data = G_B2[gene.orf].reads.flatten()

            means_list_a1.append(numpy.mean(A1_data))
            var_list_a1.append(numpy.var(A1_data))

            means_list_b1.append(numpy.mean(B1_data))
            var_list_b1.append(numpy.var(B1_data))

            means_list_a2.append(numpy.mean(A2_data))
            var_list_a2.append(numpy.var(A2_data))

            means_list_b2.append(numpy.mean(B2_data))
            var_list_b2.append(numpy.var(B2_data))

    # Priors
    mu0_A1 = scipy.stats.trim_mean(means_list_a1, 0.01)
    mu0_B1 = scipy.stats.trim_mean(means_list_b1, 0.01)
    mu0_A2 = scipy.stats.trim_mean(means_list_a2, 0.01)
    mu0_B2 = scipy.stats.trim_mean(means_list_b2, 0.01)

    s20_A1 = scipy.stats.trim_mean(var_list_a1, 0.01)
    s20_B1 = scipy.stats.trim_mean(var_list_b1, 0.01)
    s20_A2 = scipy.stats.trim_mean(var_list_a2, 0.01)
    s20_B2 = scipy.stats.trim_mean(var_list_b2, 0.01)

    k0 = 1.0
    nu0 = 1.0

    data = []
    postprob = []

    if not quite:
        print("# Created with '%s'.  Copyright 2016-2017. Michael A. DeJesus & Thomas R. Ioerger" % (sys.argv[0]))
        print("# Version %1.2f; http://saclab.tamu.edu/essentiality/GI" % __version__)
        print("#")
        print("# python %s" % " ".join(sys.argv))
        print("# Samples = %d, k0=%1.1f, nu0=%1.1f" % (S, k0, nu0))
        print("# Mean Prior:       Variance Prior:")
        print("# mu0_A1 = %1.2f    s20_A1 = %1.1f" % (mu0_A1, s20_A1))
        print("# mu0_B1 = %1.2f    s20_B1 = %1.1f" % (mu0_B1, s20_B1))
        print("# mu0_A2 = %1.2f    s20_A2 = %1.1f" % (mu0_A2, s20_A2))
        print("# mu0_B2 = %1.2f    s20_B2 = %1.1f" % (mu0_B2, s20_B2))
        print("# ROPE:", rope)
        print("# TTR Factors:", ", ".join(["%1.4f" % x for x in numpy.array(factors).flatten()]))
    for gene in sorted(G_A1):

        if len(DEBUG) > 0:
            if gene.orf not in DEBUG:
                continue

        # Maybe my problems are due to scoping?
        muA1_post = varA1_post = numpy.ones(S)
        muB1_post = varB1_post = numpy.ones(S)
        muA2_post = varA2_post = numpy.ones(S)
        muB2_post = varB2_post = numpy.ones(S)

        if gene.n > 0:
            A1_data = G_A1[gene.orf].reads.flatten()
            B1_data = G_B1[gene.orf].reads.flatten()
            A2_data = G_A2[gene.orf].reads.flatten()
            B2_data = G_B2[gene.orf].reads.flatten()

            #            Time-1   Time-2
            #
            #  Strain-A     A       C
            #
            #  Strain-B     B       D

            try:
                muA1_post, varA1_post = sample_post(A1_data, S, mu0_A1, s20_A1, k0, nu0)
                muB1_post, varB1_post = sample_post(B1_data, S, mu0_B1, s20_B1, k0, nu0)
                muA2_post, varA2_post = sample_post(A2_data, S, mu0_A2, s20_A2, k0, nu0)
                muB2_post, varB2_post = sample_post(B2_data, S, mu0_B2, s20_B2, k0, nu0)
            except Exception as e:
                muA1_post = varA1_post = numpy.ones(S)
                muB1_post = varB1_post = numpy.ones(S)
                muA2_post = varA2_post = numpy.ones(S)
                muB2_post = varB2_post = numpy.ones(S)

            logFC_A_post = numpy.log2(muA2_post/muA1_post)
            logFC_B_post = numpy.log2(muB2_post/muB1_post)
            delta_logFC_post = logFC_B_post - logFC_A_post

            alpha = 0.05

            # Get Bounds of the HDI
            l_logFC_A, u_logFC_A = HDI_from_MCMC(logFC_A_post, 1-alpha)

            l_logFC_B, u_logFC_B = HDI_from_MCMC(logFC_B_post, 1-alpha)

            l_delta_logFC, u_delta_logFC = HDI_from_MCMC(delta_logFC_post, 1-alpha)

            mean_logFC_A = numpy.mean(logFC_A_post)
            mean_logFC_B = numpy.mean(logFC_B_post)
            mean_delta_logFC = numpy.mean(delta_logFC_post)

            # Is HDI significantly different than ROPE?
            not_HDI_overlap_bit = l_delta_logFC > rope or u_delta_logFC < -rope

            # Probability of posterior overlaping with ROPE
            probROPE = numpy.mean(numpy.logical_and(delta_logFC_post >= 0.0 - rope,
                                                    delta_logFC_post <= 0.0 + rope))

        else:
            A1_data = [0, 0]
            B1_data = [0, 0]
            A2_data = [0, 0]
            B2_data = [0, 0]

            mean_logFC_A = 0
            mean_logFC_B = 0
            mean_delta_logFC = 0
            l_logFC_A = 0
            u_logFC_A = 0
            l_logFC_B = 0
            u_logFC_B = 0
            l_delta_logFC = 0
            u_delta_logFC = 0
            probROPE = 1.0
            not_HDI_overlap_bit = False

        if numpy.isnan(l_logFC_A):
            l_logFC_A = -10
            u_logFC_A = 10
        if numpy.isnan(l_logFC_B):
            l_logFC_B = -10
            u_logFC_B = 10
        if numpy.isnan(l_delta_logFC):
            l_delta_logFC = -10
            u_delta_logFC = 10

        if DEBUG:
            out = open("%s.%s_muA1_post" % (label, gene.orf), "w")
            for x in muA1_post:
                print(x, file=out)

            out = open("%s.%s_muA2_post" % (label, gene.orf), "w")
            for x in muA2_post:
                print(x, file=out)

            out = open("%s.%s_logFC_A_post" % (label, gene.orf), "w")
            for x in logFC_A_post:
                print(x, file=out)

            out = open("%s.%s_muB1_post" % (label, gene.orf), "w")
            for x in muB1_post:
                print(x, file=out)

            out = open("%s.%s_muB2_post" % (label, gene.orf), "w")
            for x in muB2_post:
                print(x, file=out)

            out = open("%s.%s_logFC_B_post" % (label, gene.orf), "w")
            for x in logFC_A_post:
                print(x, file=out)

            out = open("%s.%s_delta_logFC_post" % (label, gene.orf), "w")
            for x in delta_logFC_post:
                print(x, file=out)

        postprob.append(probROPE)
        data.append((gene.orf, gene.name, gene.n, numpy.mean(muA1_post), numpy.mean(muA2_post),
                     numpy.mean(muB1_post), numpy.mean(muB2_post), mean_logFC_A, mean_logFC_B,
                     mean_delta_logFC, l_delta_logFC, u_delta_logFC, probROPE, not_HDI_overlap_bit))

    if doBFDR or not doFWER:
        postprob = numpy.array(postprob)
        postprob.sort()
        bfdr = numpy.cumsum(postprob)/numpy.arange(1, len(postprob)+1)
        adjusted_prob = bfdr
        adjusted_label = "BFDR"
        if doBFDR:
            data.sort(key=lambda x: x[-2])
        else:
            data.sort(key=lambda x: x[-1], reverse=True)
    elif doFWER:
        fwer = FWER_Bayes(postprob)
        fwer.sort()
        adjusted_prob = fwer
        adjusted_label = "FWER"
        data.sort(key=lambda x: x[-2])

    return (data, adjusted_prob, adjusted_label)


def print_results(args, kwargs, data, adjusted_prob, adjusted_label):
    doBFDR = kwargs.get("-bfdr", False)
    doFWER = kwargs.get("-fwer", False)

    # Write notice of classification criteria
    print("#")
    if doBFDR or doFWER:
        print("# Significant interactions are those whose adjusted probability of the delta-logFC falling within ROPE is < 0.05 (Adjusted using %s)" % adjusted_label)
    else:
        print("# Significant interactions are those genes whose delta-logFC HDI does not overlap the ROPE")
    print("#")

    # Write column names
    sys.stdout.write("#ORF\tName\tNumber of TA Sites\tMean count (Strain A Time 1)\tMean count (Strain A Time 2)\tMean count (Strain B Time 1)\tMean count (Strain B Time 2)\tMean logFC (Strain A)\tMean logFC (Strain B) \tMean delta logFC\tLower Bound delta logFC\tUpper Bound delta logFC\tProb. of delta-logFC being within ROPE\tAdjusted Probability (%s)\tIs HDI outside ROPE?\tType of Interaction\n" % adjusted_label)

    # Write gene results
    for i, row in enumerate(data):
        # 1   2    3        4                5              6               7                8            9            10              11             12            13         14
        orf, name, n, mean_muA1_post, mean_muA2_post, mean_muB1_post, mean_muB2_post, mean_logFC_A, mean_logFC_B, mean_delta_logFC, l_delta_logFC, u_delta_logFC, probROPE, not_HDI_overlap_bit = row
        type_of_interaction = "No Interaction"
        if ((doBFDR or doFWER) and adjusted_prob[i] < 0.05):
            type_of_interaction = classify_interaction(mean_delta_logFC, mean_logFC_B, mean_logFC_A)
        elif not (doBFDR or doFWER) and not_HDI_overlap_bit:
            type_of_interaction = classify_interaction(mean_delta_logFC, mean_logFC_B, mean_logFC_A)

        new_row = tuple(list(row[:-1])+[adjusted_prob[i], not_HDI_overlap_bit, type_of_interaction])
        sys.stdout.write("%s\t%s\t%d\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%1.8f\t%1.8f\t%s\t%s\n" % new_row)


if __name__ == "__main__":
    (args, kwargs) = transit.cleanargs(sys.argv[1:])
    (data, adjusted_prob, adjusted_label) = main(args, kwargs)
    print_results(args, kwargs, data, adjusted_prob, adjusted_label)
