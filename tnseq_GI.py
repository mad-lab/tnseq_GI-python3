import sys
import traceback
import operator
import numpy
import scipy.stats
import pytransit.transit_tools as transit_tools
import pytransit.norm_tools as norm_tools
import pytransit.stat_tools as stat_tools
import pytransit.tnseq_tools as tnseq_tools


__version__ = 1.00

def HDI_from_MCMC(posterior_samples, credible_mass=0.95):
    # Credit to 'user72564'
    # https://stackoverflow.com/questions/22284502/highest-posterior-density-region-and-central-credible-region
    # Computes highest density interval from a sample of representative values,
    # estimated as the shortest credible interval
    # Takes Arguments posterior_samples (samples from posterior) and credible mass (normally .95)
    sorted_points = sorted(posterior_samples)
    ciIdxInc = numpy.ceil(credible_mass * len(sorted_points)).astype('int')
    nCIs = len(sorted_points) - ciIdxInc
    ciWidth = [0]*nCIs
    for i in range(0, nCIs):
        ciWidth[i] = sorted_points[i + ciIdxInc] - sorted_points[i]
    HDImin = sorted_points[ciWidth.index(min(ciWidth))]
    HDImax = sorted_points[ciWidth.index(min(ciWidth))+ciIdxInc]
    return(HDImin, HDImax)


def sample_post(data, S, mu0, s20, k0, nu0):
    n = len(data)
    s2 = numpy.var(data,ddof=0)
    if numpy.isnan(s2):
        s2 = 0.0
    ybar = numpy.mean(data)

    kn = k0+n
    nun = nu0+n
    mun = (k0*mu0 + n*ybar)/float(kn)
    s2n = (1.0/nun) * (nu0*s20 + (n-1)*s2 + (k0*n/float(kn))*numpy.power(ybar-mu0,2))

    s2_post = 1.0/scipy.stats.gamma.rvs(nun/2.0, scale=2.0/(s2n*nun), size=S)
    mu_post = scipy.stats.norm.rvs(mun, numpy.sqrt(s2_post/float(kn)), size=S)

    return (mu_post, s2_post)


def usage():
    return """python %s -wt1 <comma-separated wig files> -wt2 <comma-separated wig files> -ko1 <comma-separated wig files> -ko2 <comma-separated wig files> -pt <path to annotation> [-rope <+/- rope window>, -s <samples> --debug]""" % (sys.argv[0])


(args, kwargs) = transit_tools.cleanargs(sys.argv)


missingArgs = False
if "wt1" not in kwargs:
    missingArgs = True
    print "Error: Missing -wt1 argument."
if "wt2" not in kwargs:
    missingArgs = True
    print "Error: Missing -wt2 argument."
if "ko1" not in kwargs:
    missingArgs = True
    print "Error: Missing -ko1 argument."
if "ko2" not in kwargs:
    missingArgs = True
    print "Error: Missing -ko2 argument."
if "pt" not in kwargs:
    missingArgs = True
    print "Error: Missing -pt argument."


if missingArgs:
    print usage()
    sys.exit()



wt_0list = kwargs["wt1"].split(",")
wt_32list = kwargs["wt2"].split(",")
ko_0list = kwargs["ko1"].split(",")
ko_32list = kwargs["ko2"].split(",")
wiglist = wt_0list + ko_0list + wt_32list + ko_32list

Nwig = len(wiglist)
Nwt0 = len(wt_0list)
Nko0 = len(ko_0list)
Nwt32 = len(wt_32list)
Nko32 = len(ko_32list)

rope = float(kwargs.get("rope", 0.5))
S = int(kwargs.get("s", 20000))
DEBUG = []
if "-debug" in kwargs:
    DEBUG = kwargs["-debug"].split(",") 


annotation =  kwargs["pt"]

hash = transit_tools.get_pos_hash(annotation)
orf2info = transit_tools.get_gene_info(annotation)

GENES = tnseq_tools.Genes(wiglist, annotation, norm="TTR")


# PRIORS
mu0=numpy.mean(GENES.data)
s20=1.0
k0=1.0
nu0=2.0
HDI_alpha = 0.05


data = []
postprob = []
print "# Copyright 2016. Michael A. DeJesus & Thomas R. Ioerger"
print "# Version %1.2f; http://saclab.tamu.edu/essentiality/GI" % __version__
print "#"
print "# python %s" % " ".join(sys.argv)
print "# mu0=%1.2f, S=%d, s20=%1.1f, k0=%1.1f, nu0=%1.1f" % (mu0, S, s20, k0, nu0)
print "# ROPE:", rope
for gene in GENES:


    if DEBUG and gene.orf not in DEBUG: continue

    all_data = gene.reads 

    if all_data.size > 0:

        try:
            wt0_data = all_data[0:Nwt0,:].flatten()
            ko0_data = all_data[Nwt0:(Nwt0+Nko0),:].flatten()
            wt32_data = all_data[(Nwt0+Nko0):(Nwt0+Nko0+Nwt32),:].flatten()
            ko32_data = all_data[(Nwt0+Nko0+Nwt32):,:].flatten()

            #       0    32
            #
            #  wt   A    C
            #
            #  ko   B    D

            muA_post, varA_post = sample_post(wt0_data, S, mu0, s20, k0, nu0)
            muB_post, varB_post = sample_post(ko0_data, S, mu0, s20, k0, nu0)
            muC_post, varC_post = sample_post(wt32_data, S, mu0, s20, k0, nu0)
            muD_post, varD_post = sample_post(ko32_data, S, mu0, s20, k0, nu0)

            varAC_post = varA_post + varC_post
            varBD_post = varB_post + varD_post
            varBDAC_post = varAC_post + varBD_post

            muA_post[muA_post<=0] = 0.001
            muB_post[muB_post<=0] = 0.001
            muC_post[muC_post<=0] = 0.001
            muD_post[muD_post<=0] = 0.001

            muAC_post = numpy.log2(muC_post/muA_post)
            muBD_post = numpy.log2(muD_post/muB_post)
            muBDAC_post = muBD_post - muAC_post

    
            l_AC, u_AC = HDI_from_MCMC(muAC_post, 1.0-HDI_alpha)

            l_BD, u_BD = HDI_from_MCMC(muBD_post, 1.0-HDI_alpha)

            l_BDAC, u_BDAC = HDI_from_MCMC(muBDAC_post, 1.0-HDI_alpha)
        

            mu_AC = numpy.mean(muAC_post)
            mu_BD = numpy.mean(muBD_post)
            mu_BDAC = numpy.mean(muBDAC_post)
        

            postprobBDAC = min(numpy.mean(muBDAC_post<=0), numpy.mean(muBDAC_post>=0))
            probROPE = numpy.mean(numpy.logical_and(muBDAC_post>=0.0-rope,  muBDAC_post<=0.0+rope))


        except Exception as e:

            print "Encountered the following Exception:"
            traceback.print_exc()
            print "Quitting."
            sys.exit()

    else:
        wt0_data = [0]
        ko0_data = [0]
        wt32_data = [0]
        ko32_data = [0]

        mu_AC = 0
        mu_BD = 0
        mu_BDAC = 0
        l_AC = 0
        u_AC = 0
        l_BD = 0
        u_BD = 0
        l_BDAC = 0 
        u_BDAC = 0
        postprobBDAC = 1.0
        probROPE = 1.0
        

    if numpy.isnan(l_AC):
        l_AC = -10
        u_AC = 10
    if numpy.isnan(l_BD):
        l_BD = -10
        u_BD = 10
    if numpy.isnan(l_BDAC):
        l_BDAC = -10
        u_BDAC = 10


    bit_AC = not (l_AC <= 1.0 <= u_AC)
    bit_BD = not (l_BD <= 1.0 <= u_BD)
    bit_BDAC = not (l_BDAC <= 0.0 <= u_BDAC)
    bit_BDAC_ROPE = (u_BDAC < 0.0-rope) or (0.0+rope < l_BDAC)

    n = len(all_data)

    if DEBUG:

        out = open("dump.%s_muA_normlog_post" % gene.orf, "w")
        for x in muA_post:
            print >> out, x

        out = open("dump.%s_muC_normlog_post" % gene.orf, "w")
        for x in muC_post:
            print >> out, x

        out = open("dump.%s_muAC_normlog_post" % gene.orf, "w")
        for x in muAC_post:
            print >> out, x        


        out = open("dump.%s_muB_normlog_post" % gene.orf, "w")
        for x in muB_post:
            print >> out, x

        out = open("dump.%s_muD_normlog_post" % gene.orf, "w")
        for x in muD_post:
            print >> out, x

        out = open("dump.%s_muBD_normlog_post" % gene.orf, "w")
        for x in muBD_post:
            print >> out, x


        out = open("dump.%s_muBDAC_normlog_post" % gene.orf, "w")
        for x in muBDAC_post:
            print >> out, x



    data.append((gene.orf, gene.name, gene.desc, gene.n, numpy.mean(wt0_data), numpy.mean(wt32_data), numpy.mean(ko0_data), numpy.mean(ko32_data), mu_AC, mu_BD, mu_BDAC, l_BDAC, u_BDAC, bit_BDAC_ROPE == 1))
    

data.sort(key = operator.itemgetter(-1, -4), reverse=True)
print "#ORF\tName\tDescription\tN\tMean WT-1\tMean WT-2\tMean KO-1\tMean KO-2\tMean logFC WT-2/WT-1\tMean log FC KO-2/KO-1\tMean delta logFC\tL. Bound\tU. Bound\tOutside of HDI?"
for row in data:
    print "%s\t%s\t%s\t%d\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%s" % row



                


