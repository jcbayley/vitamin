import vitamin
import numpy as np
import matplotlib.pyplot as plt
import corner
from collections import OrderedDict
import pickle
#from vitamin import gw
import seaborn
import pandas
import scipy.stats as st
from scipy.spatial.distance import jensenshannon
import os



def make_js_comp(output_dir, allbox):
    """_summary_

    Args:
        output_dir (str): _description_
        allbox (DataFrame): _description_
    """
    fg, ax = plt.subplots(figsize=(12,6))
    meltbox = allbox.melt(id_vars=['label'], 
            value_vars=[f"p{i}" for i in range(allbox.shape[1] - 1)],
            var_name='Parameter', 
            value_name='log10 KL divergence'
            )

    kl_fg2 = seaborn.boxplot(
        data=meltbox, 
        x="Parameter",
        y="log10 KL divergence",
        hue="label"
        )
    fg.savefig(os.path.join(output_dir, "js_comp_box.png"))


def hist_js(output_dir, data, append=""):
    fg, ax = plt.subplots(figsize=(15,7))
    kl_fg = seaborn.histplot(data*10**3, ax=ax)
    kl_fg.set_xlabel("JS divergence 1e-3")
    fg.savefig(os.path.join(output_dir, f"js_{append}.png"))

def make_js_par_grid(output_dir, js_data, test_data):

    fig, ax = plt.subplots()
    pars = np.zeros((test_data[0].shape))
    meanjs = np.zeros((len(test_data[0])))
    for i, t_par in enumerate(test_data[0]):
        meanjs[i] = js_data.iloc[i].mean()
        pars[i] = [t_par[0], t_par[1]]
    ax.scatter(pars[:,0], pars[:,1], c = meanjs)

    fig.savefig(os.path.join(output_dir, f"js_par_space.png"))

def get_js_data(output_dir, test_data):

    with open(os.path.join(output_dir, "kl_vit_mc_divs.pkl"),"rb") as f:
        kls_vm = np.array(pickle.load(f))
        kls_vm = pandas.DataFrame({f"p{ind}":kls_vm[:,ind] for ind in range(len(kls_vm[0]))})
        
    with open(os.path.join(output_dir, "kl_vit_an_divs.pkl"),"rb") as f:
        kls_va = np.array(pickle.load(f))
        kls_va = pandas.DataFrame({f"p{ind}":kls_va[:,ind] for ind in range(len(kls_va[0]))})

    with open(os.path.join(output_dir, "kl_an_mc_divs.pkl"),"rb") as f:
        kls_am = np.array(pickle.load(f))
        kls_am = pandas.DataFrame({f"p{ind}":kls_am[:,ind] for ind in range(len(kls_am[0]))})

    lkls_vm = np.log10(kls_vm)
    lkls_va = np.log10(kls_va)
    lkls_am = np.log10(kls_am)

    lkls_vm["label"] = "vit_mcmc"
    lkls_va["label"] = "vit_an"
    lkls_am["label"] = "an_mcmc"

    allbox = pandas.concat([lkls_vm, lkls_va, lkls_am])
    allbox.index = allbox.index.where(~allbox.index.duplicated(), allbox.index+501)
    allbox.index = allbox.index.where(~allbox.index.duplicated(), allbox.index+1002)


    make_js_comp(output_dir, allbox)

    hist_js(output_dir, kls_vm, append="vm")
    hist_js(output_dir, kls_va, append="va")
    hist_js(output_dir, kls_am, append="am")

    if len(test_data[0][0]) == 2:
        make_js_par_grid(output_dir, kls_va, test_data)

    

def plot_test_example(output_dir, test_data):
    
    fig, ax = plt.subplots()

    for i in range(5):
        print(np.shape(test_data[1]))
        ax.plot(test_data[1][i][0], label = test_data[0][i])
    ax.legend()
    fig.savefig(os.path.join(output_dir, "example_data.png"))

def plot_posteriors(output_dir, test_data):

    samples_output = os.path.join(output_dir, "cornerplots")

    if not os.path.isdir(samples_output):
        os.makedirs(samples_output)

    with open(os.path.join(output_dir, "samples.pkl"),"rb") as f:
        vit_samples_lin = pickle.load(f)

    with open(os.path.join(output_dir, "mcmc_samples.pkl"),"rb") as f:
        mcmc_samples_lin = pickle.load(f)
        
    with open(os.path.join(output_dir,"analytic_meancov.pkl"),"rb") as f:
        an_meancov_lin = pickle.load(f)  

    num_params_lin = vit_samples_lin.shape[2]


    # vitmain samples to dataframe
    vit_dfs = []
    for ind in range(len(vit_samples_lin)):
        vit_samps_dict = {"type":["vitamin"]*len(vit_samples_lin[ind])}
        for pnum in range(num_params_lin):
            vit_samps_dict[f"p{pnum}"] = vit_samples_lin[ind][:,pnum]
        vit_dfs.append(pandas.DataFrame(vit_samps_dict))
    vit_dfs = pandas.concat(vit_dfs, axis=1,keys=(np.arange(ind)))

    # mcmc samples to dataframe
    mc_dfs = []
    for ind in range(len(mcmc_samples_lin)):
        mc_samps_dict = {"type":["mcmc"]*np.prod(np.shape(mcmc_samples_lin[0].posterior.to_array())[1:])}
        for pnum in range(num_params_lin):
            mc_samps_dict[f"p{pnum}"] = np.array(np.concatenate(np.array(getattr(mcmc_samples_lin[ind].posterior,f"p{pnum}"))))
        mc_dfs.append(pandas.DataFrame(mc_samps_dict))

    mc_dfs = pandas.concat(mc_dfs, axis=1,keys=(np.arange(ind)))

    # analytic samples to dataframe
    an_dfs = []
    for ind in range(len(vit_samples_lin)):
        m,c = an_meancov_lin[ind]
        an_mvn = st.multivariate_normal(m.reshape(-1), c)
        an_samples = an_mvn.rvs(10000)
        an_samps_dict = {"type":["analytic"]*len(an_samples)}
        for pnum in range(num_params_lin):
            an_samps_dict[f"p{pnum}"] = an_samples[:,pnum]
        an_dfs.append(pandas.DataFrame(an_samps_dict))
    an_dfs = pandas.concat(an_dfs, axis=1,keys=(np.arange(ind)))


    alldf = pandas.concat([vit_dfs,mc_dfs,an_dfs])

    alldf.index = alldf.index.where(~alldf.index.duplicated(), alldf.index+10001)
    alldf.index = alldf.index.where(~alldf.index.duplicated(), alldf.index+20001)


    for out_ind in range(len(alldf)):
        sfg = seaborn.pairplot(alldf[out_ind], corner=True, hue="type", kind="hist", diag_kind = "hist", height=4)
        for i in range(len(sfg.axes)):
            for j in range(len(sfg.axes)):
                if j>i: 
                    continue
                sfg.axes[i][j].axvline(test_data[0][out_ind][j], color="r")
                if j!=i:
                    sfg.axes[i][j].axhline(test_data[0][out_ind][i], color="r")

        sfg.savefig(os.path.join(samples_output, f"corner_{out_ind}.png"))



if __name__ == "__main__":
    output_dir = "./outputs_analytic_fulltest"

    with open(os.path.join(output_dir, "test_data.pkl"),"rb") as f:
        test_data = pickle.load(f)

    plot_test_example(output_dir, test_data)

    get_js_data(output_dir, test_data)

    plot_posteriors(output_dir, test_data)
