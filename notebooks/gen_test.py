import bilby


def gen_pars():

    priors = bilby.gw.prior.BBHPriorDict()
    priors.pop('chirp_mass')

    pars = priors.sample()

    return pars


if __name__ == "__main__":
    pars = gen_pars()
    print(pars)
