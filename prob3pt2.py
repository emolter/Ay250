import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner


def model(x,m,b):
    return m*x + b

def fakedata(sigma,sigmax,npts):
    x = np.arange(-10,10+20/npts,20/npts)
    x = np.random.normal(loc=x,scale=sigmax)
    y = model(x,m,b)
    y = np.random.normal(loc=y,scale=sigma)
    yerr = sigma*np.random.randn(npts)
    return x,y,yerr

m = 5.
b = -2.
sigma = 5.
sigmax = 3
npts = 10.

ndim, nwalkers = 2, 10

x,y,yerr = fakedata(sigma,sigmax,npts)

def lnlike(theta,x,y,sigma):
    return -0.5 * np.sum((y - model(x,theta[0],theta[1]))**2/(sigma**2) + np.log(2.*np.pi*sigma**2))

def lnprior(theta):
    if -20 < theta[0] < 20 and -20 < theta[1] < 20:
        return 0.0
    else:
        return -np.inf

def lnprob(theta,x,y,yerr):
    return lnprior(theta) + lnlike(theta,x,y,sigma)

#randomize initial position of walkers
p0 = [1e-4*np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x,y,yerr))
sampler.run_mcmc(p0, 1000)


fig = corner.corner(sampler.flatchain, labels=['$m$', '$b$'])
plt.show()

print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))
                
[plt.plot(sampler.lnprobability[i,:]) for i in range(nwalkers)]
plt.xlabel('step number')
plt.ylabel('lnP')
plt.show()

#plots in m
[plt.plot(sampler.chain[i,:,0]) for i in range(nwalkers)]
#[plt.plot(sampler.chain[i,:,0]) for i in range(sampler.chain.shape[0])]
plt.xlabel('step')
plt.ylabel('m')
plt.show()

[plt.scatter(sampler.chain[i,:,0], sampler.lnprobability[i,:]) for i in range(1)]
plt.xlabel('m')
plt.ylabel('lnp')
plt.show()

#plots in b
[plt.plot(sampler.chain[i,:,1]) for i in range(nwalkers)]
#[plt.plot(sampler.chain[i,:,0]) for i in range(sampler.chain.shape[0])]
plt.xlabel('step')
plt.ylabel('b')
plt.show()

[plt.scatter(sampler.chain[i,:,1], sampler.lnprobability[i,:]) for i in range(1)]
plt.xlabel('b')
plt.ylabel('lnp')
plt.show()



cutchain = sampler.chain[:,200:,:]
chainm = cutchain[:,:,0]
chainb = cutchain[:,:,1]
chainmplot = chainm.flatten()
chainbplot = chainb.flatten()

if len(chainmplot) > 100:
    erase = np.random.choice(range(len(chainmplot)), len(chainmplot)-100, replace=False)
    chainmplot = np.delete(chainmplot,erase,0)
    chainbplot = np.delete(chainbplot,erase,0)    
    
fig = plt.figure(figsize=(10,8))

xstd = np.arange(-20,22,20)
for i in range(len(chainmplot)):
    (mi,bi) = chainmplot[i],chainbplot[i]
    plt.plot(xstd,model(xstd,mi,bi),alpha=0.1,color='k')

data = plt.errorbar(x,y,yerr=sigma,linestyle='None',marker = '.')
real, = plt.plot(xstd,model(xstd,m,b),color='r')
plt.legend([real,data],['Actual','Data'],loc='upper left')    

plt.xlim([-15,15])
plt.ylabel('y')
plt.xlabel('x')

plt.show()
plt.close()