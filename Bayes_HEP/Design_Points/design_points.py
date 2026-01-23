import numpy as np
import bilby


def latin_hypercube_sampling(dimensions, samples, seed):
        np.random.seed(seed)
        result = np.empty((samples, dimensions))
        for i in range(dimensions):
            points = np.linspace(0, 1, samples, endpoint=False) + np.random.rand(samples) / samples
            np.random.shuffle(points)
            result[:, i] = points

        return result

def detmax(A, n, max_iter=1000, tol=1e-6):
    N = A.shape[0]

    # initialize
    initidx = np.random.choice(N, size=n, replace=False)

    inidx = initidx.copy()
    outidx = np.setdiff1d(np.arange(N), inidx)

    for iter in range(max_iter):        
        B = A[inidx]
        W, U = np.linalg.eigh(B.T @ B)
        Bh = U / np.sqrt(W) @ U.T
        
        logdet = np.log(W).sum()
        plus2  = ((A[outidx] @ Bh)**2).sum(1)
        minus2 = ((A[inidx] @ Bh)**2).sum(1)
        pm2 = (A[inidx] @ Bh @ Bh.T @ A[outidx].T) ** 2

        delta = pm2 - np.outer(minus2, 1 + plus2) + plus2

        maxdelta_ind = np.argmax(delta)
        maxdelta_loc = (maxdelta_ind // (N - n), maxdelta_ind % (N - n))

        if delta[maxdelta_loc] < tol:
            print('maxdelta = {:.3E}, falling below tolerance of {:.3E} at '\
                  'iteration {:d}'.format(delta[maxdelta_loc], tol, iter))
            break
        else:
            whichout = outidx[maxdelta_loc[1]]
            whichin = inidx[maxdelta_loc[0]]
            
            # Swap the indices
            inidx[maxdelta_loc[0]] = whichout
            outidx[maxdelta_loc[1]] = whichin

    if iter >= max_iter - 1:
        print('maxdelta = {:.3E}, reached iteration {:d}'.format(delta[maxdelta_loc], iter))

    B = A[inidx]
    return B, initidx, inidx

def get_design(n_samples, priors, seed):
    lhs_samples = latin_hypercube_sampling(dimensions=len(priors), samples=n_samples, seed=seed)

    scaled_samples = np.zeros_like(lhs_samples)
    for i, key in enumerate(priors.keys()):
        xmin = priors[key].minimum
        xmax = priors[key].maximum
        scaled_samples[:, i] = xmin + lhs_samples[:, i] * (xmax - xmin)

    return scaled_samples


def load_data(train_size, validation_size, design_points, priors, seed):

    if design_points is None:
        n_samples = train_size + validation_size
        lhs_samples = latin_hypercube_sampling(dimensions=len(priors), samples=n_samples, seed=seed)

        scaled_samples = np.zeros_like(lhs_samples)
        for i, key in enumerate(priors.keys()):
            xmin = priors[key].minimum
            xmax = priors[key].maximum
            scaled_samples[:, i] = xmin + lhs_samples[:, i] * (xmax - xmin)

    else:
        scaled_samples = np.array(design_points)
        #train_size = int(len(scaled_samples) * (train_size * 0.01)) 
        #validation_size = int(len(scaled_samples) * (validation_size * 0.01)) 

    #Using DETMX
    _, initidx, inidx = detmax(scaled_samples, train_size)

           
    train_indices = np.array(inidx)
    remaining_indices = np.setdiff1d(np.arange(len(scaled_samples)), train_indices)
    validation_indices = remaining_indices[:validation_size]
    
    train_points = scaled_samples[train_indices]
    validation_points = scaled_samples[validation_indices]

    # Return the indices along with the points
    return train_points, validation_points, train_indices, validation_indices

def get_prior(RawDesign):

    # Map string names to Bilby prior classes
    prior_type_map = {
        "Linear": bilby.core.prior.Uniform,
        "Log": bilby.core.prior.LogUniform,
        "Gaussian": bilby.core.prior.Normal,
        "TruncatedGaussian": bilby.core.prior.TruncatedGaussian,
        "Delta": bilby.core.prior.DeltaFunction,
        "PowerLaw": bilby.core.prior.PowerLaw,
        # Add more mappings as needed
    }

    priors = {}

    for param in RawDesign['Parameter']:
        key = f"{param}:"
        if key in RawDesign:
            dist_type, *range_vals = RawDesign[key]
            range_vals = [val.strip('[],') for val in range_vals]
            range_vals = list(map(float, range_vals))

            if dist_type in prior_type_map:
                PriorClass = prior_type_map[dist_type]

                # Handle arguments per prior type
                if dist_type in ("Linear", "Log", "PowerLaw"):
                    priors[param] = PriorClass(
                        minimum=range_vals[0], maximum=range_vals[1], name=param
                    )
                elif dist_type == "Gaussian":
                    priors[param] = PriorClass(
                        mu=range_vals[0], sigma=range_vals[1], name=param
                    )
                elif dist_type == "TruncatedGaussian":
                    priors[param] = PriorClass(
                        mu=range_vals[0], sigma=range_vals[1],
                        minimum=range_vals[2], maximum=range_vals[3], name=param
                    )
                elif dist_type == "Delta":
                    priors[param] = PriorClass(
                        peak=range_vals[0], name=param
                    )
            else:
                raise ValueError(f"Unsupported prior type '{dist_type}' for parameter '{param}'")
    
    parameter_names = list(priors.keys())
    dim = len(parameter_names) 
    
    return priors, parameter_names, dim

def generate_param_tag(param_names, values):
    return '_'.join(f"{name}_{value:.6g}" for name, value in zip(param_names, values))
