import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

import optim
import test_functions


def main(args):
    # Set up test function first
    match args.test_fn:
        case "sphere": 
            test_fn = test_functions.Sphere()
        case "beale": 
            test_fn = test_functions.Beale()
        case "goldstein_price": 
            test_fn = test_functions.GoldsteinPrice()
        case "booth": 
            test_fn = test_functions.Booth()
        case "himmelblau": 
            test_fn = test_functions.Himmelblau()
        case "threehumpcamel":
            test_fn = test_functions.ThreeHumpCamel()
        case "rastrigin":
            test_fn = test_functions.Rastrigin()
        case "rosenbrock":
            test_fn = test_functions.Rosenbrock()
        case "ackley":
            test_fn = test_functions.Ackley()
        case _: raise NotImplementedError


    # Plotting setup
    domain_x = np.arange(test_fn.range[0], test_fn.range[1], (test_fn.range[1] - test_fn.range[0]) / 100)
    domain_y = np.arange(test_fn.range[0], test_fn.range[1], (test_fn.range[1] - test_fn.range[0]) / 100)

    X, Y = np.meshgrid(domain_x, domain_y)
    Z = test_fn(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='magma')

    # Optimiser setup
    if args.start_point:
        start_point = args.start_point
    else:
        rng = np.random.default_rng()
        start_point = rng.random(2) * (test_fn.range[1] - test_fn.range[0]) + test_fn.range[0]
    
    max_steps = 10000

    # Set up optimisers using their default arguments
    opts = []
    for opt in args.optimiser:
        match opt:
            case "sgd":
                opts.append(optim.SGD(pos=start_point))
            case "momentum":
                opts.append(optim.Momentum(pos=start_point))
            case "adagrad":
                opts.append(optim.AdaGrad(pos=start_point))
            case "rmsprop":
                opts.append(optim.RMSProp(pos=start_point))
            case "adam":
                opts.append(optim.Adam(pos=start_point))
            case _:
                raise NotImplementedError

    # opt = opts[-1]  # Temporary hack until multiple optimisers are properly implemented.
    colours = ["black", "red", "green", "orange", "pink", "brown"]

    for i, opt in enumerate(opts):
        # Run optimisation
        opt.loss_history.append(test_fn(opt.pos[0], opt.pos[1]))
        for j in range(max_steps):
            delta = 1e-9
            step_pos = np.asarray([opt.pos - np.asarray([delta, 0]), opt.pos - [0, delta]], dtype=np.float64)
            diff = np.asarray([
                opt.loss_history[-1] - test_fn(step_pos[0][0], step_pos[0][1]),
                opt.loss_history[-1] - test_fn(step_pos[1][0], step_pos[1][1])], dtype=np.float64)
            grad = diff / delta
            opt.step(grad)
            opt.loss_history.append(test_fn(opt.pos[0], opt.pos[1]))


        # Plotting
        x_coords, y_coords = opt.get_xy_lists()
        loss_history = np.array(opt.loss_history)
        ax.plot(x_coords, y_coords, loss_history, zorder=10, color = colours[i % len(colours)], label=opt.__class__.__name__)

    ax.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Optimiser Playground",
        description="Demo for different optimisers, implemented in numpy, with choice of test functions.",
    )
    parser.add_argument("-o", "--optimiser", 
                        choices=[
                            "sgd",
                            "momentum",
                            "adagrad",
                            "rmsprop",
                            "adam"
                            ],
                        default=["sgd"],
                        type=str.lower,
                        nargs='+',
                        help="Which optimiser(s) to use.",
                        )
    parser.add_argument("-f", "--test_fn",
                        choices=[
                            "sphere",
                            "beale",
                            "goldstein_price",
                            "booth",
                            "himmelblau",
                            "threehumpcamel",
                            "rastrigin",
                            "rosenbrock",
                            "ackley",
                            ],
                        default="himmelblau",
                        type=str.lower,
                        help="What demo function to use.",
                        )
    parser.add_argument("--start_point", type=float, nargs=2)

    args = parser.parse_args()
    main(args)