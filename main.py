import test_functions
import optim
import numpy as np
import matplotlib.pyplot as plt


def main():
    test_fn = test_functions.himmelblau

    # Plotting setup
    domain_x = np.arange(-5, 5, 0.2)
    domain_y = np.arange(-5, 5, 0.2)

    X, Y = np.meshgrid(domain_x, domain_y)
    Z = test_fn(X, Y)
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z)

    # Optimiser setup
    start_point = [0, -4]
    max_steps = 10000
    # opt = optim.SGD(lr=1e-5, pos=start_point)
    opt = optim.Momentum(lr=1e-3, decay=1e-2, pos=start_point)

    loss_history = []
    loss_history.append(test_fn(opt.pos[0], opt.pos[1]))
    for i in range(max_steps):
        delta = 1e-9
        step_pos = np.asarray([opt.pos - np.asarray([delta, 0]), opt.pos - [0, delta]], dtype=np.float64)
        diff = np.asarray([
            loss_history[-1] - test_fn(step_pos[0][0], step_pos[0][1]),
            loss_history[-1] - test_fn(step_pos[1][0], step_pos[1][1])], dtype=np.float64)
        grad = diff / delta
        opt.step(grad)
        loss_history.append(test_fn(opt.pos[0], opt.pos[1]))

    x_coords, y_coords = opt.get_xy_lists()
    loss_history = np.array(loss_history)
    ax.plot3D(x_coords, y_coords, loss_history, "r", zorder=10)
    plt.show()




if __name__ == "__main__":
    main()