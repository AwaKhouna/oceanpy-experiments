import json
import matplotlib.pyplot as plt
import numpy as np
from parameters import N_ESTIMATORS, MAX_DEPTHS


def load_times(
    filename: str,
    n_estimators: int | None = None,
    max_depth: int | None = 0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load JSON and return two lists: cp_times, mip_times,
    one entry per instance.
    """
    with open(filename, "r") as f:
        data = json.load(f)
    if n_estimators is not None or max_depth != 0 or seed is not None:
        data = [
            item
            for item in data
            if (item["n_estimators"] == n_estimators if n_estimators else True)
            and (item["max_depth"] == max_depth if max_depth != 0 else True)
            and (item["seed"] == seed if seed else True)
        ]
    cp_times = []
    mip_times = []
    print(
        f"Loaded {len(data)} instances for n_estimators={n_estimators}, max_depth={max_depth}, seed={seed}"
    )
    for item in data:
        cp_list = item["explanations"]["cp"]
        mip_list = item["explanations"]["mip"]
        if len(cp_list) != len(mip_list):
            raise ValueError(
                f"Instance count mismatch: cp has {len(cp_list)}, mip has {len(mip_list)}"
            )
        for cp_e, mip_e in zip(cp_list, mip_list):
            cp_times.append(cp_e["time"])
            mip_times.append(mip_e["time"])
    return np.array(cp_times), np.array(mip_times)


def load_callbacks(
    filename: str,
    n_estimators: int = None,
    max_depth: int = None,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load JSON and return two lists: cp_distances, mip_distances,
    one entry per instance.
    """
    with open(filename, "r") as f:
        data = json.load(f)
    if n_estimators is not None or max_depth is not None or seed is not None:
        data = [
            item
            for item in data
            if (item["n_estimators"] == n_estimators if n_estimators else True)
            and (item["max_depth"] == max_depth if max_depth else True)
            and (item["seed"] == seed if seed else True)
        ]
    cp_callbacks = []
    mip_callbacks = []
    for item in data:
        cp_list = item["explanations"]["cp"]
        mip_list = item["explanations"]["mip"]
        if len(cp_list) != len(mip_list):
            raise ValueError(
                f"Instance count mismatch: cp has {len(cp_list)}, mip has {len(mip_list)}"
            )
        for cp_e, mip_e in zip(cp_list, mip_list):
            cp_cb = cp_e.get("callback", [])
            mip_cb = mip_e.get("callback", [])
            cp_cb.append(
                {"objective_value": cp_cb[-1]["objective_value"], "time": cp_e["time"]}
            )
            mip_cb.append(
                {
                    "objective_value": mip_cb[-1]["objective_value"],
                    "time": mip_e["time"],
                }
            )
            cp_callbacks.append(cp_cb)
            mip_callbacks.append(mip_cb)
            if cp_e["status"] != "OPTIMAL":
                print(f"CP not optimal for instance {item['n_estimators']}")
            if mip_e["status"] != 2:
                print(f"MIP not optimal for instance {item['n_estimators']}")
            if mip_e["status"] == 2 and cp_e["status"] == "OPTIMAL":
                if not np.isclose(
                    mip_cb[-1]["objective_value"], cp_cb[-1]["objective_value"]
                ):
                    print(
                        f"\tObjective values differ for instance {item['n_estimators']}: "
                        f"CP={cp_cb[-1]['objective_value']}, MIP={mip_cb[-1]['objective_value']}"
                    )
    return cp_callbacks, mip_callbacks


def aggregate_callbacks(
    cp_callbacks: list[dict],
    mip_callbacks: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize the distances for each instance and interpolate them on a common time grid.
    and return:
        - cp_mean_distances 1D : shape (,T)
        - cp_std_distances 1D : shape (,T)
        - mip_mean_distances 1D : shape (,T)
        - mip_std_distances 1D : shape (,T)
        - common_times 1D : shape (,T)
    """
    cp_distances = []
    mip_distances = []
    all_times = set()
    for cp_cb, mip_cb in zip(cp_callbacks, mip_callbacks):
        cp_times = [entry["time"] for entry in cp_cb]
        mip_times = [entry["time"] for entry in mip_cb]
        all_times.update(cp_times)
        all_times.update(mip_times)
    common_times = np.sort(np.array(list(all_times)))
    for cp_cb, mip_cb in zip(cp_callbacks, mip_callbacks):
        cp_distances.append([entry["objective_value"] for entry in cp_cb])
        mip_distances.append([entry["objective_value"] for entry in mip_cb])
    cp_interp = []
    mip_interp = []
    for cp_cb, mip_cb in zip(cp_callbacks, mip_callbacks):
        cp_times = [entry["time"] for entry in cp_cb]
        mip_times = [entry["time"] for entry in mip_cb]
        cp_values = [entry["objective_value"] for entry in cp_cb]
        mip_values = [entry["objective_value"] for entry in mip_cb]
        cp_interp.append(np.interp(common_times, cp_times, cp_values))
        mip_interp.append(np.interp(common_times, mip_times, mip_values))
    cp_interp = np.array(cp_interp)
    mip_interp = np.array(mip_interp)
    # normalize cp_interp and mip_interp
    cp_interp = cp_interp - cp_interp.min(axis=1, keepdims=True)
    mip_interp = mip_interp - mip_interp.min(axis=1, keepdims=True)
    cp_interp /= cp_interp.max(axis=1, keepdims=True) + 1e-10
    mip_interp /= mip_interp.max(axis=1, keepdims=True) + 1e-10
    cp_mean_distances = np.mean(cp_interp, axis=0)
    cp_std_distances = np.std(cp_interp, axis=0)
    mip_mean_distances = np.mean(mip_interp, axis=0)
    mip_std_distances = np.std(mip_interp, axis=0)
    return (
        cp_mean_distances,
        cp_std_distances,
        mip_mean_distances,
        mip_std_distances,
        common_times,
    )


def compute_performance_profile(
    times_dict: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Given a dict of {method: times_array}, compute:
      - ratios[method]: array of ratios per instance
      - tau_values: sorted unique set of all ratios
      - profile[method]: fraction solved at each tau
    """
    methods = list(times_dict.keys())
    times = np.vstack([times_dict[m] for m in methods])  # shape (M, N)
    t_min = times.min(axis=0)  # shape (N,)
    ratios = {m: times_dict[m] / t_min for m in methods}

    # Choose tau grid: unique ratios sorted
    all_r = np.concatenate(list(ratios.values()))
    tau_vals = np.unique(np.sort(all_r))

    profile = {}
    N = times.shape[1]
    for m in methods:
        profile[m] = np.array([np.sum(ratios[m] <= tau) / N for tau in tau_vals])

    return tau_vals, profile


def plot_profile(tau, profile):
    plt.figure(figsize=(6, 4))
    for method, rho in profile.items():
        plt.step(tau, rho, where="post", label=method)
    plt.xlabel(r"Performance ratio $\tau$")
    plt.ylabel(r"$\rho(\tau)$: fraction of instances")
    plt.title("Performance Profile")
    plt.xlim(1, tau.max())
    plt.ylim(0, 1.02)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/profile.pdf")
    plt.close()


def scatter_plot(cp_times, mip_times):
    plt.figure(figsize=(5, 5))
    plt.scatter(cp_times, mip_times, alpha=0.6)
    # tracer la diagonale y=x
    lims = [min(cp_times.min(), mip_times.min()), max(cp_times.max(), mip_times.max())]
    plt.plot(lims, lims, linestyle="--", color="gray")
    plt.xlabel("Temps CP (s)")
    plt.ylabel("Temps MIP (s)")
    plt.title("Scatter CP vs MIP")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/scatter.pdf")
    plt.close()


def cactus_plot(cp_times, mip_times):
    plt.figure(figsize=(6, 4))
    N = len(cp_times)
    cp_sorted = np.sort(cp_times)
    mip_sorted = np.sort(mip_times)
    # plot: nombre d’instances (1..N) vs temps
    plt.step(cp_sorted, np.arange(1, N + 1), where="post", label="cp")
    plt.step(mip_sorted, np.arange(1, N + 1), where="post", label="mip")
    plt.xlabel("Temps (s)")
    plt.ylabel("Instances résolues")
    plt.title("Cactus Plot")
    plt.xscale("log")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/cactus.pdf")
    plt.close()


def times_ratio(cp_times, mip_times):
    ratios = cp_times / mip_times
    plt.figure(figsize=(6, 4))
    plt.hist(ratios, bins=30, alpha=0.7)
    plt.axvline(1, color="black", linestyle="--")
    plt.xlabel("Ratio CP/MIP")
    plt.ylabel("Nombre d’instances")
    plt.title("Histogramme des ratios de temps")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/ratio_histogram.pdf")
    plt.close()


def cactus_cdf_plot(cp_times, mip_times):
    plt.figure(figsize=(6, 4))
    for times, name in [(cp_times, "cp"), (mip_times, "mip")]:
        ts = np.sort(times)
        cdf = np.arange(1, len(ts) + 1) / len(ts)
        plt.step(ts, cdf, where="post", label=name)
    plt.xlabel("Temps (s)")
    plt.ylabel("Fraction cumulée")
    plt.title("Empirical CDF des temps")
    plt.xscale("log")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/cactus_cdf.pdf")
    plt.close()


def distance_plot(
    cp_mean_distances,
    cp_std_distances,
    mip_mean_distances,
    mip_std_distances,
    common_times,
):
    plt.figure(figsize=(6, 4))
    plt.plot(common_times, cp_mean_distances, label="cp")
    plt.fill_between(
        common_times,
        np.maximum(cp_mean_distances - cp_std_distances, 0),
        np.minimum(cp_mean_distances + cp_std_distances, 1),
        alpha=0.2,
    )
    plt.plot(common_times, mip_mean_distances, label="mip")
    plt.fill_between(
        common_times,
        np.maximum(mip_mean_distances - mip_std_distances, 0),
        np.minimum(mip_mean_distances + mip_std_distances, 1),
        alpha=0.2,
    )
    plt.xlabel("Temps (s)")
    plt.ylabel("Objective")
    plt.title("Objective vs Temps")
    plt.grid(True, ls="--", alpha=0.5)
    # plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/distance_plot.pdf")
    plt.close()


def remove_constant_parts(cp_mean_distances, mip_mean_distances):
    """
    Remove constant parts from the end of the distance arrays.
    Whenever the last value is the same as the second to last,
    we remove it until the last two values are different.
    """
    while 0.0 == cp_mean_distances[-2]:
        cp_mean_distances = cp_mean_distances[:-1]
    while 0.0 == mip_mean_distances[-2]:
        mip_mean_distances = mip_mean_distances[:-1]
    return cp_mean_distances, mip_mean_distances


def estimators_distance_plot(filename: str, seed: int | None = None):
    figure, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(N_ESTIMATORS)))
    for i, n_estimators in enumerate(N_ESTIMATORS[::-1]):
        cp_callbacks, mip_callbacks = load_callbacks(
            filename, n_estimators=n_estimators, seed=seed
        )
        (
            cp_mean_distances,
            _,
            mip_mean_distances,
            _,
            common_times,
        ) = aggregate_callbacks(cp_callbacks, mip_callbacks)
        cp_mean_distances, mip_mean_distances = remove_constant_parts(
            cp_mean_distances,
            mip_mean_distances,
        )
        ax.plot(
            common_times[: len(cp_mean_distances)],
            cp_mean_distances,
            label=f"cp_{n_estimators}",
            color=colors[len(N_ESTIMATORS) - 1 - i],
        )
        ax.plot(
            common_times[: len(mip_mean_distances)],
            mip_mean_distances,
            label=f"mip_{n_estimators}",
            linestyle="--",
            color=colors[len(N_ESTIMATORS) - 1 - i],
        )
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Objective")
    ax.set_title("Objective vs Temps for different n_estimators")
    ax.grid(True, ls="--", alpha=0.5)
    # ax.set_xscale("log")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/estimators_distance_plot.pdf")
    plt.close()


def depth_distance_plot(
    filename: str,
    n_estimators: int | None = None,
    seed: int | None = None,
):
    figure, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(MAX_DEPTHS)))
    for i, max_depth in enumerate(MAX_DEPTHS[::-1]):
        cp_callbacks, mip_callbacks = load_callbacks(
            filename,
            n_estimators=n_estimators,
            max_depth=max_depth,
            seed=seed,
        )
        (
            cp_mean_distances,
            _,
            mip_mean_distances,
            _,
            common_times,
        ) = aggregate_callbacks(cp_callbacks, mip_callbacks)
        cp_mean_distances, mip_mean_distances = remove_constant_parts(
            cp_mean_distances,
            mip_mean_distances,
        )
        ax.plot(
            common_times[: len(cp_mean_distances)],
            cp_mean_distances,
            label=f"cp_{max_depth}",
            color=colors[len(MAX_DEPTHS) - 1 - i],
        )
        ax.plot(
            common_times[: len(mip_mean_distances)],
            mip_mean_distances,
            label=f"mip_{max_depth}",
            linestyle="--",
            color=colors[len(MAX_DEPTHS) - 1 - i],
        )
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Objective")
    ax.set_title("Objective vs Temps for different max_depth")
    ax.grid(True, ls="--", alpha=0.5)
    ax.set_xscale("log")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/depth_distance_plot.pdf")
    plt.close()


def plot_times_vs_anything(
    filename: str,
    n_estimators: int | None = None,
    max_depth: int | None = None,
    seed: int | None = None,
):
    vs_estimators = n_estimators is None
    if n_estimators is not None:
        VS = MAX_DEPTHS
    elif max_depth is not None:
        VS = N_ESTIMATORS
    else:
        raise ValueError("At least one of n_estimators or max_depth must be provided")

    avg_cp_times = np.zeros(len(VS))
    avg_mip_times = np.zeros(len(VS))
    std_cp_times = np.zeros(len(VS))
    std_mip_times = np.zeros(len(VS))
    for i, vs in enumerate(VS):
        if vs_estimators:
            n_estimators = vs
            max_depth = max_depth
        else:
            n_estimators = n_estimators
            max_depth = vs
        cp_times, mip_times = load_times(
            filename, n_estimators=n_estimators, max_depth=max_depth, seed=seed
        )
        # print(
        #    f"Loaded {len(cp_times)} instances for n_estimators={n_estimators}, max_depth={max_depth}"
        # )
        avg_cp_times[i] = np.mean(cp_times)
        avg_mip_times[i] = np.mean(mip_times)
        std_cp_times[i] = np.std(cp_times)
        std_mip_times[i] = np.std(mip_times)
    print(
        f"Average CP times: {avg_cp_times}",
        f"\nStd CP times: {std_cp_times}",
        f"\nAverage MIP times: {avg_mip_times}",
        f"\nStd MIP times: {std_mip_times}",
    )

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(VS)), avg_cp_times, label="CP", marker="o")
    plt.fill_between(
        range(len(VS)),
        avg_cp_times - std_cp_times,
        avg_cp_times + std_cp_times,
        alpha=0.2,
    )
    plt.plot(range(len(VS)), avg_mip_times, label="MIP", marker="o", linestyle="--")
    plt.fill_between(
        range(len(VS)),
        avg_mip_times - std_mip_times,
        avg_mip_times + std_mip_times,
        alpha=0.2,
    )
    plt.xticks(range(len(VS)), VS if vs_estimators else VS[:-1] + ["None"])
    plt.xlabel("Number of Trees" if vs_estimators else "Max Depth")
    plt.ylabel("Average Time (s)")
    plt.yscale("log")
    plt.title(
        "Average Time vs " + ("Number of Trees" if vs_estimators else "Max Depth")
    )
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "plots/times_vs_" + ("estimators" if vs_estimators else "depth") + ".pdf"
    )
    plt.close()


def main():
    filename = "results/COMPAS1.json"
    cp_times, mip_times = load_times(filename)
    times_dict = {"cp": cp_times, "mip": mip_times}
    # scatter_plot(cp_times, mip_times)
    # cactus_plot(cp_times, mip_times)
    # times_ratio(cp_times, mip_times)
    # cactus_cdf_plot(cp_times, mip_times)

    tau, profile = compute_performance_profile(times_dict)
    plot_profile(tau, profile)

    cp_callbacks, mip_callbacks = load_callbacks(filename)
    (
        cp_mean_distances,
        cp_std_distances,
        mip_mean_distances,
        mip_std_distances,
        common_times,
    ) = aggregate_callbacks(cp_callbacks, mip_callbacks)
    distance_plot(
        cp_mean_distances,
        cp_std_distances,
        mip_mean_distances,
        mip_std_distances,
        common_times,
    )
    # estimators_distance_plot(filename, seed=42)
    # depth_distance_plot(filename, n_estimators=100)
    # plot_times_vs_anything(filename, n_estimators=100)
    # plot_times_vs_anything(filename, max_depth=5)


if __name__ == "__main__":
    main()
