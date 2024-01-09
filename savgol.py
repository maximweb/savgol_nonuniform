import numpy as np


def _check_parameters(x, y, window_length, polyorder, deriv):
    if not isinstance(window_length, int) or (isinstance(window_length, float) and not window_length.is_integer()):
        raise ValueError("window_length must be an integer")

    if window_length < 2:
        raise ValueError("window_length must be at least 2")

    if not isinstance(polyorder, int) or (isinstance(polyorder, float) and not polyorder.is_integer()):
        raise ValueError("polyorder must be an integer")

    if polyorder < 0:
        raise ValueError("polyorder must be non-negative")

    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length")

    if not isinstance(deriv, int) or (isinstance(deriv, float) and not deriv.is_integer()):
        raise ValueError("deriv must be an integer")

    if deriv < 0:
        raise ValueError("deriv must be non-negative")

    if deriv > polyorder:
        raise ValueError("deriv must be less than or equal to polyorder")

    if np.asarray(x).ndim != 1 or np.asarray(y).ndim != 1:
        raise ValueError("x and y must be 1D arrays")

    if np.asarray(x).size != np.asarray(y).size:
        raise ValueError("x and y must have the same length")


def savgol_nonuniform(
    x: np.ndarray,
    y: np.ndarray,
    window_length: int,
    polyorder: int,
    deriv: int = 0,
) -> np.ndarray:
    """Savitzky-Golay filter for nonuniformly spaced independent data.

    The savgol_filter function in scipy.signal only supports uniformly spaced data in x.

    This implementation is based on the paper by Gorry 1991 [1], which describes a recursive calculation of the
    Savitzky-Golay filter coefficients based on orthogonal polynomials. This allows for a significant speedup compared
    to fitting each polynomial individually.

    Args:
        x (np.ndarray): 1D array of independent data
        y (np.ndarray): 1D array of data, same length as x
        window_length (int): Length of the window. Must be at least 2 and less than polyorder.
        polyorder (int): Order of the polynomial to fit. Must be non-negative and less than window_length.
        deriv (int, optional): Order of the derivative to compute. Must be non-negative and less than or equal to
            polyorder. Defaults to 0.

    Returns:
        np.ndarray: Filtered data, same length as x and y.
    """
    _check_parameters(x, y, window_length, polyorder, deriv)

    # == Prepare data and constants
    x = np.asarray(x)
    y = np.asarray(y)

    N_points = x.size

    if window_length % 2 != 0:
        N_start_edge = window_length // 2
        N_end_edge = window_length // 2
    else:
        N_start_edge = window_length // 2 - 1
        N_end_edge = window_length // 2

    # == Create windowed x matrix
    x_windowed = np.lib.stride_tricks.sliding_window_view(
        x,
        window_shape=(window_length),
    )

    # == Center x_windowed
    if window_length % 2 != 0:
        x_windowed_centered = x_windowed - x_windowed[:, window_length // 2, np.newaxis]  # [N_centers, window_length]
    else:
        x_windowed_centered = (
            x_windowed - np.mean(x_windowed[:, window_length // 2 - 1 : window_length // 2 + 1], axis=1)[:, np.newaxis]
        )

    # == Idenitfy identical windows
    x_windowed_centered_unique, i_unqiue_inverse = np.unique(x_windowed_centered, axis=0, return_inverse=True)
    N_centers_unique = x_windowed_centered_unique.shape[0]

    # == Calculate x^i matrix (eq. 19) only for the unique centered windows
    x_power_unique = np.power(
        x_windowed_centered_unique[:, :, np.newaxis], np.arange(0, 2 * polyorder + 2, 1)[np.newaxis, np.newaxis, :]
    )  # [N_centers, window_length, exponent]

    # == Calculate rolling sum of powered x for each exponent (eq. 19) only for the unique windows
    x_powersum_unique = np.sum(x_power_unique, axis=1)  # [N_centers_unique, exponent]
    x_powersum_unique = x_powersum_unique.transpose()  # [exponent, N_centers_unique]

    # == Reshape x_power_unique for facilitated P_k calculation
    x_power_unique = x_power_unique.transpose((0, 2, 1))  # [N_centers, exponent, window_length]

    # == Initialize A[k, r, N_centers] (eq. 14)
    A_kr = np.ones((polyorder + 1, polyorder + 1)) * np.nan
    A_kr = np.tril(A_kr, k=0)  # A_kr = 0 for r > k
    A_kr = np.tile(A_kr[..., np.newaxis], (1, 1, N_centers_unique))

    # == Initialize gamma[k, N_centers] (eq. 16)
    # We only need the last two gamma k for C_k, so we rotate index by modulo 2
    gamma_k = np.ones((2, N_centers_unique)) * np.nan

    # == Initialize B[k, N_centers] (eq. 17)
    #  we only need the last two, so we rotate index by modulo 2
    B_k = np.ones((2, N_centers_unique)) * np.nan

    # == Initialize C (eq. 12)
    # We only need C_{k-1} for A_k=f(C_{k-1})
    C_k = np.ones(N_centers_unique) * np.nan

    # == Initialize P[k, N_centers] (eq. 13)
    P_k = np.zeros((polyorder + 1, window_length, N_centers_unique)) * np.nan

    # == Initialize P_ks[k, N_points] (eq. 13)
    P_ks = np.zeros((polyorder + 1, N_points)) * np.nan

    # == Initialize h_{t,i}^{k,s} (eq. 10) weight factors
    # t: index of centered x_t, intrinsic in results
    # i: index within window
    # k: current polynomial order, recursively summed
    # s: derivative = const.
    h_ik = np.zeros((window_length, N_points))

    # == Prepare factorial coefficients
    if deriv == 0:
        factors = np.ones(polyorder - deriv + 1)
    else:
        factorials_numerator = np.asarray(
            [np.math.factorial(i + deriv) for i in np.arange(0, polyorder - deriv + 1, 1)]
        )
        factorials_denominator = np.asarray([np.math.factorial(i) for i in np.arange(0, polyorder - deriv + 1, 1)])
        factors = factorials_numerator / factorials_denominator

    # == Iterate over all polynomial orders for recursive parameter calculation
    for k in range(0, polyorder + 1):
        i_rolling2 = k % 2

        # == Update A_kr
        if k == 0:
            A_kr[0, 0, :] = 1.0
        elif k == 1:
            A_kr[1, 1, :] = 1
            A_kr[1, 0, :] = -B_k[0, :]
        else:
            A_kr[k, k, :] = 1
            A_kr[k, :k, :] = -B_k[k % 2 - 1, :] * A_kr[k - 1, :k, :] - C_k * A_kr[k - 2, :k, :]
            A_kr[k, 1:k, :] += A_kr[k - 1, 0 : k - 1, :]

        # == Update gamma_k
        # C(k) = f(gamma(k), gamma(k-1))
        # As we only need the last two gamma_k, we rotate index with k % 2
        if k == 0:
            # gamma_k[0, :] = window_length  # A_00 * sum_window x^0 = 1 * window_length
            gamma_k[i_rolling2, :] = window_length  # A_00 * sum_window x^0 = 1 * window_length
        else:
            # gamma_k[k, :] = np.sum(A_kr[k, 0 : k + 1, :] * x_powersum[k : 2 * k + 1, :], axis=0)
            gamma_k[i_rolling2, :] = np.sum(A_kr[k, 0 : k + 1, :] * x_powersum_unique[k : 2 * k + 1, :], axis=0)

        # == Update B_k
        # As we only need the last two B_k, we rotate index with k % 2
        if k == 0:
            # B_k[k, :] = x_powersum[1, :] / gamma_k[k, :]  # + A_{0,-1} = 0
            B_k[i_rolling2, :] = x_powersum_unique[1, :] / gamma_k[i_rolling2, :]  # + A_{0,-1} = 0
        else:
            B_k[i_rolling2, :] = (
                np.sum(A_kr[k, 0 : k + 1, :] * x_powersum_unique[k + 1 : 2 * k + 2, :], axis=0) / gamma_k[i_rolling2, :]
                + A_kr[k, k - 1, :]
            )

        # == Update C_k
        if k > 0:
            C_k = gamma_k[i_rolling2, :] / gamma_k[i_rolling2 - 1, :]

        # == Update P_k
        if k == 0:
            P_k[k, :, :] = 1  # A_00 = 1
        else:
            P_k[k, :, :] = np.sum(
                A_kr[k, : k + 1, np.newaxis, :] * x_power_unique.transpose((1, 2, 0))[: k + 1, :, :], axis=0
            )

        # == Update P_ks
        m = k - deriv

        if m < 0:
            P_ks[k, :] = 0
        elif m == 0:
            P_ks[k, N_start_edge:-N_end_edge] = factors[0] * A_kr[k, k, i_unqiue_inverse]

            # == Handle edges
            P_ks[k, : N_start_edge + 1] = factors[0] * A_kr[k, k, i_unqiue_inverse[0]]
            P_ks[k, -N_end_edge:] = factors[0] * A_kr[k, k, i_unqiue_inverse[-1]]
        else:
            P_ks[k, N_start_edge:-N_end_edge] = factors[0] * A_kr[k, deriv, i_unqiue_inverse]

            # == Handle edges
            P_ks[k, : N_start_edge + 1] = np.sum(
                factors[: m + 1, np.newaxis]
                * A_kr[k, deriv : k + 1, i_unqiue_inverse[0], np.newaxis]
                * x_power_unique[
                    i_unqiue_inverse[0],
                    : m + 1,
                    : N_start_edge + 1,
                ],
                axis=0,
            )
            P_ks[k, -N_end_edge:] = np.sum(
                factors[: m + 1, np.newaxis]
                * A_kr[k, deriv : k + 1, i_unqiue_inverse[-1], np.newaxis]
                * x_power_unique[i_unqiue_inverse[-1], : m + 1, -N_end_edge:],
                axis=0,
            )

        # == Update h_ik
        h_ik[:, N_start_edge:-N_end_edge] += (
            P_k[k, :, i_unqiue_inverse].transpose()  # TODO: why does indexing by array triggers transpose?
            * P_ks[k, N_start_edge:-N_end_edge]
            / gamma_k[i_rolling2, i_unqiue_inverse]
        )

        # == Handle edges
        h_ik[:, :N_start_edge] += (
            P_k[k, :, i_unqiue_inverse[0], np.newaxis]
            * P_ks[k, :N_start_edge]
            / gamma_k[i_rolling2, i_unqiue_inverse[0]]
        )
        h_ik[:, -N_end_edge:] += (
            P_k[k, :, i_unqiue_inverse[-1], np.newaxis]
            * P_ks[k, -N_end_edge:]
            / gamma_k[i_rolling2, i_unqiue_inverse[-1]]
        )

    # == Apply calculated weight factors h on data
    y_windowed = np.lib.stride_tricks.sliding_window_view(y, window_shape=window_length).transpose()
    y_filtered = np.sum(h_ik[:, N_start_edge:-N_end_edge] * y_windowed, axis=0)

    # -- Handle edges
    y_filtered_start_edge_unique = np.sum(h_ik[:, :N_start_edge] * y_windowed[:, 0, np.newaxis], axis=0)
    y_filtered_end_edge_unique = np.sum(h_ik[:, -N_end_edge:] * y_windowed[:, -1, np.newaxis], axis=0)

    # == Concatenate results with edges
    y_filtered = np.concatenate((y_filtered_start_edge_unique, y_filtered, y_filtered_end_edge_unique))

    return y_filtered
