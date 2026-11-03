import numpy as np
from stl import mesh


def parse_naca4(digits):
    """Return (max camber, camber position, thickness) as fractions of chord.

    Accepts the usual designation as a string or int ("0012", "NACA 2412",
    2412), four single digits ("0", "0", "1", "2"), or a (camber %, camber
    position in tenths, thickness %) triple such as (2, 4, 12). The last two
    digits of a 4-digit designation are the thickness, so "0012" is 12% thick.
    """
    if isinstance(digits, int):
        digits = f"{digits:04d}"
    if isinstance(digits, str):
        digits = digits.strip().upper()
        if digits.startswith("NACA"):
            digits = digits[4:].strip()
        if len(digits) != 4:
            raise ValueError(f"Not a NACA 4-digit designation: {digits!r}")
    if len(digits) == 4:
        digits = "".join(str(d) for d in digits)
        if not digits.isdigit():
            raise ValueError(f"Not a NACA 4-digit designation: {digits!r}")
        m, p, t = digits[0], digits[1], digits[2:]
    elif len(digits) == 3:
        m, p, t = digits
    else:
        raise ValueError(f"Not a NACA 4-digit designation: {digits!r}")
    return int(m) / 100.0, int(p) / 10.0, int(t) / 100.0


def naca4_airfoil_points(digits, n_points=100, te_thickness_ratio=1e-4):
    m, p, t = parse_naca4(digits)

    beta = np.linspace(0, np.pi, n_points)
    x = (1 - np.cos(beta)) / 2
    yt = (t / 0.2) * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4
    )

    if m == 0 or p == 0 or p == 1:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    else:
        yc = np.where(
            x < p,
            m / (p**2) * (2 * p * x - x**2),
            m / ((1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x - x**2),
        )
        dyc_dx = np.where(
            x < p, 2 * m / (p**2) * (p - x), 2 * m / ((1 - p) ** 2) * (p - x)
        )

    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    zu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    zl = yc - yt * np.cos(theta)

    # Trailing edge thickening
    te_x = 1.0
    te_gap = te_thickness_ratio
    xu[-1] = te_x - te_gap / 2
    zu[-1] = 0
    xl[-1] = te_x + te_gap / 2
    zl[-1] = 0

    x_coords = np.concatenate([xu[::-1], xl[1:]])
    z_coords = np.concatenate([zu[::-1], zl[1:]])
    return x_coords, z_coords


def generate_wing_mesh(
    naca_digits,
    chord_root,
    span,
    taper_ratio,
    sweep,
    n_chord=640,
    n_span=380,
    close_root=True,
):
    x, z = naca4_airfoil_points(naca_digits, n_chord)

    # Force the trailing edge to close: average last two points (upper + lower)
    x[-1] = x[0] = (x[0] + x[-1]) / 2
    z[-1] = z[0] = (z[0] + z[-1]) / 2

    n_profile = len(x)
    sweep = np.radians(sweep)
    chord_tip = taper_ratio * chord_root

    y_coords = np.linspace(0, span, n_span)
    profiles = []

    for i, y in enumerate(y_coords):
        chord = chord_root + (chord_tip - chord_root) * (y / span)
        sweep_offset = np.tan(sweep) * y

        x_scaled = x * chord + sweep_offset
        z_scaled = z * chord

        profile = np.stack([x_scaled, np.full_like(x_scaled, y), z_scaled], axis=1)
        profiles.append(profile)

    profiles = np.array(profiles)
    faces = []

    # Wing surface panels
    for i in range(n_span - 1):
        for j in range(n_profile - 1):
            p0 = profiles[i, j]
            p1 = profiles[i + 1, j]
            p2 = profiles[i + 1, j + 1]
            p3 = profiles[i, j + 1]
            faces.append([p0, p1, p2])
            faces.append([p0, p2, p3])

    # Close tip with triangles (fan)
    tip_profile = profiles[-1]
    tip_center = tip_profile.mean(axis=0)
    for j in range(n_profile - 1):
        faces.append([tip_profile[j], tip_profile[j + 1], tip_center])

    # Close root if needed
    if close_root:
        root_profile = profiles[0]
        root_center = root_profile.mean(axis=0)
        for j in range(n_profile - 1):
            faces.append([root_profile[j + 1], root_profile[j], root_center])

    faces = np.array(faces)
    wing_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    wing_mesh.vectors[:] = faces
    return wing_mesh
