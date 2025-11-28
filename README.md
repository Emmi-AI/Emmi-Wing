# Emmi-Wing



## OpenFOAM Template Setup

The simulation files are located in the `Template` directory. To prepare the case for execution, follow the steps below to import the geometry and configure the flow parameters.

### 1. Generate and Place Geometry
First, generate the wing geometry using the provided function:

`generate_wing_mesh(
    naca_digits,
    chord_root,
    span,
    taper_ratio,
    sweep,
    n_chord=640,
    n_span=380,
    close_root=True)`

This will output an `.stl` file. You must place this file into the surface directory of the template:

`Template/constant/triSurface/`

### 2. Configure Mesh Dictionaries
You need to link the generated `.stl` file to the meshing dictionaries. Open the following two files:

1.  `Template/system/surfaceFeatureExtractDict`
2.  `Template/system/snappyHexMeshDict`

In both files, locate the placeholder string **`NAME_OF_CAD`** and replace it with the actual name of your `.stl` file (e.g., `wing.stl`).

### 3. Set Velocity Components
Finally, define the initial velocity boundary conditions. Open the velocity file:

`Template/0.orig/U`

Update the internal field and inlet values by prescribing the velocity components **$U_x$** and **$U_y$**.
