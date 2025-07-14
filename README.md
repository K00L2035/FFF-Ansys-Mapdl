 # FFF Simulation Modules in Ansys Mapdl and Pyanys

To create a Fused Filament Fabrication (FFF) simulation in ANSYS using MAPDL or PyAnsys MAPDL, the following modules and features are commonly used:

- **PREP7**: Preprocessing for geometry, material properties, and meshing.
- **SOLUTION**: Applying loads, boundary conditions, and solving the simulation.
- **POST1**: Postprocessing for analyzing and visualizing results.
- **THERMAL**: Thermal analysis for simulating temperature distribution during printing.
- **STRUCTURAL**: Structural analysis for stress, strain, and deformation evaluation.

## Modules Used:
1. ansys.mapdl.core
2. imageio
3. matplotlib
4. pyvista


## Element-by-Element Activation

For FFF simulations, element-by-element activation is essential to mimic the layer-by-layer deposition process. In ANSYS MAPDL, this can be achieved using the `EALIVE` and `EKILL` commands to activate or deactivate elements during the simulation. With PyAnsys MAPDL, these commands can be scripted programmatically for automation and advanced control.

This is coupled field Trainsient Analysis with elemnent type "SOLID226"

**Example Workflow:**
1. Define geometry and mesh in PREP7.
2. Assign material properties (including temperature-dependent behavior).
3. Use element activation (`EALIVE`) to simulate the extrusion process layer by layer.
4. Apply thermal and structural loads as needed.
5. Solve using SOLUTION.
6. Analyse results in POST1.


Refer to the ANSYS MAPDL and PyAnsys MAPDL documentation for detailed commands, scripting examples, and workflow customization.
