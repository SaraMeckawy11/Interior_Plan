# Interior Plan Studio

The application turns a measured 2D floor plan into a furnished, first-person
3D walkthrough.

## Local AI 3D furniture

The **Local AI furniture** option keeps the complete generation pipeline on
this computer:

1. The existing local FLUX.2-klein model creates furniture reference images
   from each room's selected type, style, and design direction.
2. The open-source TripoSR model converts those references into actual GLB
   meshes.
3. Generated meshes are scaled to real-world dimensions, placed inside the
   measured floor plan, and cached under `room_designs/local_3d_assets`.

The first generation is slower because the assets must be created. Later
walkthroughs reuse the cache and start much faster. If a local mesh is missing
or generation fails, the walkthrough uses its procedural equivalent instead
of blocking the user.

Run the desktop application with `run_app.bat`.
