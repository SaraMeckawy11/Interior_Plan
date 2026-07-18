# Interior Plan Studio

The application turns a measured 2D floor plan into a furnished, first-person
3D walkthrough.

## Preference-driven 3D walkthrough

The walkthrough is generated directly from the measured floor plan. The
procedural furniture engine is the instant, stable default. An optional local
TripoSR mode reconstructs primary furniture as AI-generated 3D meshes and
reuses them from the local cache, without a paid cloud service.

For a new design direction, the local image model first creates isolated
furniture references and TripoSR converts them to GLB meshes. The cache key
includes the style, design profile, color mood, personal brief, floor finish,
and wall finish, so different user preferences do not share the wrong assets.

The setup dialog controls the complete room direction:

- room type and style;
- airy, curated, or layered decoration;
- color mood and personal material/color notes;
- flooring such as oak, walnut, stone, concrete, terrazzo, or tile;
- wall treatments such as paint, limewash, timber slats, moulding, or concrete.

The renderer applies those choices to floors, focal walls, ceiling coves,
recessed lighting, sconces, curtains, coordinated furniture, rugs, artwork,
mirrors, plants, and room-specific decoration. Walls, doors, and windows remain
authoritative from the floor plan. TripoSR changes furniture geometry only, so
the whole-room layout, scale, finishes, lighting, decor, collision, and
circulation remain controlled by the same coordinated design engine.

Run the desktop application with `run_app.bat`.
